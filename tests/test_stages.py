"""CPU unit tests for the stages (no model, no data)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign.stages import HeadStage, Running, SpectralGate, Stack, StageConfig, dct_basis, fold, sqrt_psd, unfold  # noqa: E402


def _src(d_head=8, seed=0):
    rng = np.random.default_rng(seed)
    L = torch.tensor(rng.normal(size=(768, 768)) * 0.03)
    tok_cov = L @ L.T + 0.05 * torch.eye(768, dtype=torch.float64)
    Lh = torch.tensor(rng.normal(size=(d_head, d_head)))
    return {'tok_mu': torch.zeros(768, dtype=torch.float64), 'tok_cov': tok_cov,
            'head_mu': torch.zeros(d_head, dtype=torch.float64), 'head_cov': Lh @ Lh.T + torch.eye(d_head, dtype=torch.float64)}


def _images(src, n=8, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.multivariate_normal(np.zeros(768), src['tok_cov'].numpy(), size=n * 196) * scale
    return fold(torch.tensor(p, dtype=torch.float32), n, (224, 224))


def test_running_cumulative_and_ema():
    x = torch.tensor(np.random.default_rng(0).normal(size=(300, 5)))
    r = Running(5, 'cpu')
    for s in range(0, 300, 64):
        r.update(x[s:s + 64])
    assert torch.allclose(r.cov(), torch.tensor(np.cov(x.numpy(), rowvar=False, bias=True)), atol=1e-9)
    e = Running(5, 'cpu', ema=0.5)
    e.update(x[:64])
    assert torch.allclose(e.mean(), x[:64].mean(0))


def test_sqrt_psd_roundtrip():
    A = torch.tensor(np.random.default_rng(2).normal(size=(6, 6)))
    S = A @ A.T + 0.1 * torch.eye(6, dtype=torch.float64)
    assert torch.allclose(sqrt_psd(S) @ sqrt_psd(S, inv=True), torch.eye(6, dtype=torch.float64), atol=1e-8)


def test_unfold_fold_roundtrip_and_dct_orthonormal():
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(fold(unfold(x), 2, (224, 224)), x)
    C = dct_basis()
    assert np.allclose(C @ C.T, np.eye(768), atol=1e-10)


def test_tokalign_is_near_identity_on_source_data():
    src = _src()
    st = Stack(src, {'tok'}, StageConfig(), 8, 'cpu')
    x = _images(src, n=64)
    y = st.tokenizer(x)
    assert float((y - x).norm() / x.norm()) < 0.15


def test_gate_is_off_on_source_and_on_under_noise():
    src = _src()
    g = SpectralGate(src, StageConfig(), 'cpu')
    clean = unfold(_images(src, n=16))
    noisy = unfold(_images(src, n=16) + 2.0 * torch.randn(16, 3, 224, 224))
    assert g(clean) is False and g(noisy) is True


def test_head_stage_neo_and_cov():
    src = _src()
    h = torch.randn(64, 8) + 3.0
    neo = HeadStage(src, StageConfig(), 8, 'cpu', with_cov=False)
    assert torch.allclose(neo(h), (h - h.mean(0)).float(), atol=1e-5)
    cov = HeadStage(src, StageConfig(), 8, 'cpu', with_cov=True)
    assert cov(h).shape == (64, 8)
    off = HeadStage(src, StageConfig(), 8, 'cpu', with_cov=True)
    assert torch.allclose(off(h, cov_on=False), (h - h.mean(0)).float(), atol=1e-5)


class _TinyViT(torch.nn.Module):
    """Two blocks of d-dim tokens with one prefix token, enough to hook (no patch embedding; the input is the token tensor)."""

    def __init__(self, d=6):
        super().__init__()
        self.embed_dim = d
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, t):
        for b in self.blocks:
            t = b(t)
        return self.norm(t)[:, 0]


def _place_src(d=6, seed=3):
    from tokalign.stages import LayerAlign  # noqa: F401
    rng = np.random.default_rng(seed)
    A = torch.tensor(rng.normal(size=(d, d)))
    cov = A @ A.T + 0.1 * torch.eye(d, dtype=torch.float64)
    return {'patch_mu': [torch.zeros(d, dtype=torch.float64), torch.tensor(rng.normal(size=d))], 'patch_cov': [cov, 2 * cov]}


def test_layer_align_moves_tokens_toward_source_and_leaves_cls():
    from tokalign.stages import LayerAlign
    torch.manual_seed(0)
    m = _TinyViT().double(); src = _place_src()
    la = LayerAlign(m, 2, src, StageConfig(), 'cpu')
    t = torch.randn(4, 1 + 9, 6, dtype=torch.float64) * 3 + 1
    la.on = False
    with torch.no_grad():
        raw = m.blocks[1](m.blocks[0](t))                 # the block output with the alignment off
    la.on = True
    seen = {}
    m.blocks[1].register_forward_hook(lambda mod, i, o: seen.__setitem__('out', o))   # registered after LayerAlign: sees the aligned tokens
    with torch.no_grad():
        for _ in range(3):
            m(t)
    out = seen['out']
    assert torch.allclose(out[:, 0], raw[:, 0])          # CLS untouched
    patches = out[:, 1:].reshape(-1, 6)
    mu, cov = patches.mean(0), torch.cov(patches.T, correction=0)
    # after three cumulative batches the aligned tokens are closer to the block's source statistics than the raw ones
    rp = raw[:, 1:].reshape(-1, 6)
    assert (mu - src['patch_mu'][1]).norm() < (rp.mean(0) - src['patch_mu'][1]).norm()
    assert (cov - src['patch_cov'][1]).norm() < (torch.cov(rp.T, correction=0) - src['patch_cov'][1]).norm()
    la.on = False
    with torch.no_grad():
        m(t)
    assert torch.allclose(seen['out'], raw)               # gate off: identity
    la.remove()


def test_layer_align_statistics_are_detached_and_gradients_flow():
    from tokalign.stages import LayerAlign
    m = _TinyViT().double(); src = _place_src()
    la = LayerAlign(m, 1, src, StageConfig(), 'cpu')
    t = torch.randn(2, 1 + 4, 6, dtype=torch.float64, requires_grad=True)
    m(t).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert la.run.s2.grad_fn is None and la.map[0].grad_fn is None
    la.remove()


def test_stack_with_place_follows_the_gate():
    from tokalign.stages import LayerAlign  # noqa: F401
    src = _src(); src.update(_place_src(d=8))
    m = _TinyViT(8)
    st = Stack(src, {'tok', 'gate'}, StageConfig(), 8, 'cpu', model=m, place=(1, 2))
    assert len(st.layers) == 2 and all(la.on for la in st.layers)
    st.tokenizer(_images(src, n=2, scale=1.0))           # source-like batch: gate off
    assert not st.on and not any(la.on for la in st.layers)
    st.tokenizer(_images(src, n=2, scale=3.0))           # inflated covariance: gate on
    assert st.on and all(la.on for la in st.layers)
    st.remove()


def test_drift_log_is_zero_on_source_and_grows_with_variance():
    import importlib.util
    spec = importlib.util.spec_from_file_location('dbd', Path(__file__).resolve().parents[1] / 'scripts' / 'analysis' / 'drift_by_depth.py')
    dbd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dbd)
    d = 6
    m = torch.nn.Module(); m.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
    rng = np.random.default_rng(7)
    A = torch.tensor(rng.normal(size=(d, d))); cov = A @ A.T + 0.1 * torch.eye(d, dtype=torch.float64)
    src = {**_src(), 'patch_mu': [torch.zeros(d, dtype=torch.float64)] * 12, 'patch_cov': [cov] * 12}
    log = dbd.DriftLog(m, src, 'cpu')
    big = torch.tensor(rng.multivariate_normal(np.zeros(d), cov.numpy(), size=200000))
    spec, shift = log.measure('b3', big)
    assert spec < 0.02 and shift < 0.02                      # a large batch from the source distribution: no drift
    spec2, _ = log.measure('b3', 3 * big)
    assert abs(spec2 - np.log(9)) < 0.05                     # variance x9 along every direction: |log 9|
    log.new_batch(unfold(_images(src, n=2)))
    assert len(log.rows['pixel']) == 1 and log.rows['pixel'][0][0] < 0.5
    log.remove()
