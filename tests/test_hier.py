"""CPU tests of the hierarchical (within/between-image) alignment in tokalign/aligner.py (no model, no data)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign.aligner import Aligner  # noqa: E402


class _TinyViT(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.embed_dim = d
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        self.norm = torch.nn.Identity()

    def forward(self, t):
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


def _spd(rng, d, scale=1.0):
    A = torch.tensor(rng.normal(size=(d, d)))
    return scale * (A @ A.T / d + 0.1 * torch.eye(d, dtype=torch.float64))


def _tokens(rng, n, T, mu, S_b, S_w):
    """n images of T tokens: image mean ~ N(mu, S_b), deviations ~ N(0, S_w) (centered per image)."""
    m = rng.multivariate_normal(mu.numpy(), S_b.numpy(), size=n)
    e = rng.multivariate_normal(np.zeros(len(mu)), S_w.numpy(), size=(n, T))
    e -= e.mean(1, keepdims=True)
    return torch.tensor(m[:, None, :] + e, dtype=torch.float32)


def _src(rng, d):
    S_b, S_w = _spd(rng, d, 1.0), _spd(rng, d, 1.0)
    mu = torch.tensor(rng.normal(size=d))
    return {'tok_mu': torch.zeros(768, dtype=torch.float64), 'tok_cov': torch.eye(768, dtype=torch.float64),
            'head_mu': torch.zeros(d, dtype=torch.float64), 'head_cov': torch.eye(d, dtype=torch.float64),
            'patch_mu': [mu, mu], 'patch_cov': [S_b + S_w, S_b + S_w], 'blk_mu': [mu, mu], 'blk_cov': [S_b + S_w] * 2,
            'patch_cov_between': [S_b, S_b], 'patch_cov_within': [S_w, S_w]}, mu, S_b, S_w


def _run(arm, src, t):
    m = _TinyViT(t.shape[-1])
    al = Aligner(m, src, dict(tok=False, mean=(), cov=(), head='none', **arm), 'cpu')
    seen = {}
    m.blocks[1].register_forward_hook(lambda mod, i, o: seen.__setitem__('out', o))
    with torch.no_grad():
        for _ in range(3):                                  # cumulative statistics over three passes of the batch
            m(t)
    al.remove()
    return seen['out'][:, 1:].double()


def _parts(u):
    m = u.mean(1); e = (u - m[:, None, :]).reshape(-1, u.shape[-1])
    return m.mean(0), torch.cov(m.T, correction=0), torch.cov(e.T, correction=0)


def test_hier_fixes_an_inflated_within_part_and_leaves_the_between_part():
    rng = np.random.default_rng(0); d, n, T = 6, 256, 32
    src, mu, S_b, S_w = _src(rng, d)
    t = torch.cat([torch.zeros(n, 1, d), _tokens(rng, n, T, mu, S_b, 9 * S_w)], 1)   # noise-like shift: within x9, between intact
    raw_mu, raw_b, raw_w = _parts(t[:, 1:].double())
    y = _run(dict(place=(1,), place_mode='hier', n0_between_images=16), src, t)
    out_mu, out_b, out_w = _parts(y)
    # the within part is brought back to the source (from x9), the between part stays close to the source
    assert (out_w - S_w).norm() < 0.25 * (raw_w - S_w).norm()
    assert (out_b - S_b).norm() < 1.5 * (raw_b - S_b).norm() + 0.1 * S_b.norm()
    # the pooled map on the same data distorts the between part more than the hierarchical one
    yp = _run(dict(place=(1,), n0_tok_images=1), src, t)
    _, pooled_b, _ = _parts(yp)
    assert (out_b - S_b).norm() < (pooled_b - S_b).norm()


def test_within_and_between_modes_touch_one_part_only():
    rng = np.random.default_rng(1); d, n, T = 5, 256, 32
    src, mu, S_b, S_w = _src(rng, d)
    t = torch.cat([torch.zeros(n, 1, d), _tokens(rng, n, T, mu + 2, 4 * S_b, 0.25 * S_w)], 1)
    _, raw_b, raw_w = _parts(t[:, 1:].double())
    _, b_w, w_w = _parts(_run(dict(place=(1,), place_mode='within', n0_between_images=16), src, t))
    _, b_b, w_b = _parts(_run(dict(place=(1,), place_mode='between', n0_between_images=16), src, t))
    assert (w_w - S_w).norm() < 0.4 * (raw_w - S_w).norm() and (b_w - raw_b).norm() < 0.05 * raw_b.norm()   # within: between kept
    assert (b_b - S_b).norm() < 0.5 * (raw_b - S_b).norm() and (w_b - raw_w).norm() < 0.05 * raw_w.norm()   # between: within kept


def test_hier_is_near_identity_on_source_like_tokens_and_recenters_the_mean():
    rng = np.random.default_rng(2); d, n, T = 6, 512, 32
    src, mu, S_b, S_w = _src(rng, d)
    t = torch.cat([torch.zeros(n, 1, d), _tokens(rng, n, T, mu + 3, S_b, S_w)], 1)
    y = _run(dict(place=(1,), place_mode='hier', n0_between_images=16), src, t)
    out_mu, out_b, out_w = _parts(y)
    assert (out_mu - mu).norm() < 0.1 * 3 * np.sqrt(d)           # the mean shift of 3 per dim is removed
    assert float((y - (t[:, 1:].double() - 3)).norm() / y.norm()) < 0.25   # otherwise close to the (recentered) input


def test_hier_statistics_are_detached_and_gradients_flow():
    rng = np.random.default_rng(3); d = 4
    src, mu, S_b, S_w = _src(rng, d)
    m = _TinyViT(d)
    al = Aligner(m, src, dict(tok=False, mean=(), cov=(), head='none', place=(0,), place_mode='hier'), 'cpu')
    t = torch.randn(8, 1 + 16, d, requires_grad=True)
    m(t).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert al.hier[0]['within'].s2.grad_fn is None and al.hier[0]['map'][0].grad_fn is None
    al.remove()


# --- the proposition behind the method (population statistics, closed form; docs/method_hier.md) -------------------

def _maps(S_b, S_w, S_b_t, S_w_t):
    """Pooled map and hierarchical maps (row-vector convention y = x M) for source parts (S_b, S_w) and target parts."""
    from tokalign.stages import sqrt_psd
    pooled = sqrt_psd(S_b_t + S_w_t, inv=True) @ sqrt_psd(S_b + S_w)
    M_b = sqrt_psd(S_b_t, inv=True) @ sqrt_psd(S_b)
    M_w = sqrt_psd(S_w_t, inv=True) @ sqrt_psd(S_w)
    return pooled, M_b, M_w


def test_proposition_hier_restores_both_parts_and_pooled_does_not():
    """A shift of the within-image part alone: the pooled map restores the total covariance but not the between part;
    the hierarchical maps restore both parts exactly (general, non-commuting matrices)."""
    rng, d = np.random.default_rng(3), 12
    S_b, S_w, D = _spd(rng, d), _spd(rng, d), _spd(rng, d, 3.0)
    pooled, M_b, M_w = _maps(S_b, S_w, S_b, S_w + D)
    tot_t, tot_s = S_b + S_w + D, S_b + S_w
    assert torch.allclose(pooled.T @ tot_t @ pooled, tot_s, atol=1e-4)           # pooled: total restored
    assert (pooled.T @ S_b @ pooled - S_b).norm() / S_b.norm() > 0.1               # pooled: between part distorted
    assert torch.allclose(M_b.T @ S_b @ M_b, S_b, atol=1e-4)                      # hier: M_b = I here
    assert torch.allclose(M_w.T @ (S_w + D) @ M_w, S_w, atol=1e-4)                # hier: within part restored


def test_proposition_pooled_map_shrinks_the_other_part_by_a_over_a_plus_delta():
    """Shared eigenbasis: with eigenvalues b, w of the parts and an inflation delta of the within part, the pooled map
    leaves the between part with variance b (b+w)/(b+w+delta) along each direction (compression where the noise is),
    and symmetrically an inflation gamma of the between part flattens the within part to w (b+w)/(b+w+gamma)."""
    rng, d = np.random.default_rng(4), 8
    Q, _ = torch.linalg.qr(torch.tensor(rng.standard_normal((d, d))))
    b, w, delta = (torch.tensor(rng.uniform(0.5, 2.0, d)) for _ in range(3))
    diag = lambda v: Q @ torch.diag(v) @ Q.T
    pooled, _, _ = _maps(diag(b), diag(w), diag(b), diag(w + delta))
    got = torch.diag(Q.T @ (pooled.T @ diag(b) @ pooled) @ Q)
    assert torch.allclose(got, b * (b + w) / (b + w + delta), atol=1e-6)
    assert (got < b).all()
    pooled2, _, _ = _maps(diag(b), diag(w), diag(b + delta), diag(w))
    got2 = torch.diag(Q.T @ (pooled2.T @ diag(w) @ pooled2) @ Q)
    assert torch.allclose(got2, w * (b + w) / (b + w + delta), atol=1e-6)


# --- library API: tokalign.stages.HierLayerAlign gives the same tokens as the runner's Aligner ----------------------

def test_stages_hier_layer_align_matches_the_aligner():
    from tokalign.stages import HierLayerAlign, StageConfig
    rng, d, T = np.random.default_rng(5), 6, 9
    src, mu, S_b, S_w = _src(rng, d)
    t = _tokens(rng, 8, T + 1, mu, 1.5 * S_b, 4.0 * S_w)      # shifted tokens; column 0 plays the CLS token
    for mode in ('hier', 'within', 'between'):
        ref = _run(dict(place=(1,), place_mode=mode), src, t)
        m = _TinyViT(d)
        la = HierLayerAlign(m, 2, src, StageConfig(), 'cpu', mode)
        seen = {}
        m.blocks[1].register_forward_hook(lambda mod, i, o: seen.__setitem__('out', o))
        with torch.no_grad():
            for _ in range(3):
                m(t)
        la.remove()
        assert torch.allclose(seen['out'][:, 1:].double(), ref, atol=1e-4), mode
        assert torch.allclose(seen['out'][:, 0], t[:, 0])                  # CLS untouched
