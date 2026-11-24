"""CPU tests of the paper pipeline (no model, no data): the committed scores rebuild results/paper_numbers.json exactly,
paper/numbers.tex and results/number_inventory.csv are up to date with it, every \\pn key used in the report is defined,
the site-selection analysis rebuilds its committed score file, and the paper-run implementation (tokalign/aligner.py)
computes the same TokAlign map, block-site map and gate scores as the library (tokalign/stages.py)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokalign.aligner import Aligner  # noqa: E402
from tokalign.stages import SpectralGate, Stack, StageConfig, fold, unfold  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scores_rebuild_paper_numbers_exactly():
    agg = _module('make_paper_numbers', ROOT / 'scripts' / 'make_paper_numbers.py')
    committed = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())
    assert agg.compare(committed, agg.build()) == []


def test_numbers_tex_is_current_and_covers_the_report():
    mk = _module('make_numbers', ROOT / 'paper' / 'make_numbers.py')
    numbers = mk.build()
    assert (ROOT / 'paper' / 'numbers.tex').read_text() == mk.render(numbers)
    for tex in (ROOT / 'paper' / 'report' / 'report.tex',):
        used = set(re.findall(r'\\pn\{([^}]*)\}', tex.read_text()))
        assert used and used <= set(numbers), sorted(used - set(numbers))


def test_number_inventory_is_current():
    inv = _module('number_inventory', ROOT / 'scripts' / 'analysis' / 'number_inventory.py')
    assert (ROOT / 'results' / 'number_inventory.csv').read_text() == inv.render(inv.build())


def test_hier_section_covers_every_arm_site_and_cell():
    """The two-level study in paper_numbers.json: the eight arms of the design run against the pooled arms of the two-site
    run (same NEO reference), and the decomposition at all 13 sites for the 30 cells with both statistics."""
    H = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())['hier']
    arms = {'hier_px', 'hier_b6', 'hier_px+hier_b6', 'tok+hier_b6', 'within_b6', 'between_b6', 'hier_b6_n0x4', 'hier_b6_n0d4'}
    assert set(H['design_vitb']['episodic']) == arms and set(H['design_vitb']['clean']) == arms
    assert H['design_vitb']['reference'] == 'neo from forward_only_design_vitb_two_site.json'
    dec = H['decomp_vitb']
    assert dec['sites'] == ['pixel'] + [f'b{k}' for k in range(1, 13)] and len(dec['mean']) == 30
    for c, sites in dec['mean'].items():
        assert set(sites) == set(dec['sites'])
        assert all(set(v) == set(dec['columns']) for v in sites.values())
    # the pooled trace is the sum of the two parts, so a map on the sum cannot restore both when only one moved (the
    # proposition); the data show the case: at block 6 every corruption shrinks the between-image part and not the within
    b6 = {c: sites['b6'] for c, sites in dec['mean'].items()}
    assert all(v['between_logtrace'] < -0.1 and v['within_logtrace'] > -0.15 for v in b6.values())


def test_site_selection_rebuilds_exactly(tmp_path):
    """The model-free site-selection analysis reproduces its committed score file from the committed scores."""
    import subprocess
    sc = ROOT / 'results' / 'scores'
    out = tmp_path / 'site_selection.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'analysis' / 'site_selection.py'), '--two-site', str(sc / 'forward_only_design_vitb_two_site.json'),
                    '--sweep', str(sc / 'forward_only_design_vitb_sweep.json'), '--drift', str(sc / 'drift_by_depth_design_vitb.json'), '--out', str(out)],
                   check=True, capture_output=True)
    assert json.loads(out.read_text()) == json.loads((sc / 'site_selection_design_vitb.json').read_text())


class _Model:
    embed_dim = 8


def _src(seed=0):
    rng = np.random.default_rng(seed)
    L = torch.tensor(rng.normal(size=(768, 768)) * 0.03)
    return {'tok_mu': torch.zeros(768, dtype=torch.float64), 'tok_cov': L @ L.T + 0.05 * torch.eye(768, dtype=torch.float64),
            'head_cov': torch.eye(8, dtype=torch.float64)}


def test_aligner_tokalign_equals_library_stage():
    src = _src()
    rng = np.random.default_rng(1)
    al = Aligner(_Model(), src, dict(tok=True, mean=(), cov=(), head='none'), 'cpu')
    st = Stack(src, {'tok'}, StageConfig(), 8, 'cpu')
    for b in range(3):
        p = rng.multivariate_normal(np.zeros(768), src['tok_cov'].numpy(), size=4 * 196) * (1.5 + b)
        x = fold(torch.tensor(p, dtype=torch.float32), 4, (224, 224))
        assert torch.allclose(al.tokenizer(x), st.tokenizer(x), atol=1e-4)


def test_aligner_gate_scores_equal_library_gate():
    src = _src()
    al = Aligner(_Model(), src, dict(tok=True, mean=(), cov=(), head='none', gate={'spec': 0.4, 'mean': 0.3}), 'cpu')
    g = SpectralGate(src, StageConfig(), 'cpu')
    x = torch.randn(4, 3, 224, 224)
    spec, shift = al.drift(unfold(x))
    g(unfold(x))
    assert abs(spec - g.log[-1][0]) < 1e-6 and abs(shift - g.log[-1][1]) < 1e-9


class _Blocks(torch.nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.embed_dim = d
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(2)])
        self.norm = torch.nn.Identity()

    def forward(self, t):
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


def test_aligner_place_equals_library_layer_align():
    """The paper-run block-site alignment (Aligner ``place``) and the library LayerAlign give the same tokens, batch by batch."""
    from tokalign.stages import LayerAlign
    rng = np.random.default_rng(5)
    A = torch.tensor(rng.normal(size=(6, 6)))
    cov = A @ A.T + 0.1 * torch.eye(6, dtype=torch.float64)
    src = {**_src(), 'patch_mu': [torch.zeros(6, dtype=torch.float64), torch.tensor(rng.normal(size=6))], 'patch_cov': [cov, 2 * cov],
           'head_cov': torch.eye(6, dtype=torch.float64)}
    torch.manual_seed(1)
    m1 = _Blocks().double(); m2 = _Blocks().double(); m2.load_state_dict(m1.state_dict())
    al = Aligner(m1, src, dict(tok=False, mean=(), cov=(), head='none', place=(1,)), 'cpu')
    la = LayerAlign(m2, 2, src, StageConfig(), 'cpu')
    for b in range(3):
        t = torch.randn(3, 1 + 5, 6, dtype=torch.float64) * (1 + b)
        with torch.no_grad():
            assert torch.allclose(m1(t), m2(t), atol=1e-9)
    t = torch.randn(3, 1 + 5, 6, dtype=torch.float64, requires_grad=True)
    m1(t).sum().backward()                                 # a gradient host's backward passes through the map
    assert torch.isfinite(t.grad).all() and al.run_place[1].s2.grad_fn is None
    al.remove(); la.remove()


def test_block6_choice_is_stable_under_leave_one_out():
    """The block site of the pair is chosen on the design sweep; choosing it again with any corruption family or any one
    corruption left out of the sweep gives block 6 every time (report Section 3 and Section B of the supplement)."""
    import numpy as np
    T = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())['two_site']['sweep_vitb']
    cells = [c for c in T['neo_per_cell'] if c != 'clean']
    blocks = [f'b{k}' for k in range(1, 13)]
    gain = lambda s, cs: np.mean([T['frozen_per_cell'][s][c] - T['neo_per_cell'][c] for c in cs])
    corrs = sorted({c.rsplit('-', 1)[0] for c in cells})
    assert len(corrs) == 15
    for co in corrs:
        assert max(blocks, key=lambda s: gain(s, [c for c in cells if c.rsplit('-', 1)[0] != co])) == 'b6'
    fams = {'noise': ('gaussian_noise', 'shot_noise', 'impulse_noise'), 'blur': ('defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'),
            'weather': ('snow', 'frost', 'fog', 'brightness'), 'digital': ('contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')}
    for cos in fams.values():
        assert max(blocks, key=lambda s: gain(s, [c for c in cells if c.rsplit('-', 1)[0] not in cos])) == 'b6'


def test_extra_timing_run_has_the_two_site_arms():
    """The wall-clock run with the two-site stack (report Table S6, lower block): six arms on six cells, NEO at 1.0x, and
    the two-site stack at k=16 within 1.35x of NEO (the expectation of docs/heldout_test_plan_extra.md)."""
    T = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())['extra']['timing']
    assert set(T) == {'neo', 'stack_k16_fp32_stats32', 'twosite_k16_fp32_stats32', 'twosite_k1', 'eta', 'eta+twosite'}
    assert T['neo']['x_neo'] == 1.0 and T['twosite_k16_fp32_stats32']['x_neo'] <= 1.35
    assert T['twosite_k16_fp32_stats32']['acc_6cells'] > T['stack_k16_fp32_stats32']['acc_6cells'] > T['neo']['acc_6cells']


def test_sar_design_run_has_the_two_site_arms():
    """The SAR design run with the two-site arms (report Table S9): twelve arms on 30 cells, the two sites at least 0.3 points
    above TokAlign+SAR and within 0.5 points of SAR on the clean cell (the design bar of docs/heldout_test_plan_extra.md)."""
    P = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())
    pc, d = P['extra']['sar_design']['per_cell'], P['design_vitb']
    assert P['extra']['sar_design']['n_cells'] == 30 and len(pc) == 12
    assert {'sar', 'tok+sar', 'sar+place_b6', 'tok+sar+place_b6', 'tok+sar+place_b6+neo'} <= set(pc) <= set(d['episodic'])
    assert d['episodic']['tok+sar+place_b6'] >= d['episodic']['tok+sar'] + 0.3
    assert d['clean']['tok+sar+place_b6'] >= d['clean']['sar'] - 0.5


def test_sar_heldout_run_has_the_two_site_arms():
    """The heldout3 SAR run (report Table S9, lower block): the same twelve arms on 30 cells, the two sites at least 0.3 points
    above TokAlign+SAR (the margin of docs/heldout_test_plan_extra.md, as for ETA)."""
    H = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())['extra']['sar_held3']
    assert H['n_cells'] == 30 and len(H['per_cell']) == 12 and set(H['episodic']) == set(H['per_cell'])
    assert H['episodic']['tok+sar+place_b6'] >= H['episodic']['tok+sar'] + 0.3


def test_imagenet_c_bar_runs_cover_the_ten_corruptions():
    """ImageNet-C-bar (report Table S10): both splits run the four forward-only arms on the 20 cells of tokalign.data.CBAR_CELLS
    plus clean, and the design run meets the bar of docs/heldout_test_plan_extra.md (pair >= NEO + 1.0 and >= best single + 0.3)."""
    from tokalign import data
    CB = json.loads((ROOT / 'results' / 'paper_numbers.json').read_text())['extra']['cbar']
    for S in ('design', 'held3'):
        assert CB[S]['n_cells'] == 20 and CB[S]['cells'] == data.CBAR_CELLS
        assert set(CB[S]['episodic']) == {'neo', 'tok', 'place_b6', 'tok+place_b6'}
    ep = CB['design']['episodic']
    assert ep['tok+place_b6'] >= ep['neo'] + 1.0 and ep['tok+place_b6'] >= max(ep['tok'], ep['place_b6']) + 0.3
