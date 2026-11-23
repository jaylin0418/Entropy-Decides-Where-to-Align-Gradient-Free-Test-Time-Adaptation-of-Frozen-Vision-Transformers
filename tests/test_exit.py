"""The adaptive site choice (docs/heldout_test_plan_exit.md): the heldout4 files hold the eight frozen arms on 30 / 20 cells plus the
shared clean cell, carry the routed fraction of the early-exit arms, and the design files hold the arms of the plan."""
import json
from pathlib import Path

S = Path(__file__).resolve().parents[1] / 'results' / 'scores'
load = lambda n: json.loads((S / f'{n}.json').read_text())
FROZEN = ['neo', 'tok', 'place_b6', 'tok+place_b6', 'tok+place_b6+headcov', 'pair_gated', 'exit_b9_gated', 'exit_b5b9_gated']


def test_heldout4_runs_hold_the_frozen_arms_and_share_the_clean_cell():
    inc, cb = load('forward_only_heldout4_vitb_exit'), load('forward_only_heldout4_vitb_cbar_exit')
    assert list(inc['per_cell']) == FROZEN and list(cb['per_cell']) == FROZEN
    assert inc['n_cells'] == 30 and cb['n_cells'] == 20
    for a in FROZEN:
        assert inc['clean'][a] == cb['clean'][a], a          # the same 2,000 clean images of heldout4
    for f in (inc, cb):
        for a in ('exit_b9_gated', 'exit_b5b9_gated'):
            assert a in f['routed_fraction'] and 0.0 <= f['routed_fraction_mean'][a] <= 1.0
            assert f['routed_fraction'][a]['clean'] == 0.0     # the gate closed every clean batch


def test_design_exit_and_block_runs_hold_their_arms():
    for n in ('forward_only_design_vitb_exit', 'forward_only_design_vitb_cbar_exit'):
        assert set(load(n)['per_cell']) == {'exit_b9', 'exit_b5b9', 'exit_b5b9_gated', 'pair_gated'}
    for n in ('forward_only_design_vitb_select', 'forward_only_design_vitb_cbar_select'):
        arms = set(load(n)['per_cell'])
        assert {f'tok+place_b{k}' for k in range(1, 13) if k != 6} <= arms


def test_site_choice_files_match_the_scored_pair():
    for n, ref in (('site_choice_design_vitb', 'forward_only_design_vitb_two_site'), ('site_choice_design_vitb_cbar', 'forward_only_design_vitb_cbar')):
        sc, r = load(n), load(ref)
        assert abs(sc['pair'] - r['mean_30_cells']['tok+place_b6']) < 1e-6
        assert sc['oracle_per_cell'] >= sc['pair'] and sc['theta'] == 1.5


def test_range_runs_hold_the_gated_exit_arms():
    for n in ('forward_only_heldout_vitb_exit', 'forward_only_heldout2_vitb_exit', 'forward_only_heldout3_vitb_exit', 'forward_only_heldout3_vitb_cbar_exit'):
        f = load(n)
        assert set(f['per_cell']) == {'pair_gated', 'exit_b9_gated', 'exit_b5b9_gated'}
        assert f['routed_fraction']['exit_b5b9_gated']['clean'] <= 0.01   # the gate let at most one clean batch through on these slices


def test_controls_and_vits_runs_hold_their_arms():
    for n in ('forward_only_heldout4_vitb_exitctl', 'forward_only_heldout4_vitb_cbar_exitctl'):
        assert set(load(n)['per_cell']) == {'exit_b5b9_random_gated', 'exit_b5b9_equal_gated', 'mix_b5b9_gated'}
    assert set(load('forward_only_heldout3_vits_exit')['per_cell']) == {'pair_gated', 'exit_b9_gated', 'exit_b5b9_gated'}
    assert {'tok+place_b6', 'neo', 'exit_b5b9_gated'} <= set(load('forward_only_design_vits_exit')['per_cell'])


def test_stream_run_holds_the_three_arms():
    t = load('streams_heldout4_vitb_exit')['table']
    assert set(t) == {'neo', 'pair_gated', 'exit_b5b9_gated'}
    assert all(t[a]['cont'] is not None and t[a]['cleanret_clean'] is not None for a in t)


def test_label_free_choice_is_recorded_and_picks_the_most_accurate_trunk():
    for n in ('site_choice_design_vitb', 'site_choice_design_vitb_cbar'):
        lf = load(n)['label_free']
        assert lf['trunk_pick_by_entropy'] == lf['trunk_best_by_accuracy']
        assert lf['second_opinion_pick_gain'] <= lf['second_opinion_best_gain'] + 1e-9
