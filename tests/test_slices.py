"""The pair over three held-out image sets (docs/heldout_test_plan_slices.md): the two earlier sets hold every version of the
third set's run, scored once each."""
import json
from pathlib import Path

S = Path(__file__).resolve().parents[1] / 'results' / 'scores'
load = lambda n: json.loads((S / f'{n}.json').read_text())


def test_both_slices_hold_every_heldout3_arm():
    h3f, h3e = load('forward_only_heldout3_vitb'), load('eta_heldout3_vitb')
    for sl in ('heldout', 'heldout2'):
        f, e = load(f'forward_only_{sl}_vitb_two_site'), load(f'eta_{sl}_vitb_two_site')
        assert set(f['per_cell']) >= set(h3f['per_cell']) and set(e['per_cell']) >= set(h3e['per_cell'])
        for d in (f, e, h3f, h3e):
            assert all(len([c for c in d['per_cell'][a] if c != 'clean']) == 30 for a in d['per_cell'])
