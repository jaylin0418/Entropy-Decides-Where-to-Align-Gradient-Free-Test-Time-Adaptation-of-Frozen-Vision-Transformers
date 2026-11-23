"""Which site to align, per cell (CPU; no model, no labels beyond the scored files): from the scored depth sweep, the
two-site design run and the drift-by-depth scores of the same split, the mean accuracy of fixed sites, of the two-site
combination, of two online selection rules that pick the site from the drift of the first K batches, and of the per-cell
oracles. Writes results/scores/site_selection_<split>_<model>.json.

    python scripts/analysis/site_selection.py --two-site results/scores/forward_only_design_vitb_two_site.json \\
        --sweep results/scores/forward_only_design_vitb_sweep.json --drift results/scores/drift_by_depth_design_vitb.json \\
        --out results/scores/site_selection_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13))


def arm_of(site):
    return 'tok' if site == 'pixel' else f'place_{site}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--two-site', required=True, help='scored forward-only run with the arms neo, tok, place_b6, tok+place_b6')
    ap.add_argument('--sweep', required=True, help='scored forward-only run with the arms place_b1 .. place_b12 (b6 may be missing)')
    ap.add_argument('--drift', required=True, help='scored drift-by-depth run of the same split (score_drift_by_depth.py)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    two, sweep, drift = (json.loads(Path(p).read_text()) for p in (args.two_site, args.sweep, args.drift))
    cells = [c for c in two['per_cell']['neo'] if c != 'clean']
    acc = {s: {**sweep['per_cell'].get(arm_of(s), {}), **two['per_cell'].get(arm_of(s), {})} for s in SITES}   # the two-site run wins where both hold the arm
    for s in SITES:
        assert all(c in acc[s] for c in cells), f'site {s}: cells missing'
    neo, both = two['per_cell']['neo'], two['per_cell']['tok+place_b6']
    z = {c: {s: drift['cells'][c]['frozen'][s]['spec']['z_first_k'] for s in SITES} for c in cells}
    rules = {
        'pixel': {c: 'pixel' for c in cells},
        'b6': {c: 'b6' for c in cells},
        'select_pixel_or_b6_by_drift': {c: 'pixel' if z[c]['pixel'] >= z[c]['b6'] else 'b6' for c in cells},
        'select_site_of_largest_drift': {c: max(SITES, key=lambda s: z[c][s]) for c in cells},
        'oracle_pixel_or_b6': {c: max(('pixel', 'b6'), key=lambda s: acc[s][c]) for c in cells},
        'oracle_any_site': {c: max(SITES, key=lambda s: acc[s][c]) for c in cells},
    }
    mean = lambda f: float(np.mean([f(c) for c in cells]))
    out = {'k_batches': drift['k'], 'statistic': 'z of the spectral drift of the frozen tokens over the first k batches, against the clean cell',
           'mean_30_cells': {'neo': mean(lambda c: neo[c]), 'two_site': mean(lambda c: both[c]),
                             **{r: mean(lambda c, r=r: acc[rules[r][c]][c]) for r in rules}},
           'gain_over_neo': {}, 'chosen_site': rules,
           'site_counts': {r: {s: sum(v == s for v in rules[r].values()) for s in SITES if any(v == s for v in rules[r].values())} for r in rules},
           'per_cell': {'neo': {c: neo[c] for c in cells}, 'two_site': {c: both[c] for c in cells}, 'sites': {s: {c: acc[s][c] for c in cells} for s in SITES}}}
    out['gain_over_neo'] = {k: v - out['mean_30_cells']['neo'] for k, v in out['mean_30_cells'].items() if k != 'neo'}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print('| rule | mean | vs NEO |\n|---|---|---|')
    for k, v in out['mean_30_cells'].items():
        print(f'| {k} | {v:.2f} | {v - out["mean_30_cells"]["neo"]:+.2f} |')
    print('sites chosen:', json.dumps(out['site_counts']))


if __name__ == '__main__':
    main()
