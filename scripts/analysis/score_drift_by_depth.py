"""Scores a drift-by-depth run (CPU; no labels): for every corrupted cell, site and pass, the mean of the two drift
statistics over the first K batches and over all batches, and their z-scores against the clean cell's batches of the
same split (mean and sd over the clean batches, frozen pass). Also the site of the largest z per cell. Writes
results/scores/drift_by_depth_<split>_<model>.json.

    python scripts/analysis/score_drift_by_depth.py --runs runs/drift_by_depth_design_vitb.json \\
        --out results/scores/drift_by_depth_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

K = 5          # batches a stream needs before an online rule could act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rec = json.loads(Path(args.runs).read_text())
    z = np.load(rec['path'])
    sites, cells = list(rec['sites']), [c for c in rec['cells'] if c != 'clean']
    stat = ('spec', 'shift')
    norm = {s: [(float(z[f'drift_frozen_clean_{s}'][:, j].mean()), float(z[f'drift_frozen_clean_{s}'][:, j].std())) for j in range(2)] for s in sites}
    out = {'k': K, 'sites': sites, 'clean_norm': {s: {stat[j]: {'mean': norm[s][j][0], 'sd': norm[s][j][1]} for j in range(2)} for s in sites}, 'cells': {}}
    for c in cells:
        out['cells'][c] = {}
        for pas in ('frozen', 'tok'):
            d = {}
            for s in sites:
                a = z[f'drift_{pas}_{c}_{s}']
                d[s] = {stat[j]: {'first_k': float(a[:K, j].mean()), 'all': float(a[:, j].mean()),
                                  'z_first_k': float((a[:K, j].mean() - norm[s][j][0]) / norm[s][j][1]),
                                  'z_all': float((a[:, j].mean() - norm[s][j][0]) / norm[s][j][1])} for j in range(2)}
            d['argmax_z_spec_first_k'] = max(sites, key=lambda s: d[s]['spec']['z_first_k'])
            out['cells'][c][pas] = d
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    L = ['| cell | ' + ' | '.join(sites) + ' |', '|---' * (len(sites) + 1) + '|']
    for c in cells:
        L.append(f'| {c} | ' + ' | '.join(f"{out['cells'][c]['frozen'][s]['spec']['z_first_k']:.0f}" for s in sites) + ' |')
    print('z of spec drift, frozen pass, first 5 batches\n' + '\n'.join(L))


if __name__ == '__main__':
    main()
