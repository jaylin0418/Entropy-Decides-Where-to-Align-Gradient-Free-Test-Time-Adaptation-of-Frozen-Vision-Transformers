"""Scores a decompose-by-depth run (CPU; no labels): per corrupted cell and site, the mean of each statistic over all
batches and its z-score against the clean cell's batches of the same split. Prints the within and between log trace
ratios at the pixel site and block 6. Writes results/scores/decompose_by_depth_<split>_<model>.json.

    python scripts/analysis/score_decompose_by_depth.py --runs runs/decompose_by_depth_design_vitb.json \\
        --out results/scores/decompose_by_depth_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rec = json.loads(Path(args.runs).read_text())
    z = np.load(rec['path'])
    sites, cols = list(rec['sites']), list(rec['columns'])
    cells = [c for c in rec['cells'] if c != 'clean']
    norm = {s: (z[f'dec_clean_{s}'].mean(0), z[f'dec_clean_{s}'].std(0) + 1e-12) for s in sites}
    out = {'sites': sites, 'columns': cols, 'clean': {s: {c: float(norm[s][0][j]) for j, c in enumerate(cols)} for s in sites}, 'cells': {}}
    for c in cells:
        out['cells'][c] = {}
        for s in sites:
            a = z[f'dec_{c}_{s}']
            out['cells'][c][s] = {col: {'mean': float(a[:, j].mean()), 'z': float((a[:, j].mean() - norm[s][0][j]) / norm[s][1][j])}
                                  for j, col in enumerate(cols)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    L = ['| cell | within px | between px | within b6 | between b6 |', '|---|---|---|---|---|']
    for c in cells:
        d = out['cells'][c]
        L.append(f"| {c} | {d['pixel']['within_logtrace']['mean']:+.2f} | {d['pixel']['between_logtrace']['mean']:+.2f} | "
                 f"{d['b6']['within_logtrace']['mean']:+.2f} | {d['b6']['between_logtrace']['mean']:+.2f} |")
    print('log trace ratio to the source (within-image / between-image), all batches\n' + '\n'.join(L))


if __name__ == '__main__':
    main()
