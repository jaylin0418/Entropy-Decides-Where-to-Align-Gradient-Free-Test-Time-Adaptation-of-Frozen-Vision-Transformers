"""Scores a wall-clock run (CPU; labels): ms per batch per arm (mean over cells), ratio to NEO, and accuracy on the
benchmark cells from the saved logits. (Research name: scripts/spa/score_bench.py, which wrote
results/spa/bench_summary*.json.)

    python scripts/cost/score_timing.py --runs runs/timing_design_vitb.json --imagenet-val $IMAGENET_VAL \\
        --out results/scores/timing_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())['runs']
    index = data.ImageNetIndex(args.imagenet_val)
    res = {}
    for arm, cells in runs.items():
        ms = float(np.mean([c['ms_per_batch_mean'] for c in cells.values()]))
        acc = []
        for c in cells.values():
            z = np.load(c['path']); acc.append(float((z['logits'].astype(np.float32).argmax(1) == index.labels(z['index'])).mean() * 100))
        res[arm] = {'ms_per_batch': ms, 'x_neo': None, 'acc_cells': float(np.mean(acc)), 'n_cells': len(cells)}
    for arm in res:
        res[arm]['x_neo'] = res[arm]['ms_per_batch'] / res['neo']['ms_per_batch']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    for arm, r in res.items():
        print(f"{arm:28s} {r['ms_per_batch']:7.1f} ms  {r['x_neo']:.3f}x  acc {r['acc_cells']:.2f} ({r['n_cells']} cells)")


if __name__ == '__main__':
    main()
