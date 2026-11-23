"""Scores an episodic forward-only run (CPU; labels): accuracy per (arm, cell) from the saved logits, 30-cell means, clean
harm, first-640-image (warm-up) accuracy and per-corruption gains over NEO. (Research name: scripts/spa/score_probe.py,
which wrote results/spa/spa1*.json.)

    python scripts/episodic/score_forward_only.py --runs runs/forward_only_design_vitb.json \\
        --imagenet-val $IMAGENET_VAL --out results/scores/forward_only_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402

CELLS = data.CELLS
GATE_GAIN, GATE_CLEAN = 4.0, 0.5       # the bar declared for the first forward-only probe (kept in the output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='manifest written by run_forward_only.py')
    ap.add_argument('--imagenet-val', required=True, help='ImageNet val in class-folder layout (labels)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--neo-from', default=None, metavar='SCORED_JSON',
                    help="scored forward-only file of the same split whose per_cell['neo'] is the NEO reference when the run has no neo arm (depth sweeps, hierarchical arms)")
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    index = data.ImageNetIndex(args.imagenet_val)
    arms = list(runs['runs'])
    run_cells = list(runs['runs'][arms[0]])     # the run's own cells: ImageNet-C by default, ImageNet-C-bar cells when run with --cells
    cells = [c for c in ['clean'] + CELLS if c in run_cells] + [c for c in run_cells if c != 'clean' and c not in CELLS]
    std = [c for c in cells if c != 'clean']
    acc, warm = {a: {} for a in arms}, {a: {} for a in arms}
    for a in arms:
        for c in cells:
            z = np.load(runs['runs'][a][c]['path']); y = z['labels'] if 'labels' in z.files else index.labels(z['index'])
            corr = z['logits'].astype(np.float32).argmax(1) == y
            acc[a][c] = float(corr.mean() * 100); warm[a][c] = float(corr[:640].mean() * 100)
    mean = {a: float(np.mean([acc[a][c] for c in std])) for a in arms}
    # the reference for gains: the run's own neo arm, else the neo arm of another scored file of the same split (--neo-from),
    # else the first arm (then the "vs NEO" columns are relative to that arm, and 'reference' says so)
    if 'neo' in arms:
        ref, reference = acc['neo'], 'neo'
    elif args.neo_from:
        ref, reference = json.loads(Path(args.neo_from).read_text())['per_cell']['neo'], f'neo from {Path(args.neo_from).name}'
    else:
        ref, reference = acc[arms[0]], arms[0]
    mean_ref = float(np.mean([ref[c] for c in std]))
    best = max(arms, key=lambda a: mean[a])
    clean_harm = {a: acc[a]['clean'] - ref['clean'] for a in arms} if 'clean' in cells else {}
    gate = {'best': best, 'gain_over_neo': mean[best] - mean_ref, 'clean_harm_best': clean_harm.get(best, 0.0),
            'verdict': 'PASS' if mean[best] - mean_ref >= GATE_GAIN and clean_harm.get(best, 0.0) >= -GATE_CLEAN else 'FAIL'}
    fam = {co: {a: float(np.mean([acc[a][f'{co}-3'], acc[a][f'{co}-5']]) - np.mean([ref[f'{co}-3'], ref[f'{co}-5']])) for a in arms}
           for co in data.CORRUPTIONS if f'{co}-3' in std and f'{co}-5' in std}
    out = {'gate': gate, 'reference': reference, 'n_cells': len(std), 'cells': std, 'mean_30_cells': mean, 'clean': {a: acc[a].get('clean') for a in arms}, 'clean_harm': clean_harm,
           'warmup_first640': {a: float(np.mean([warm[a][c] for c in std])) for a in arms}, 'per_family_vs_neo': fam, 'per_cell': acc}
    # early-exit arms record the fraction of images routed to the extra branches per cell (the manifest field 'routed_fraction')
    routed = {a: {c: runs['runs'][a][c]['routed_fraction'] for c in cells if 'routed_fraction' in runs['runs'][a][c]} for a in arms}
    routed = {a: r for a, r in routed.items() if r}
    if routed:
        out['routed_fraction'] = routed
        out['routed_fraction_mean'] = {a: float(np.mean([r[c] for c in std if c in r])) for a, r in routed.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    L = [f'# forward-only, {len(std)} cells x {runs["n_eval"]:,} images (reference: {reference})', '', '| arm | mean | vs NEO | clean | first 640 |', '|---|---|---|---|---|']
    for a in arms:
        L.append(f'| {a} | {mean[a]:.2f} | {mean[a] - mean_ref:+.2f} | {acc[a].get("clean", float("nan")):.2f} | {out["warmup_first640"][a]:.2f} |')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
