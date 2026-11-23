"""Dynamic use of the alignment sites, evaluated exactly from the saved per-image logits of the forward-only runs (CPU, no
model): every arm ran independently from batch 1 of a cell, so any label-free combination or choice among their outputs is
what a parallel implementation outputs on the same images.

Per image, the branch mixture averages the softmax outputs of several arms with entropy weights w_a ∝ exp(-H_a / tau), where
H_a is the prediction entropy of arm a on that image (tau = 1). The pair 'tok+place_b6' and the pixel-only arm 'tok' share
the pixel map and blocks 1 to 6, so their mixture is one forward pass with a branch after block 6 (1.5 passes). The warm-up
gate runs the arms on the first K batches, picks the arm with the lowest mean entropy and outputs the pair during the warm-up.

    python scripts/analysis/site_mixture.py --logits runs/logits --run forward_only_heldout3_vitb --imagenet-val $IMAGENET_VAL \\
        --cells inc --out results/scores/site_mixture_heldout3_vitb.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402

ARMS = ('neo', 'tok', 'place_b6', 'tok+place_b6')
MIXTURES = {'branch': ('tok', 'tok+place_b6'), 'three': ('tok', 'place_b6', 'tok+place_b6'), 'four': ARMS}
BATCH, K_WARMUP, TAU = 64, 3, 1.0


def softmax(lg):
    p = np.exp(lg - lg.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


def mixture(P, arms, tau=TAU):
    H = np.stack([entropy(P[a]) for a in arms]); w = np.exp(-H / tau); w /= w.sum(0, keepdims=True)
    return np.einsum('an,anc->nc', w, np.stack([P[a] for a in arms]))


def warmup_gate(P, arms, k=K_WARMUP, default='tok+place_b6'):
    n = k * BATCH; a = min(arms, key=lambda a: entropy(P[a][:n]).mean())
    return np.concatenate([P[default][:n].argmax(1), P[a][n:].argmax(1)]), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits', required=True, help='directory of <run>_<arm>_<cell>.npz files')
    ap.add_argument('--run', required=True, help='run name, e.g. forward_only_heldout3_vitb')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--cells', choices=['inc', 'cbar'], default='inc')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    index = data.ImageNetIndex(args.imagenet_val)
    cells = ['clean'] + ([f'{c}-{s}' for c in data.CORRUPTIONS for s in (3, 5)] if args.cells == 'inc' else list(data.CBAR_CELLS))
    per_cell = {}
    for c in cells:
        P, y = {}, None
        for a in ARMS:
            z = np.load(Path(args.logits) / f'{args.run}_{a}_{c}.npz'); P[a] = softmax(z['logits'].astype(np.float32))
            yy = z['labels'] if 'labels' in z.files else np.array([index.label(int(i)) for i in z['index']]); assert y is None or (yy == y).all(); y = yy
        r = {a: float((P[a].argmax(1) == y).mean() * 100) for a in ARMS}
        for m, arms in MIXTURES.items():
            r[f'mix_{m}'] = float((mixture(P, arms).argmax(1) == y).mean() * 100)
            r[f'mean_{m}'] = float((np.mean([P[a] for a in arms], 0).argmax(1) == y).mean() * 100)
        pred, chosen = warmup_gate(P, ARMS); r['gate_k3'] = float((pred == y).mean() * 100); r['gate_k3_chosen'] = chosen
        r['oracle_arm'] = max(r[a] for a in ARMS)
        per_cell[c] = r
    keys = [k for k in per_cell['clean'] if k != 'gate_k3_chosen']
    corrupted = [c for c in cells if c != 'clean']
    out = {'run': args.run, 'cells': corrupted, 'tau': TAU, 'k_warmup': K_WARMUP, 'batch': BATCH, 'mixtures': MIXTURES,
           'mean_corrupted': {k: float(np.mean([per_cell[c][k] for c in corrupted])) for k in keys},
           'clean': {k: per_cell['clean'][k] for k in keys}, 'per_cell': per_cell}
    out['gain_over_pair'] = {k: out['mean_corrupted'][k] - out['mean_corrupted']['tok+place_b6'] for k in keys}
    out['clean_delta_vs_pair'] = {k: out['clean'][k] - out['clean']['tok+place_b6'] for k in keys}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(f"{args.run}: {len(corrupted)} corrupted cells; pair {out['mean_corrupted']['tok+place_b6']:.2f}, clean {out['clean']['tok+place_b6']:.2f}")
    for k in keys:
        if k != 'tok+place_b6':
            print(f"  {k:18s} {out['mean_corrupted'][k]:.2f} ({out['gain_over_pair'][k]:+.2f})  clean {out['clean'][k]:.2f} ({out['clean_delta_vs_pair'][k]:+.2f})")


if __name__ == '__main__':
    main()
