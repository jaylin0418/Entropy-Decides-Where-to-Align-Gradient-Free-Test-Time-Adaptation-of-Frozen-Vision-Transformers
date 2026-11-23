"""Scores an episodic gradient-host run (CPU; labels): each arm raw, with NEO and with NEO + head covariance (source
prior, n0 = 256, cumulative within the cell, current batch included), both applied to the recorded head input.
(Research name: scripts/spa/score_eta.py, which wrote results/spa/spa2*.json.)

    python scripts/episodic/score_gradient_host.py --runs runs/eta_heldout_vitb.json --stats stats/vitb.pt \\
        --imagenet-val $IMAGENET_VAL --out results/scores/eta_heldout_vitb.json \\
        [--forward-only results/scores/forward_only_heldout_vitb.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data, models  # noqa: E402

CELLS = data.CELLS
BATCH, N0, EPS = 64, 256, 1e-5


def sqrt_psd(S, inv=False):
    w, V = np.linalg.eigh(S); w = np.clip(w, EPS, None)
    return (V / np.sqrt(w)) @ V.T if inv else (V * np.sqrt(w)) @ V.T


def head_variants(h, W, b, y, S_s_half, S_s, keep=None, headcov=True):
    """Per-image correctness for raw, NEO and NEO + source-prior covariance, cumulative online, batch included. ``keep``: the
    classes kept by a 200-class dataset (the other logits are masked before the argmax, as in the run). ``headcov=False`` skips
    the covariance variant (one d x d eigendecomposition per batch, the cost of this script; the report does not use it for
    the gradient hosts) and returns None for it."""
    n, d = h.shape
    if keep is not None:
        W, b = W[keep], b[keep]
        y = np.searchsorted(keep, y)
    raw = (h.astype(np.float32) @ W.T + b).argmax(1) == y
    neo, cov = np.empty(n, bool), np.empty(n, bool)
    s1, s2, cnt = np.zeros(d), np.zeros((d, d)), 0
    for st in range(0, n, BATCH):
        hb = h[st:st + BATCH].astype(np.float64); s1 += hb.sum(0); s2 += hb.T @ hb; cnt += len(hb)
        mu = s1 / cnt
        neo[st:st + BATCH] = ((hb - mu) @ W.T + b).argmax(1) == y[st:st + BATCH]
        if not headcov:
            continue
        Sn = s2 / cnt - np.outer(mu, mu)
        S = (cnt * Sn + N0 * S_s) / (cnt + N0)
        cov[st:st + BATCH] = (((hb - mu) @ sqrt_psd(S, inv=True) @ S_s_half) @ W.T + b).argmax(1) == y[st:st + BATCH]
    return raw, neo, (cov if headcov else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='manifest written by run_gradient_host.py')
    ap.add_argument('--stats', required=True, help='source statistics of the same model (head covariance prior)')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--forward-only', default=None, help='scored forward-only run on the same cells (reference only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-headcov', action='store_true', help='skip the NEO + head covariance variant (the slow part; unused by the report for gradient hosts)')
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    method = 'sar' if any(a.endswith('sar') for a in runs['runs']) else 'eta'
    index = data.ImageNetIndex(args.imagenet_val)
    W, b = models.head(runs['model'])
    src = torch.load(args.stats, map_location='cpu')
    S_s = src['head_cov'].double().numpy(); S_s_half = sqrt_psd(S_s)
    arms = list(runs['runs'])
    run_cells = list(runs['runs'][arms[0]])     # the run's own cells: ImageNet-C by default, ImageNet-C-bar cells when run with --cells
    cells = [c for c in ['clean'] + CELLS if c in run_cells] + [c for c in run_cells if c != 'clean' and c not in CELLS]
    std = [c for c in cells if c != 'clean']
    acc, upd, routed = {}, {}, {}
    for a in arms:
        for c in cells:
            z = np.load(runs['runs'][a][c]['path']); y = z['labels'] if 'labels' in z.files else index.labels(z['index'])
            keep = np.sort(z['class_labels']) if 'class_labels' in z.files and len(z['class_labels']) < 1000 else None
            raw, neo, cov = head_variants(z['h'], W, b, y, S_s_half, S_s, keep, headcov=not args.no_headcov)
            if 'mix' in z.files:                                     # the stack inside the host: the routed rows take the confidence-weighted average
                r = z['routed']; neo = neo.copy(); neo[r] = z['mix'][r].astype(np.float32).argmax(1) == y[r]
                routed.setdefault(a + '+neo', {})[c] = float(r.mean())
            for k, v in (('', raw), ('+neo', neo), ('+headcov', cov)):
                if v is not None:
                    acc.setdefault(a + k, {})[c] = float(v.mean() * 100)
            upd[(a, c)] = runs['runs'][a][c]['eta_updated_samples']
    ref = json.loads(Path(args.forward_only).read_text())['mean_30_cells'] if args.forward_only else {}
    mean = {a: float(np.mean([acc[a][c] for c in std])) for a in acc}
    base = f'{method}+neo' if f'{method}+neo' in mean else list(mean)[0]
    fam = {co: {a: float(np.mean([acc[a][f'{co}-3'], acc[a][f'{co}-5']]) - np.mean([acc[base][f'{co}-3'], acc[base][f'{co}-5']])) for a in acc}
           for co in data.CORRUPTIONS if f'{co}-3' in std and f'{co}-5' in std}
    out = {'n_cells': len([c for c in cells if c != 'clean']), 'mean_30_cells': mean, 'clean': {a: acc[a].get('clean') for a in acc}, 'forward_only_reference': ref,
           'eta_updated_samples_mean': float(np.mean([upd[(a, c)] for (a, c) in upd if c != 'clean'])), 'per_family_vs_eta_neo': fam,
           'routed_fraction_mean': {a: float(np.mean([routed[a][c] for c in std])) for a in routed},
           'per_cell': acc}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    L = [f'# {method.upper()} host, {len(std)} cells x {runs["n_eval"]:,} images (NEO / head cov applied when scoring)', '',
         f'| arm | mean | vs {base} | clean |', '|---|---|---|---|']
    for a in acc:
        L.append(f'| {a} | {mean[a]:.2f} | {mean[a] - mean[base]:+.2f} | {acc[a].get("clean", float("nan")):.2f} |')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
