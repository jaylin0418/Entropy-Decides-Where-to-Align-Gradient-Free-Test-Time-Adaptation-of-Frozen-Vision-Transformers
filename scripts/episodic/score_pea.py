"""Scores a PEA run (CPU; labels): PEA arms with and without TokAlign, the tokenizer gain, PEA's resets, and the
forward-only and ETA arms of the same cells as references. (Research name: scripts/spa/score_pea.py, which wrote
results/spa/spa2_pea*.json.)

    python scripts/episodic/score_pea.py --runs runs/pea_design_vitb_src20k.json --imagenet-val $IMAGENET_VAL \\
        --forward-only results/scores/forward_only_design_vitb.json --gradient-host results/scores/eta_design_vitb.json \\
        --out results/scores/pea_design_vitb_src20k.json
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
    ap.add_argument('--forward-only', default=None)
    ap.add_argument('--gradient-host', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    index = data.ImageNetIndex(args.imagenet_val)
    arms = list(runs['runs']); cells = list(runs['runs'][arms[0]]); std = [c for c in cells if c != 'clean']
    acc = {a: {c: float((np.load(runs['runs'][a][c]['path'])['logits'].astype(np.float32).argmax(1) == index.labels(np.load(runs['runs'][a][c]['path'])['index'])).mean() * 100) for c in cells} for a in arms}
    mean = {a: float(np.mean([acc[a][c] for c in std])) for a in arms}
    resets = {a: int(sum(runs['runs'][a][c]['resets'] for c in std)) for a in arms}
    refs = {}
    for name, path in (('spa1', args.forward_only), ('spa2', args.gradient_host)):    # keys as in the research outputs
        if path:
            r = json.loads(Path(path).read_text())
            refs[name] = {a: float(np.mean([v[c] for c in std])) for a, v in r['per_cell'].items() if all(c in v for c in std)}
    gains = {k: mean[f'tok+{k}'] - mean[k] for k in ('pea', 'pea_aug') if k in mean and f'tok+{k}' in mean}
    clean = {a: acc[a].get('clean') for a in arms}
    bar = {k: bool(v >= 1.0 and (clean[f'tok+{k}'] - clean[k]) >= -0.5) for k, v in gains.items()}
    fam = {co: {a: float(np.mean([acc[a][f'{co}-3'], acc[a][f'{co}-5']]) - np.mean([acc['pea'][f'{co}-3'], acc['pea'][f'{co}-5']])) for a in arms}
           for co in data.CORRUPTIONS if f'{co}-3' in std and f'{co}-5' in std}
    out = {'mean_cells': mean, 'n_cells': len(std), 'clean': clean, 'tokenizer_gain': gains, 'bar_pass': bar, 'resets_total': resets, 'references_same_cells': refs,
           'per_family_vs_pea': fam, 'per_cell': acc}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    L = [f'# PEA, {len(std)} cells x {runs["n_eval"]:,} images; source statistics from {runs["pea"]["source_images"]} clean images', '',
         '| arm | mean | clean | resets |', '|---|---|---|---|'] + [f'| {a} | {mean[a]:.2f} | {clean[a]:.2f} | {resets[a]} |' for a in arms]
    print('\n'.join(L))


if __name__ == '__main__':
    main()
