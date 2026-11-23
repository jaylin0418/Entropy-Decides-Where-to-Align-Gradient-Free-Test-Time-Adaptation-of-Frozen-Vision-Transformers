"""Scores an FOA run (CPU; labels): FOA arms (logits already shifted by FOA), with the forward-only and ETA arms of the
same cells as references. (Research name: scripts/spa/score_foa.py, which wrote results/spa/spa3*.json.)

    python scripts/episodic/score_foa.py --runs runs/foa_design_vitb.json --imagenet-val $IMAGENET_VAL \\
        --forward-only results/scores/forward_only_design_vitb.json --gradient-host results/scores/eta_design_vitb.json \\
        --out results/scores/foa_design_vitb.json
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
    ap.add_argument('--forward-only', default=None, help='scored forward-only run (reference on the same cells)')
    ap.add_argument('--gradient-host', default=None, help='scored ETA run (reference on the same cells)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    index = data.ImageNetIndex(args.imagenet_val)
    arms = list(runs['runs'])
    cells = list(runs['runs'][arms[0]])
    std = [c for c in cells if c != 'clean']
    acc = {a: {} for a in arms}
    for a in arms:
        for c in cells:
            z = np.load(runs['runs'][a][c]['path']); y = index.labels(z['index'])
            acc[a][c] = float((z['logits'].astype(np.float32).argmax(1) == y).mean() * 100)
    mean = {a: float(np.mean([acc[a][c] for c in std])) for a in arms}
    refs = {}
    for name, path in (('spa1', args.forward_only), ('spa2', args.gradient_host)):    # keys as in the research outputs
        if path:
            r = json.loads(Path(path).read_text())
            refs[name] = {a: float(np.mean([v[c] for c in std if c in v])) for a, v in r['per_cell'].items() if all(c in v for c in std)}
    out = {'mean_cells': mean, 'n_cells': len(std), 'clean': {a: acc[a].get('clean') for a in arms}, 'references_same_cells': refs,
           'per_cell': acc, 'seconds': {a: {c: runs['runs'][a][c]['seconds'] for c in cells} for a in arms}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    L = [f'# FOA, {len(std)} cells x {runs["n_eval"]:,} images', '', '| arm | mean | clean |', '|---|---|---|'] + \
        [f'| {a} | {mean[a]:.2f} | {acc[a].get("clean", float("nan")):.2f} |' for a in arms]
    for name, r in refs.items():
        L += ['', f'Same cells, {name}: ' + ', '.join(f'{a} {v:.2f}' for a, v in r.items())]
    print('\n'.join(L))


if __name__ == '__main__':
    main()
