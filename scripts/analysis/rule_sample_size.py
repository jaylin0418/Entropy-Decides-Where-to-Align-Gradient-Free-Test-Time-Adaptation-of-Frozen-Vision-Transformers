"""How many unlabeled images the entropy rule needs (CPU; labels only to score): from the saved logits of the design runs,
the rule of docs/heldout_test_plan_rule.md (trunk = block of lowest mean entropy over the pixel-plus-block arms; the branches = the
pair of other blocks whose routed mixture with the trunk has the lowest mean entropy on the unsure images) is applied to
(1) the first n images of every cell, pooled over the cells (one choice per benchmark), for n from 64 to 2,000, and to
(2) the first K batches of each cell alone (one choice per shift, the pair's predictions output during the warm-up).
Each choice is scored by the routed-mixture accuracy of the configuration it names on the full cells. Writes one JSON.

    python scripts/analysis/rule_sample_size.py --logits runs/logits --base forward_only_design_vitb_two_site \\
        --blocks forward_only_design_vitb_select --imagenet-val $IMAGENET_VAL --cells inc --out results/scores/rule_sample_size_design_vitb.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402
from scripts.analysis.site_mixture import BATCH, entropy, mixture, softmax  # noqa: E402

THETA = 1.5
NB = 12                                                       # blocks of the model (24 for ViT-L/16); set from --n-blocks
SIZES = (64, 128, 192, 320, 640, 1280, 2000)
KS = (1, 2, 3, 5, 10)


def rule(P, cells, n=None):
    """The picks (trunk, shallower, deeper) from the first n images of each cell in `cells` (all images when n is None): the trunk is
    the block of lowest mean entropy, the branches the pair of other blocks whose routed mixture with the trunk has the lowest mean
    entropy on the unsure images, shallower < deeper."""
    sl = slice(0, n)
    arms = [f'px+b{k}' for k in range(1, NB + 1)]
    t = min(arms, key=lambda a: float(np.mean([entropy(P[c][a][sl]).mean() for c in cells])))

    def mix_ent(pair):
        vals = []
        for c in cells:
            H = entropy(P[c][t][sl]); r = H > THETA
            if r.sum() == 0:
                continue
            arm_set = [t, *pair]
            vals.append(entropy(mixture({a: P[c][a][sl][r] for a in arm_set}, arm_set)).mean())
        return float(np.mean(vals)) if vals else np.inf
    others = [a for a in arms if a != t]
    lo, hi = min(itertools.combinations(others, 2), key=mix_ent)
    return t, lo, hi


def routed_pred(P, c, t, lo, hi, start=0):
    """Predictions of the configuration on cell c from image `start` on: the trunk, or the routed mixture where H > theta."""
    arms = [a for a in (t, lo, hi) if a]
    H = entropy(P[c][t][start:]); r = H > THETA
    m = mixture({a: P[c][a][start:] for a in arms}, arms)
    return np.where(r[:, None], m, P[c][t][start:]).argmax(1), float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits', required=True); ap.add_argument('--base', required=True); ap.add_argument('--blocks', required=True)
    ap.add_argument('--imagenet-val', required=True); ap.add_argument('--cells', choices=['inc', 'cbar'] + list(data.NATURAL), default='inc'); ap.add_argument('--out', required=True)
    ap.add_argument('--n-blocks', type=int, default=12)
    args = ap.parse_args()
    index = data.ImageNetIndex(args.imagenet_val); L = Path(args.logits)
    cells = [f'{c}-{s}' for c in data.CORRUPTIONS for s in (3, 5)] if args.cells == 'inc' else list(data.CBAR_CELLS) if args.cells == 'cbar' else [f'{args.cells}-d{k}' for k in range(1, data.N_DESIGN + 1)]
    global NB
    NB = args.n_blocks
    P, Y = {}, {}
    for c in cells:
        z = np.load(L / f'{args.base}_tok+place_b6_{c}.npz'); Y[c] = z['labels'] if 'labels' in z.files else np.array([index.label(int(i)) for i in z['index']])
        P[c] = {'px+b6': softmax(z['logits'].astype(np.float32))}
        for k in range(1, NB + 1):
            if k != 6:
                P[c][f'px+b{k}'] = softmax(np.load(L / f'{args.blocks}_tok+place_b{k}_{c}.npz')['logits'].astype(np.float32))
    acc = lambda c, pred, start=0: float((pred == Y[c][start:]).mean() * 100)
    pair = float(np.mean([acc(c, P[c]['px+b6'].argmax(1)) for c in cells]))
    full = rule(P, cells)
    full_gain = float(np.mean([acc(c, routed_pred(P, c, *full)[0]) for c in cells])) - pair
    # (1) one choice per benchmark from the first n images of every cell
    pooled = {}
    for n in SIZES:
        t, lo, hi = rule(P, cells, n)
        g = float(np.mean([acc(c, routed_pred(P, c, t, lo, hi)[0]) for c in cells])) - pair
        pooled[n] = {'trunk': t, 'shallower': lo, 'deeper': hi, 'gain_over_pair': g, 'same_as_full': [t, lo, hi] == list(full),
                     'images_used': n * len(cells)}
    # (2) one choice per cell from its first K batches, the pair output during the warm-up
    per_cell = {}
    for K in KS:
        n = K * BATCH; picks, accs, routed = {}, [], []
        for c in cells:
            t, lo, hi = rule(P, [c], n); picks[c] = [t, lo, hi]
            pred_after, r = routed_pred(P, c, t, lo, hi, start=n)
            pred = np.concatenate([P[c]['px+b6'][:n].argmax(1), pred_after]); accs.append(acc(c, pred)); routed.append(r)
        tr = [p[0] for p in picks.values()]
        per_cell[K] = {'images_per_cell': n, 'gain_over_pair': float(np.mean(accs)) - pair, 'routed_fraction_after_warmup': float(np.mean(routed)),
                       'trunk_counts': {a: tr.count(a) for a in sorted(set(tr))}, 'picks': picks,
                       'same_trunk_as_full': float(np.mean([p[0] == full[0] for p in picks.values()]))}
    out = {'cells': cells, 'theta': THETA, 'pair': pair, 'full_rule': {'trunk': full[0], 'shallower': full[1], 'deeper': full[2], 'gain_over_pair': full_gain},
           'pooled_first_n_per_cell': pooled, 'per_cell_first_k_batches': per_cell,
           'note': 'labels are used only to score; the rule sees logits only. The selection pass costs 12 extra forward passes on the sample (one per candidate block).'}
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'full': out['full_rule'], 'pooled': {n: (v['trunk'], v['shallower'], v['deeper'], round(v['gain_over_pair'], 2)) for n, v in pooled.items()},
                      'per_cell': {K: (round(v['gain_over_pair'], 2), v['trunk_counts']) for K, v in per_cell.items()}}, indent=0))


if __name__ == '__main__':
    main()
