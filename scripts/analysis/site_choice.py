"""Which block, per cell or per image (CPU; labels only to score): from the saved logits of the design runs, the accuracy of
(1) the pixel site plus one block for every block, (2) the per-cell choice of that block by the mean prediction entropy of the
first K batches (the pair's predictions are output during the warm-up), (3) the per-cell oracle, (4) the per-image entropy
mixture of the pair with extra branches, with and without the early exit on the pair's entropy. Writes one JSON.

    python scripts/analysis/site_choice.py --logits runs/logits --base forward_only_design_vitb_two_site \\
        --blocks forward_only_design_vitb_select --imagenet-val $IMAGENET_VAL --cells inc --out results/scores/site_choice_design_vitb.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402
from scripts.analysis.site_mixture import BATCH, K_WARMUP, entropy, mixture, softmax  # noqa: E402

THETA = 1.5
EXTRA_SETS = {'b9': ('px+b9',), 'b5b9': ('px+b5', 'px+b9'), 'px': ('px',), 'b4b8': ('px+b4', 'px+b8'), 'b4b9b10': ('px+b4', 'px+b9', 'px+b10')}


def pick_second_opinions(so_ent, kt, nb):
    """The pair rule. ``so_ent['px+b<a>|px+b<b>']`` (a < b) is the mean entropy of the confidence-weighted average of the main path
    (block ``kt``) and the two branches after blocks a and b on the unsure images. The pick is the pair of blocks, any two blocks
    other than the main block, with the lowest entropy. Returns (shallower, deeper, key): the two blocks of the
    pair in depth order and the key as the pairs are named. The pixel placement alone is not a block and is left out."""
    cands = [k for k in so_ent if '|' in k and all(a.startswith('px+b') for a in k.split('|')) and kt not in [int(a[4:]) for a in k.split('|')]]
    key = min(cands, key=lambda k: so_ent[k])
    lo, hi = sorted(key.split('|'), key=lambda a: int(a[4:]))
    return lo, hi, f'{lo}|{hi}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits', required=True); ap.add_argument('--base', required=True, help='run with neo, tok, place_b6, tok+place_b6')
    ap.add_argument('--blocks', required=True, help='run with tok+place_b1..b12 except b6'); ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--cells', choices=['inc', 'cbar'] + list(data.NATURAL), default='inc'); ap.add_argument('--out', required=True)
    ap.add_argument('--n-blocks', type=int, default=12, help='blocks of the model (24 for ViT-L/16)')
    ap.add_argument('--trunk', type=int, default=6, help='the trunk block (pixel site plus the map after it); 6 for the report')
    args = ap.parse_args()
    index = data.ImageNetIndex(args.imagenet_val); L = Path(args.logits)
    cells = ['clean'] + ([f'{c}-{s}' for c in data.CORRUPTIONS for s in (3, 5)] if args.cells == 'inc' else list(data.CBAR_CELLS) if args.cells == 'cbar'
                         else [f'{args.cells}-d{k}' for k in range(1, data.N_DESIGN + 1)])
    NB = args.n_blocks
    corrupted = [c for c in cells if c != 'clean']
    P, Y = {}, {}
    for c in cells:
        z = np.load(L / f'{args.base}_tok_{c}.npz'); Y[c] = z['labels'] if 'labels' in z.files else np.array([index.label(int(i)) for i in z['index']])
        P[c] = {'px': softmax(z['logits'].astype(np.float32)), 'neo': softmax(np.load(L / f'{args.base}_neo_{c}.npz')['logits'].astype(np.float32)),
                'px+b6': softmax(np.load(L / f'{args.base}_tok+place_b6_{c}.npz')['logits'].astype(np.float32))}
        for k in range(1, NB + 1):
            if k != 6:
                P[c][f'px+b{k}'] = softmax(np.load(L / f'{args.blocks}_tok+place_b{k}_{c}.npz')['logits'].astype(np.float32))
    TR = f'px+b{args.trunk}'                                     # the trunk arm; every quantity below is relative to it
    arms = ['px'] + [f'px+b{k}' for k in range(1, NB + 1)]
    acc = lambda c, p: float((p.argmax(1) == Y[c]).mean() * 100)
    fixed = {a: float(np.mean([acc(c, P[c][a]) for c in corrupted])) for a in arms + ['neo']}
    pair = fixed[TR]
    oracle = float(np.mean([max(acc(c, P[c][a]) for a in arms) for c in corrupted]))
    n = K_WARMUP * BATCH
    choice = {}
    for c in corrupted:
        a = min(arms, key=lambda a: entropy(P[c][a][:n]).mean())
        pred = np.concatenate([P[c][TR][:n].argmax(1), P[c][a][n:].argmax(1)]); choice[c] = {'arm': a, 'acc': float((pred == Y[c]).mean() * 100)}
    per_cell_choice = float(np.mean([choice[c]['acc'] for c in corrupted]))
    mixes = {}
    for name, extra in EXTRA_SETS.items():
        cand = [TR] + list(extra); full, ex, routed = [], [], []
        for c in corrupted:
            m = mixture(P[c], cand); go = entropy(P[c][TR]) > THETA
            full.append(acc(c, m)); ex.append(float((np.where(go[:, None], m, P[c][TR]).argmax(1) == Y[c]).mean() * 100)); routed.append(float(go.mean()))
        mixes[name] = {'extra': list(extra), 'full': float(np.mean(full)), 'early_exit': float(np.mean(ex)), 'routed_fraction': float(np.mean(routed)),
                       'clean_early_exit': acc('clean', np.where((entropy(P['clean'][TR]) > THETA)[:, None], mixture(P['clean'], cand), P['clean'][TR]))}
    # equal-compute controls of the per-image method (trunk px+b6, branches px+b5 and px+b9): random routing at the same fraction
    # (five seeded draws), routing the most confident images instead, equal weights, the full mixture of every image, one branch
    cand = [TR, 'px+b5', 'px+b9']
    def routed(c, mask, arms=cand, equal=False):
        m = np.mean([P[c][a] for a in arms], 0) if equal else mixture(P[c], arms)
        return np.where(mask[:, None], m, P[c][TR])
    H = {c: entropy(P[c][TR]) for c in corrupted}
    frac = {c: float((H[c] > THETA).mean()) for c in corrupted}
    mean_acc = lambda f: float(np.mean([acc(c, f(c)) for c in corrupted]))
    controls = {'routed_fraction': float(np.mean(list(frac.values()))),
                'entropy_routing': mean_acc(lambda c: routed(c, H[c] > THETA)),
                'entropy_routing_equal_weights': mean_acc(lambda c: routed(c, H[c] > THETA, equal=True)),
                'random_routing_same_fraction': float(np.mean([mean_acc(lambda c, s=s: routed(c, np.random.default_rng(s).random(len(Y[c])) < frac[c])) for s in range(5)])),
                'most_confident_routed_same_fraction': mean_acc(lambda c: routed(c, H[c] <= np.quantile(H[c], frac[c]))),
                'full_mixture_every_image': mean_acc(lambda c: routed(c, np.ones(len(Y[c]), bool))),
                'full_mixture_every_image_equal_weights': mean_acc(lambda c: routed(c, np.ones(len(Y[c]), bool), equal=True)),
                'entropy_routing_b9_only': mean_acc(lambda c: routed(c, H[c] > THETA, arms=[TR, 'px+b9'])),
                'entropy_routing_b5_only': mean_acc(lambda c: routed(c, H[c] > THETA, arms=[TR, 'px+b5'])),
                'oracle_routing_trunk_errors': mean_acc(lambda c: routed(c, P[c][TR].argmax(1) != Y[c]))}
    # why these sites: every candidate second opinion (pixel site plus one block other than 6) on the images the trunk is unsure
    # of: its own accuracy, its agreement with the trunk, the share it rescues (trunk wrong, site right) and harms (trunk right,
    # site wrong), and the gain of the routed mixture with the trunk; then every pair of second opinions and, for the pairs
    # named, the overlap of what they rescue on the trunk's errors
    cat = lambda f: np.concatenate([f(c) for c in corrupted])
    yy = cat(lambda c: Y[c]); tp = cat(lambda c: P[c][TR].argmax(1)); rt = cat(lambda c: H[c] > THETA); tr = tp == yy
    cands = [a for a in arms if a != TR]
    pair_mean = float(np.mean([acc(c, P[c][TR]) for c in corrupted]))
    per_site = {}
    for a in cands:
        pa = cat(lambda c, a=a: P[c][a].argmax(1)); ra = pa == yy
        per_site[a] = {'acc_all': float(ra.mean() * 100), 'acc_routed': float(ra[rt].mean() * 100), 'agree_routed': float(np.mean(pa[rt] == tp[rt]) * 100),
                       'rescue_routed': float(np.mean(ra[rt] & ~tr[rt]) * 100), 'harm_routed': float(np.mean(~ra[rt] & tr[rt]) * 100),
                       'gain_routed_mixture': mean_acc(lambda c, a=a: routed(c, H[c] > THETA, arms=[TR, a])) - pair_mean}
    pairs = {}
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            pairs[f'{a}|{b}'] = mean_acc(lambda c, a=a, b=b: routed(c, H[c] > THETA, arms=[TR, a, b])) - pair_mean
    err = rt & ~tr
    overlap = {}
    for a, b in (('px+b5', 'px+b9'), ('px+b4', 'px+b9'), ('px+b3', 'px+b9'), ('px+b5', 'px+b7'), ('px+b8', 'px+b9'), ('px+b4', 'px+b5')):
        ra, rb = (cat(lambda c, a=a: P[c][a].argmax(1)) == yy), (cat(lambda c, b=b: P[c][b].argmax(1)) == yy)
        overlap[f'{a}|{b}'] = {'either': float(np.mean(ra[err] | rb[err]) * 100), 'both': float(np.mean(ra[err] & rb[err]) * 100)}
    second_opinions = {'trunk_acc_routed': float(tr[rt].mean() * 100), 'trunk_acc_kept': float(tr[~rt].mean() * 100), 'per_site': per_site,
                       'pairs': pairs, 'rescue_overlap_on_trunk_errors': overlap}
    # label-free choice of the blocks by the prediction entropy (low entropy = high confidence), the same quantity that routes and
    # weights the second opinions: (1) the trunk block as the one whose pixel-plus-block arm has the lowest mean entropy on the
    # unlabeled images; (2) the second opinions as the set (one or two sites besides the trunk) whose routed mixture has the lowest
    # mean entropy on the routed images; both compared with the labeled choice
    from scipy.stats import spearmanr
    trunk_ent = {a: float(np.mean([entropy(P[c][a]).mean() for c in corrupted])) for a in arms}
    trunk_acc = {a: fixed[a] for a in arms}
    so_sets = [(a,) for a in cands] + [tuple(k.split('|')) for k in pairs]
    so_gain, so_ent = {}, {}
    for st in so_sets:
        key = '|'.join(st); arm_set = [TR] + list(st)
        so_gain[key] = per_site[st[0]]['gain_routed_mixture'] if len(st) == 1 else pairs[key]
        so_ent[key] = float(np.mean([entropy(mixture({a: P[c][a][H[c] > THETA] for a in arm_set}, arm_set)).mean() for c in corrupted]))
    lf_pick = min(so_ent, key=so_ent.get); best_key = max(so_gain, key=so_gain.get)
    # the pair rule: the two blocks whose routed mixture with the trunk has the lowest mean entropy (the pixel site alone is not a
    # block and is left out)
    pick_lo, pick_hi, pick_ba = pick_second_opinions(so_ent, args.trunk, NB)
    label_free = {'signal': 'mean prediction entropy (lower = more confident)', 'trunk_mean_entropy': trunk_ent, 'trunk_accuracy': trunk_acc,
                  'trunk_pick_by_entropy': min(trunk_ent, key=trunk_ent.get), 'trunk_best_by_accuracy': max(trunk_acc, key=trunk_acc.get),
                  'trunk_spearman_confidence_accuracy': float(spearmanr([-trunk_ent[a] for a in arms], [trunk_acc[a] for a in arms]).correlation),
                  'second_opinion_mixture_entropy': so_ent, 'second_opinion_pick_by_entropy': lf_pick, 'second_opinion_pick_gain': so_gain[lf_pick],
                  'second_opinion_best': best_key, 'second_opinion_best_gain': so_gain[best_key],
                  'second_opinion_pick_pair': pick_ba, 'second_opinion_pick_pair_gain': so_gain[pick_ba],
                  'second_opinion_pick_shallower': pick_lo, 'second_opinion_pick_deeper': pick_hi,
                  # the threshold as a quantile of the trunk's entropy on the design images (the model- and dataset-free way to set it)
                  'trunk_entropy_quantile': {str(q): float(np.quantile(np.concatenate([entropy(P[c][TR]) for c in corrupted]), q)) for q in (0.5, 0.6, 0.65, 0.7)},
                  'routed_share_at_theta': float(np.mean([(entropy(P[c][TR]) > THETA).mean() for c in corrupted])),
                  'second_opinion_spearman_confidence_gain': float(spearmanr([-so_ent[k] for k in so_gain], [so_gain[k] for k in so_gain]).correlation),
                  'second_opinion_spearman_disagreement_gain': float(spearmanr([-np.mean([per_site[a]['agree_routed'] for a in k.split('|')]) for k in so_gain], [so_gain[k] for k in so_gain]).correlation)}
    out = {'trunk': TR, 'cells': corrupted, 'k_warmup': K_WARMUP, 'theta': THETA, 'controls': controls, 'second_opinions': second_opinions, 'label_free': label_free, 'fixed_mean': fixed, 'pair': pair, 'oracle_per_cell': oracle, 'oracle_gain': oracle - pair,
           'per_cell_entropy_choice': per_cell_choice, 'per_cell_entropy_choice_gain': per_cell_choice - pair, 'per_cell_choice': choice,
           'chosen_arm_counts': {a: sum(v['arm'] == a for v in choice.values()) for a in arms if any(v['arm'] == a for v in choice.values())},
           'per_image_mixtures': mixes, 'clean': {a: acc('clean', P['clean'][a]) for a in ['neo', 'px', TR]}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(f'pair {pair:.2f}  oracle {oracle:.2f} ({oracle-pair:+.2f})  per-cell entropy choice {per_cell_choice:.2f} ({per_cell_choice-pair:+.2f})')
    for name, m in mixes.items():
        print(f'  mixture +{"+".join(m["extra"])}: full {m["full"]:.2f} ({m["full"]-pair:+.2f}), early exit {m["early_exit"]:.2f} ({m["early_exit"]-pair:+.2f}), routed {m["routed_fraction"]:.2f}')


if __name__ == '__main__':
    main()
