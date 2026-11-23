"""Checks from the saved logits, without a model call (labels only to score): the uncertainty of the held-out gains, the
entropy-accuracy correlation without the first three blocks, the sensitivity to the threshold, the calibration's pick
among every pair of branch blocks, and the entropy the pick saves against TokAlign alone on every model-shift pair.

Writes one JSON that paper/make_numbers.py reads (prefix rc).

    python scripts/analysis/review_checks.py --logits runs/logits --imagenet-val $IMAGENET_VAL --scores results/scores \
        --out results/scores/review_checks.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data  # noqa: E402
from scripts.analysis.site_mixture import entropy, mixture, softmax  # noqa: E402

B = 2000                                                              # bootstrap draws
INC = [f'{c}-{s}' for c in data.CORRUPTIONS for s in (3, 5)]
CELLS = {'inc': INC, 'cbar': list(data.CBAR_CELLS), 'r': [f'r-h{k}' for k in (1, 2, 3)], 'sketch': [f'sketch-h{k}' for k in (1, 2, 3)]}
# the held-out run of every row of Table 3 and Table S1: (cascade file prefix, main-path file prefix, calibration source)
PAIRS = {'vitb_inc': ('forward_only_heldout3_vitb_rule_rule_t6_b4b11_gated', 'forward_only_heldout3_vitb_tok+place_b6', 'site_choice_design_vitb', 6),
         'vitb_cbar': ('forward_only_heldout4_vitb_cbar_rule_rule_t7_b5b11_gated', 'forward_only_heldout4_vitb_cbar_rule_tok+place_b7', 'site_choice_design_vitb_cbar_t7', 7),
         'vits_inc': ('forward_only_heldout3_vits_rule_rule_t6_b4b11_gated', 'forward_only_heldout3_vits_tok+place_b6', 'site_choice_design_vits_rule', 6)}


def load(L, prefix, cell, index):
    f = L / f'{prefix}_{cell}.npz'
    if not f.exists() and prefix.endswith('_gated'):                 # some runs name the per-cell files without the _gated suffix
        f = L / f'{prefix[:-6]}_{cell}.npz'
    z = np.load(f); lg = z['logits'].astype(np.float32)
    y = z['labels'] if 'labels' in z.files else index.labels(z['index'])
    if 'class_labels' in z.files:                                     # ImageNet-R: predictions restricted to its 200 classes
        keep = np.zeros(lg.shape[1], bool); keep[z['class_labels']] = True; lg = np.where(keep[None, :], lg, -np.inf)
    return lg, y


def paired_bootstrap(L, casc, main, cells, index, rng):
    """The gain of the cascade over the main path with a 95% interval over images and over cells."""
    d_img, d_cell = [], []
    for c in cells:
        lm, y = load(L, main, c, index); lc, y2 = load(L, casc, c, index); assert (y == y2).all()
        d = (lc.argmax(1) == y).astype(float) - (lm.argmax(1) == y).astype(float); d_img.append(d); d_cell.append(d.mean() * 100)
    d_img = np.concatenate(d_img); d_cell = np.array(d_cell)
    bi = np.array([d_img[rng.integers(0, len(d_img), len(d_img))].mean() * 100 for _ in range(B)])
    bc = np.array([d_cell[rng.integers(0, len(d_cell), len(d_cell))].mean() for _ in range(B)])
    return {'gain': float(d_img.mean() * 100), 'ci_images': [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))],
            'ci_cells': [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))], 'n_cells': len(cells), 'n_images': int(len(d_img)),
            'cells_above': int((d_cell > 0).sum()), 'changed': float((d_img != 0).mean() * 100), 'fixed': float((d_img > 0).mean() * 100), 'broken': float((d_img < 0).mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits', required=True); ap.add_argument('--imagenet-val', required=True); ap.add_argument('--scores', default='results/scores')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    L, S = Path(args.logits), Path(args.scores); index = data.ImageNetIndex(args.imagenet_val); rng = np.random.default_rng(0)
    out = {'bootstrap_draws': B, 'pairs': {}}
    # 1. every model-shift pair: the held-out gain with its interval, the correlation with and without blocks 1 to 3, the entropy the
    #    pick saves against TokAlign alone and the pick's accuracy gap, read from the calibration file of the pair
    for model in ('vitb', 'vits', 'deitb', 'deits'):
        for ds in ('inc', 'cbar', 'r', 'sketch'):
            key = f'{model}_{ds}'; split = 'heldout4' if ds in ('inc', 'cbar') else 'heldout'
            tag = f'{model}_cbar_ca' if ds == 'cbar' else f'{model}_ca_{ds}'
            if key in PAIRS:
                casc, main, sc, T = PAIRS[key]
            else:
                cal = S / f'calib_{model}_{ds}.json'
                if not cal.exists():
                    continue
                c = json.load(open(cal)); T = c['trunk']
                casc, main, sc = f"forward_only_{split}_{tag}_{c['arm']}", f'forward_only_{split}_{tag}_tok+place_b{T}', f'site_choice_design_{tag}_t6'
            if not glob.glob(str(L / f'{casc}_*.npz')):
                print(f'{key}: no held-out logits ({casc}), skipped', file=sys.stderr); continue
            lf = json.load(open(S / f'{sc}.json'))['label_free']; H, A = lf['trunk_mean_entropy'], lf['trunk_accuracy']
            arms = ['px'] + [f'px+b{k}' for k in range(1, 13)]; late = ['px'] + [f'px+b{k}' for k in range(4, 13)]
            pick = min(arms, key=H.get); best = max(arms, key=A.get)
            r = paired_bootstrap(L, casc, main, CELLS[ds], index, rng)
            r.update({'trunk': T, 'rho_all': float(spearmanr([-H[a] for a in arms], [A[a] for a in arms]).correlation),
                      'rho_from_b4': float(spearmanr([-H[a] for a in late], [A[a] for a in late]).correlation),
                      'pick': pick, 'best': best, 'entropy_saved_vs_px': float(H['px'] - H[pick]), 'gap': float(A[pick] - A[best]),
                      'px_gain_over_neo': float(A['px'] - lf.get('neo_accuracy', np.nan)) if 'neo_accuracy' in lf else None})
            out['pairs'][key] = r
            print(f"{key}: gain {r['gain']:+.2f} [{r['ci_images'][0]:+.2f}, {r['ci_images'][1]:+.2f}] images, [{r['ci_cells'][0]:+.2f}, {r['ci_cells'][1]:+.2f}] cells, "
                  f"rho {r['rho_all']:+.2f} -> {r['rho_from_b4']:+.2f} without blocks 1-3, pick {pick} best {best} gap {r['gap']:+.2f}, entropy saved {r['entropy_saved_vs_px']:+.2f}")
    # 2. ViT-B/16 on the fourth held-out set (Table 1): the same interval for the cascade 6/4,11 on ImageNet-C and ImageNet-C-bar
    out['table1'] = {'inc': paired_bootstrap(L, 'forward_only_heldout4_vitb_rule_rule_t6_b4b11_gated', 'forward_only_heldout4_vitb_exit_pair_gated', INC, index, rng),
                     'cbar': paired_bootstrap(L, 'forward_only_heldout4_vitb_cbar_rule_rule_t6_b4b11_gated', 'forward_only_heldout4_vitb_cbar_exit_pair_gated', CELLS['cbar'], index, rng)}
    for b, r in out['table1'].items():
        print(f"table1 {b}: gain {r['gain']:+.2f} [{r['ci_images'][0]:+.2f}, {r['ci_images'][1]:+.2f}] images, [{r['ci_cells'][0]:+.2f}, {r['ci_cells'][1]:+.2f}] cells")
    # 3. ViT-B/16 dev predictions: the threshold and the pairs (ImageNet-C, main block 6, branches 4 and 11)
    P, Y = {}, {}
    for c in INC:
        lm, y = load(L, 'forward_only_design_vitb_two_site_tok+place_b6', c, index); Y[c] = y; P[c] = {'px+b6': softmax(lm)}
        P[c]['px'] = softmax(load(L, 'forward_only_design_vitb_two_site_tok', c, index)[0])
        for k in range(1, 13):
            if k != 6:
                P[c][f'px+b{k}'] = softmax(load(L, f'forward_only_design_vitb_select_tok+place_b{k}', c, index)[0])
    H6 = {c: entropy(P[c]['px+b6']) for c in INC}; allH = np.concatenate([H6[c] for c in INC])
    base = float(np.mean([(P[c]['px+b6'].argmax(1) == Y[c]).mean() * 100 for c in INC]))

    def gated_gain(cand, th):
        accs, share = [], []
        for c in INC:
            m = mixture(P[c], cand); go = H6[c] > th
            accs.append((np.where(go[:, None], m, P[c]['px+b6']).argmax(1) == Y[c]).mean() * 100); share.append(go.mean())
        return float(np.mean(accs) - base), float(np.mean(share) * 100)

    out['theta'] = {}
    for q in (0.0, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9):
        th = float(np.quantile(allH, q)); g, s = gated_gain(['px+b6', 'px+b4', 'px+b11'], th)
        out['theta'][str(q)] = {'theta': th, 'gain': g, 'unsure': s}; print(f'quantile {q:.2f} theta {th:.2f} gain {g:+.2f} unsure {s:.0f}%')
    g, s = gated_gain(['px+b6', 'px+b4', 'px+b11'], 1.5); out['theta']['1.5nat'] = {'theta': 1.5, 'gain': g, 'unsure': s}
    pairs = {}
    for a, b in combinations([k for k in range(1, 13) if k != 6], 2):
        pairs[f'{a}|{b}'] = gated_gain(['px+b6', f'px+b{a}', f'px+b{b}'], 1.5)[0]
    ranked = sorted(pairs, key=pairs.get, reverse=True)
    adj = lambda k: abs(k - 6) == 1
    far = [k for k in pairs if not any(adj(int(x)) for x in k.split('|'))]
    out['pairs_dev'] = {'n': len(pairs), 'pick': '4|11', 'pick_gain': pairs['4|11'], 'pick_rank': ranked.index('4|11') + 1, 'best': ranked[0], 'best_gain': pairs[ranked[0]],
                        'mean': float(np.mean(list(pairs.values()))), 'median': float(np.median(list(pairs.values()))), 'worst': ranked[-1], 'worst_gain': pairs[ranked[-1]],
                        'mean_no_adjacent': float(np.mean([pairs[k] for k in far])), 'n_no_adjacent': len(far), 'adjacent_5_7': pairs['5|7'], 'ends_1_12': pairs['1|12'],
                        'top5': {k: pairs[k] for k in ranked[:5]}, 'all': pairs}
    print(f"pairs: pick 4|11 {pairs['4|11']:+.2f} rank {ranked.index('4|11') + 1}/{len(pairs)}, best {ranked[0]} {pairs[ranked[0]]:+.2f}, mean {np.mean(list(pairs.values())):+.2f}, "
          f"mean without a block next to 6 {np.mean([pairs[k] for k in far]):+.2f} ({len(far)} pairs), 5|7 {pairs['5|7']:+.2f}, 1|12 {pairs['1|12']:+.2f}, worst {ranked[-1]} {pairs[ranked[-1]]:+.2f}")
    json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
