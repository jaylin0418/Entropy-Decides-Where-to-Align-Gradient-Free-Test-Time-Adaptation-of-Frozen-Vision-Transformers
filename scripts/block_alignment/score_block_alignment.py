"""Scores the multi-block alignment run (CPU; labels): accuracy per (arm, cell) from the saved logits; 30-cell means;
clean harm; warm-up; per family; the A1 (shift diagnostics) and A2 (accuracy) bars declared for the probe. Writes
results/scores/block_alignment_design_vitb.json. (Research name: scripts/regprobe/score_a.py, which wrote
results/regprobe/probe_a.json.)

    python scripts/block_alignment/score_block_alignment.py --runs runs/block_alignment_design_vitb.json \
        --diag runs/block_alignment_design_vitb_diag.json --imagenet-val $IMAGENET_VAL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tokalign import data  # noqa: E402

CELLS = data.CELLS
FAMILY = {'noise': data.CORRUPTIONS[0:3], 'blur': data.CORRUPTIONS[3:7], 'weather': data.CORRUPTIONS[7:11], 'digital': data.CORRUPTIONS[11:15]}
GO, MARGINAL, CLEAN_HARM, SANITY = 66.4, 64.9, 0.5, 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--diag', required=True); ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'scores' / 'block_alignment_design_vitb.json'))
    args = ap.parse_args()
    index = data.ImageNetIndex(args.imagenet_val)
    runs = json.loads(Path(args.runs).read_text()); diag = json.loads(Path(args.diag).read_text())
    arms = list(runs['runs']); cells = [c for c in ['clean'] + CELLS if c in runs['runs'][arms[0]]]; std = [c for c in cells if c != 'clean']
    acc, warm = {a: {} for a in arms}, {a: {} for a in arms}
    for a in arms:
        for c in cells:
            z = np.load(runs['runs'][a][c]['path']); corr = z['logits'].astype(np.float32).argmax(1) == index.labels(z['index'])
            acc[a][c] = float(corr.mean() * 100); warm[a][c] = float(corr[:640].mean() * 100)
    mean = {a: float(np.mean([acc[a][c] for c in std])) for a in arms}
    fam = {a: {f: float(np.mean([acc[a][f'{k}-{s}'] for k in ks for s in data.SEVERITIES if f'{k}-{s}' in std])) for f, ks in FAMILY.items()} for a in arms} if std else {}
    clean = {a: acc[a].get('clean') for a in arms}
    harm = {a: (clean[a] - clean['neo']) if 'clean' in cells else None for a in arms}
    cand = [a for a in arms if a.split('_')[0] in ('mb', 'mc') and not a.startswith('mb_all')]
    best = max(cand, key=lambda a: mean[a]) if cand else None
    verdict = None
    if best:
        ok_clean = harm[best] is None or harm[best] >= -CLEAN_HARM
        verdict = 'GO' if mean[best] >= GO and ok_clean else 'MARGINAL' if mean[best] >= MARGINAL and ok_clean else 'KILL'
    sanity = {a: mean[a] < SANITY for a in arms if a.startswith('mb_all')}
    # A1 gate: per block, averaged over the corrupted cells: outlier tokens carry >= 50 % of |d mu|^2 and content shift <= half
    a1 = []
    for l in range(12):
        rows = [diag['cells'][c][l] for c in std]
        share = float(np.mean([r['share_out'] for r in rows])); ratio = float(np.mean([r['ratio_con'] for r in rows]))
        frac = float(np.mean([r['frac_out'][1] for r in rows])); frac_clean = diag['cells']['clean'][l]['frac_out'][1] if 'clean' in diag['cells'] else None
        a1.append({'block': l, 'share_out': share, 'ratio_con': ratio, 'frac_out_corrupt': frac, 'frac_out_clean': frac_clean,
                   'top2_share': float(np.mean([r['top2_share'] for r in rows])), 'pass': share >= 0.5 and ratio <= 0.5})
    a1_pass = sum(r['pass'] for r in a1) >= 6
    res = {'mean_30': mean, 'clean': clean, 'clean_harm': harm, 'warm_640': {a: float(np.mean([warm[a][c] for c in std])) for a in arms}, 'family': fam, 'per_cell': acc,
           'gate_a2': {'best': best, 'best_mean': mean.get(best), 'gain_over_neo': (mean[best] - mean['neo']) if best else None, 'verdict': verdict, 'sanity_mb_all_collapses': sanity},
           'gate_a1': {'blocks': a1, 'n_pass': int(sum(r['pass'] for r in a1)), 'pass': a1_pass}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    md = ['# multi-block alignment, design cells', '', '| arm | 30-cell mean | clean | harm | warm-640 | ' + ' | '.join(FAMILY) + ' |', '|---|---|---|---|---|' + '---|' * len(FAMILY)]
    for a in sorted(arms, key=lambda a: -mean[a]):
        md.append(f'| {a} | {mean[a]:.2f} | {clean[a]:.2f} | {harm[a]:+.2f} | {res["warm_640"][a]:.2f} | ' + ' | '.join(f'{fam[a][f]:.1f}' for f in FAMILY) + ' |' if clean[a] is not None else f'| {a} | {mean[a]:.2f} |')
    md += ['', f'**A2 gate:** best candidate `{best}` = {mean.get(best, float("nan")):.2f} ({(mean[best] - mean["neo"]) if best else 0:+.2f} over NEO) → **{verdict}** '
           f'(GO ≥ {GO}, MARGINAL ≥ {MARGINAL}, clean harm ≤ {CLEAN_HARM}). Sanity (mb_all collapses < {SANITY}): {sanity}', '',
           f'**A1 gate:** {res["gate_a1"]["n_pass"]}/12 blocks pass (need ≥ 6) → **{"PASS" if a1_pass else "FAIL"}**', '',
           '| block | outlier share of |Δμ|² | content/all shift | outlier frac (corrupt / clean) | top-2 channel share |', '|---|---|---|---|---|']
    md += [f'| {r["block"]} | {r["share_out"]:.2f} | {r["ratio_con"]:.2f} | {r["frac_out_corrupt"]:.4f} / {r["frac_out_clean"] if r["frac_out_clean"] is None else round(r["frac_out_clean"], 4)} | {r["top2_share"]:.2f} |' for r in a1]
    print('\n'.join(md))


if __name__ == '__main__':
    main()
