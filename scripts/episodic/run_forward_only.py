"""Episodic forward-only runs (GPU; no labels): NEO, TokAlign, the head covariance stage, the stack, the two-site arms
(TokAlign plus the same alignment after block 6), their ablations and the multi-block alignment arms, one stream per cell (statistics reset per cell). Saves the logits of every (arm, cell)
under <out-dir>/logits/ and a manifest <out-dir>/forward_only_<split>_<model><tag>.json, which
scripts/episodic/score_forward_only.py scores. (Research name: scripts/spa/probe_online.py.)

    python scripts/episodic/run_forward_only.py --split design --arms neo tok headcov tok+headcov \\
        --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, Aligner, branch_mixture, load_source, set_fp32  # noqa: E402

CELLS = data.CELLS
ARMS = {
    'neo': dict(tok=False, mean=(), cov=(), headcov=False),
    'tok': dict(tok=True, mean=(), cov=(), headcov=False),
    'headcov': dict(tok=False, mean=(), cov=(), headcov=True),
    'tok+headcov': dict(tok=True, mean=(), cov=(), headcov=True),
    # the two-site method: TokAlign's map also on the patch tokens after block 6 (0-based 5); block 6 alone for reference
    'place_b6': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,)),
    'tok+place_b6': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,)),
    'tok+place_b6+headcov': dict(tok=True, mean=(), cov=(), headcov=True, place=(5,)),
    # the branch mixture: the pixel map, blocks 1 to 6 once, then blocks 7 to 12 with and without the block-6 map, the two
    # softmax outputs averaged per image with entropy weights (tokalign.aligner.branch_mixture); 1.5 forward passes
    'branch': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), branch=True),
    # early-exit branches (tokalign.aligner.forward_exit): the pair is the trunk; images whose trunk entropy exceeds 1.5 nat also
    # run the extra sites (0-based blocks) and the outputs are mixed per image; '_gated' adds the spectral gate of the stack
    'exit_b9': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(8,), exit_theta=1.5),
    'exit_b5b9': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=1.5),
    'exit_b9_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(8,), exit_theta=1.5, gate={'spec': 0.4, 'mean': 0.3}),
    'exit_b5b9_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=1.5, gate={'spec': 0.4, 'mean': 0.3}),
    'pair_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), gate={'spec': 0.4, 'mean': 0.3}),
    # equal-compute controls of the method (docs/heldout_test_plan_exit.md, addendum): random routing at the method's design fraction,
    # equal weights instead of entropy weights, and every image routed (the full mixture at full cost)
    'exit_b5b9_random_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=1.5, exit_random=0.35, gate={'spec': 0.4, 'mean': 0.3}),
    'exit_b5b9_equal_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=1.5, exit_equal=True, gate={'spec': 0.4, 'mean': 0.3}),
    'mix_b5b9_gated': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=-1.0, gate={'spec': 0.4, 'mean': 0.3}),
    # hierarchical (within/between-image) alignment: the two covariance parts aligned separately, at the pixel site
    # (hier_px), at block 6 (hier_b6), at both (hier_px+hier_b6), and the pooled pixel map with the hierarchical block-6 map
    # (tok+hier_b6); 'within' / 'between' align one part only (ablation at block 6); _n0x prior-strength variants
    'hier_px': dict(tok=True, mean=(), cov=(), headcov=False, tok_mode='hier'),
    'hier_b6': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,), place_mode='hier'),
    'hier_px+hier_b6': dict(tok=True, mean=(), cov=(), headcov=False, tok_mode='hier', place=(5,), place_mode='hier'),
    'tok+hier_b6': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), place_mode='hier'),
    'within_b6': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,), place_mode='within'),
    'between_b6': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,), place_mode='between'),
    'hier_b6_n0x4': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,), place_mode='hier', n0_between_images=256),
    'hier_b6_n0d4': dict(tok=False, mean=(), cov=(), headcov=False, place=(5,), place_mode='hier', n0_between_images=16),
    **{f'hier_b{k}': dict(tok=False, mean=(), cov=(), headcov=False, place=(k - 1,), place_mode='hier') for k in range(1, 13) if k != 6},
    # depth sweep: the same alignment at the output of each block alone (place_b1 .. place_b12; place_b6 is listed above)
    **{f'place_b{k}': dict(tok=False, mean=(), cov=(), headcov=False, place=(k - 1,)) for k in range(1, 13) if k != 6},
    # the pixel site plus one block (tok+place_b1 .. tok+place_b12; tok+place_b6 is listed above)
    **{f'tok+place_b{k}': dict(tok=True, mean=(), cov=(), headcov=False, place=(k - 1,)) for k in range(1, 13) if k != 6},
    'mean_all': dict(tok=False, mean=tuple(range(12)), cov=(), headcov=False),
    'tok+mean_all': dict(tok=True, mean=tuple(range(12)), cov=(), headcov=False),
    'tok+mean_last6': dict(tok=True, mean=tuple(range(6, 12)), cov=(), headcov=False),
    'tok+mean_first6': dict(tok=True, mean=tuple(range(6)), cov=(), headcov=False),
    'tok+mean_all+cov_last4': dict(tok=True, mean=tuple(range(12)), cov=(8, 9, 10, 11), headcov=False),
    'tok+mean_all+cov_all': dict(tok=True, mean=tuple(range(12)), cov=tuple(range(12)), headcov=False),
    'tok+mean_all+headcov': dict(tok=True, mean=tuple(range(12)), cov=(), headcov=True),
    'full': dict(tok=True, mean=tuple(range(12)), cov=(8, 9, 10, 11), headcov=True),
    # ablations of the stack (tokenizer shrinkage / prior, head covariance prior, no head stage)
    'stack_lam0': dict(tok=True, mean=(), cov=(), headcov=True, lam_tok=0.0),
    'stack_lam0.3': dict(tok=True, mean=(), cov=(), headcov=True, lam_tok=0.3),
    'stack_n0x16': dict(tok=True, mean=(), cov=(), headcov=True, n0_tok_images=64),
    'stack_headn0x4': dict(tok=True, mean=(), cov=(), headcov=True, n0_head_batches=16),
    'tok_only_nohead': dict(tok=True, mean=(), cov=(), headcov=False, head='none'),
    # the first online TokAlign run had no constant shrinkage (lam_tok = 0); reported in the supplement (1.06 over NEO)
    'tok_lam0': dict(tok=True, mean=(), cov=(), headcov=False, lam_tok=0.0),
    # diagnostics of the block-mean instability (smoke test): patch tokens only, scaled, subsets
    'mean_all_p': dict(tok=False, mean=tuple(range(12)), cov=(), headcov=False, patch_only=True),
    'mean_all_half': dict(tok=False, mean=tuple(range(12)), cov=(), headcov=False, scale=0.5),
    'mean_first3': dict(tok=False, mean=(0, 1, 2), cov=(), headcov=False),
    'mean_b0': dict(tok=False, mean=(0,), cov=(), headcov=False),
    'mean_last3': dict(tok=False, mean=(9, 10, 11), cov=(), headcov=False),
    'mean_last3_p': dict(tok=False, mean=(9, 10, 11), cov=(), headcov=False, patch_only=True),
}


RULE_ARM = re.compile(r'^rule_t(\d+)_(?:b(\d+)(?:b(\d+))?|(views))(_random)?(_gated)?$')


def rule_arm(name):
    """Arm of the rule-chosen second-opinion stack, named by its blocks (1-based): rule_t<T>_b<A>[b<B>][_gated] is the pixel map,
    the map after block T (the trunk), NEO, and second opinions after blocks A and B for images whose trunk entropy exceeds
    1.5 nat; '_random' routes 35 % of the images at random instead (control); '_gated' adds the spectral gate. rule_t6_b5b9_gated
    is the labelled-search stack of the report (exit_b5b9_gated). rule_t<T>_views[_gated] is the views control: the same routed
    images run the main path on two more views (flip, center crop) instead of two more blocks."""
    m = RULE_ARM.match(name)
    if not m:
        raise KeyError(f'unknown arm {name}')
    t, a, b, views, rnd, g = m.groups()
    arm = dict(tok=True, mean=(), cov=(), headcov=False, place=(int(t) - 1,), exit_sites=tuple(sorted(int(x) - 1 for x in (a, b) if x)), exit_theta=1.5)
    if views:
        arm['exit_views'] = 2
    if rnd:                                                        # control: the method's design share of images routed at random
        arm['exit_random'] = 0.35
    if g:
        arm['gate'] = {'spec': 0.4, 'mean': 0.3}
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='CPU, clean + gaussian_noise-5, 128 images (the block-mean smoke test)')
    ap.add_argument('--model', choices=list(models.CONFIG), default='vitb')
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3', 'heldout4'], default='design')
    ap.add_argument('--dataset', choices=['inc'] + list(data.NATURAL), default='inc',
                    help="'inc': ImageNet-C cells from --imagenet-c (or ImageNet-C-bar cells with --cells); 'r' / 'sketch' / 'a': the natural dataset whose class folders are at --imagenet-c, cells <name>-d1.. (design) or <name>-h1.. (heldout)")
    ap.add_argument('--stats', required=True, help='source statistics (scripts/compute_source_stats.py)')
    ap.add_argument('--tag', default='', help='manifest/logits suffix, e.g. _src50')
    ap.add_argument('--exit-theta', type=float, default=None, help='overrides the entropy threshold of every rule_* arm (the calibrated quantile of the trunk entropy)')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--cells', nargs='*', default=None, help="subset of cells for a spot check (default 'clean' and the 30 cells)")
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    for a in args.arms:
        if a not in ARMS:
            ARMS[a] = rule_arm(a)
        if args.exit_theta is not None and a.startswith('rule_'):
            ARMS[a] = dict(ARMS[a], exit_theta=float(args.exit_theta))
    set_fp32()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cpu' if args.smoke else 'cuda')
    natural = None
    cells = args.cells or ['clean'] + CELLS
    n_eval = N_EVAL
    if args.smoke:
        cells, n_eval = ['clean', 'gaussian_noise-5'], 128
    t0 = time.time()
    model = models.build(args.model).to(dev)
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices(args.split)[:n_eval]
    if args.dataset != 'inc':                                      # a natural dataset: its own cells and labels, the clean cell from ImageNet val
        natural = data.NaturalIndex(args.imagenet_c, index, args.dataset)
        cells = args.cells or ['clean'] + natural.cells(args.split)
    resize = models.clean_resize(args.model)
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'model': args.model, 'dataset': args.dataset, 'batch': BATCH, 'n_eval': n_eval, 'split': args.split, 'stats': args.stats, 'clean_resize': resize,
           'exit_theta': args.exit_theta, 'arms': {a: ARMS[a] for a in args.arms}, 'runs': {}}
    name = f'forward_only_{args.split}_{args.model}{args.tag}{"_smoke" if args.smoke else ""}'
    for cell in cells:
        if data.is_natural_cell(cell):
            imgs, labels = data.load_natural(natural, cell, args.cache_dir, resize), natural.labels_of(cell)
            mask = models.restrict_classes(model, natural.class_labels) if len(natural.class_labels) < 1000 else None
        else:
            imgs, labels, mask = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir, clean_resize=resize), None, None
        for arm in args.arms:
            al = Aligner(model, src, ARMS[arm], dev)
            logits = np.zeros((n_eval, 1000), np.float16)
            branch_a, branch_b = (np.zeros((n_eval, 1000), np.float16) for _ in range(2))
            with torch.no_grad():
                for s in range(0, n_eval, BATCH):
                    x = models.to_input(imgs[s:s + BATCH], dev)
                    if ARMS[arm].get('exit_sites') or ARMS[arm].get('exit_views'):   # early-exit branches (or views): the aligner runs the model itself
                        out = al.forward_exit(model, x)
                    else:
                        if ARMS[arm]['tok']:
                            x = al.tokenizer(x)
                        out = model(x)
                    if not torch.isfinite(out).all():
                        raise FloatingPointError(f'non-finite logits: {arm} {cell}')
                    if ARMS[arm].get('branch'):                   # (2B, 1000): the two branches; saved: mixture + both
                        b = out.shape[0] // 2
                        branch_a[s:s + b], branch_b[s:s + b] = out[:b].half().cpu().numpy(), out[b:].half().cpu().numpy()
                        out = branch_mixture(out)
                    logits[s:s + BATCH] = out.half().cpu().numpy()
            al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'
            extra = {'labels': labels, 'class_labels': np.asarray(natural.class_labels)} if labels is not None else {}
            if ARMS[arm].get('branch'):
                np.savez(path, index=idx, logits=logits, logits_unmapped=branch_a, logits_mapped=branch_b, **extra)
            else:
                np.savez(path, index=idx, logits=logits, **extra)
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve())}
            if ARMS[arm].get('exit_sites') or ARMS[arm].get('exit_views'):
                rec['runs'][arm][cell]['routed_fraction'] = float(np.mean(al.exit_log))
        if mask is not None:
            mask.remove()
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
