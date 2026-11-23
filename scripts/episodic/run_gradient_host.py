"""Episodic runs of a gradient host (GPU; no labels): official ETA (EATA without the Fisher term) or SAR, fresh weights
per cell, with (`tok+eta`, `tok+sar`) and without TokAlign in front, and with the block-6 site of the two-site method
(`eta+place_b6`, `tok+eta+place_b6`), and with the calibrated second-opinion stack inside the host
(`tok+eta+rule_t<T>_b<E>b<L>` with `--exit-theta`: tokalign.aligner.HostStack). Records per image the host's (pre-update)
logits and the head input, so NEO and the head covariance stage are applied when scoring; for the stack also the
confidence-weighted average of the routed rows. Manifest
<out-dir>/<method>_<split>_<model>.json. (Research name: scripts/spa/probe_eta.py.)

    python scripts/episodic/run_gradient_host.py --method eta --split heldout --stats stats/vitb.pt \\
        --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
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
from tokalign import data, hosts, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, Aligner, HostStack, load_source, set_fp32  # noqa: E402

CELLS = data.CELLS
_TOK, _B6, _BOTH = dict(tok=True, mean=(), cov=(), head='none'), dict(tok=False, mean=(), cov=(), head='none', place=(5,)), dict(tok=True, mean=(), cov=(), head='none', place=(5,))
ARMS = {'eta': None, 'tok+eta': _TOK, 'eta+place_b6': _B6, 'tok+eta+place_b6': _BOTH,     # stages in front of / inside the host
        'sar': None, 'tok+sar': _TOK, 'sar+place_b6': _B6, 'tok+sar+place_b6': _BOTH,
        # depth sweep on the host: the alignment at each block alone
        **{f'eta+place_b{k}': dict(tok=False, mean=(), cov=(), head='none', place=(k - 1,)) for k in range(1, 25) if k != 6},
        # the pixel map and the map after the calibrated block inside the host (any block; block 6 is the pair of ViT-B/16)
        **{f'tok+{m}+place_b{k}': dict(tok=True, mean=(), cov=(), head='none', place=(k - 1,)) for m in ('eta', 'sar') for k in range(1, 25) if k != 6},
        # hierarchical (within/between-image) alignment inside the host: pooled pixel map + hierarchical block-6 map, and both hierarchical
        'tok+eta+hier_b6': dict(tok=True, mean=(), cov=(), head='none', place=(5,), place_mode='hier'),
        'hier_px+eta+hier_b6': dict(tok=True, mean=(), cov=(), head='none', tok_mode='hier', place=(5,), place_mode='hier')}


STACK_ARM = re.compile(r'^tok\+(eta|sar)\+rule_t(\d+)_b(\d+)b(\d+)$')


def stack_arm(name, theta):
    """tok+<host>+rule_t<T>_b<E>b<L>: the pixel map and the map after block T inside the host, second opinions after blocks E
    and L (1-based) for the images whose NEO-corrected base-path entropy exceeds theta (the calibrated quantile)."""
    m = STACK_ARM.match(name)
    if not m:
        return None
    _, t, e, l = m.groups()
    return dict(tok=True, mean=(), cov=(), head='none', place=(int(t) - 1,), exit_sites=tuple(sorted((int(e) - 1, int(l) - 1))), exit_theta=float(theta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--model', choices=list(models.CONFIG), default='vitb')
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3', 'heldout4'], default='design')
    ap.add_argument('--dataset', choices=['inc'] + list(data.NATURAL), default='inc', help='as run_forward_only.py')
    ap.add_argument('--method', choices=['eta', 'sar'], default='eta')
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--cells', nargs='*', default=None, help="subset of cells for a spot check (default 'clean' and the 30 cells)")
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    ap.add_argument('--tag', default='', help='manifest/logits suffix, e.g. _sweep')
    ap.add_argument('--exit-theta', type=float, default=1.5, help='entropy threshold of the stack arms tok+<host>+rule_t<T>_b<E>b<L>')
    args = ap.parse_args()
    for a in args.arms or []:
        if a not in ARMS:
            ARMS[a] = stack_arm(a, args.exit_theta)
            if ARMS[a] is None:
                ap.error(f'unknown arm {a}')
    if args.arms is None:
        args.arms = [args.method, f'tok+{args.method}']
    set_fp32()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cells, n_eval = (['clean', 'gaussian_noise-5'], 128) if args.smoke else (args.cells or ['clean'] + CELLS, N_EVAL)
    t0 = time.time()
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices(args.split)[:n_eval]
    natural = None
    if args.dataset != 'inc':
        natural = data.NaturalIndex(args.imagenet_c, index, args.dataset)
        cells = args.cells or ['clean'] + natural.cells(args.split)
    resize = models.clean_resize(args.model)
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'model': args.model, 'dataset': args.dataset, 'batch': BATCH, 'n_eval': n_eval, 'split': args.split, 'stats': args.stats, 'clean_resize': resize,
           'arms': {a: ARMS[a] for a in args.arms}, 'exit_theta': args.exit_theta, 'runs': {}}
    name = f'{args.method}_{args.split}_{args.model}{args.tag}{"_smoke" if args.smoke else ""}'
    for cell in cells:
        if data.is_natural_cell(cell):
            imgs, labels = data.load_natural(natural, cell, args.cache_dir, resize), natural.labels_of(cell)
        else:
            imgs, labels = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir, clean_resize=resize), None
        for arm in args.arms:
            model = models.build(args.model).to(dev)                     # fresh weights per cell and arm (episodic)
            mask = models.restrict_classes(model, natural.class_labels) if labels is not None and len(natural.class_labels) < 1000 else None
            tap = models.HeadInput(model)
            al = Aligner(model, src, ARMS[arm], dev) if ARMS[arm] else None
            n_cls = len(natural.class_labels) if labels is not None else 1000
            stack = HostStack(model, al) if al is not None and al.exit else None      # the stack inside the host: its forward is the base path
            hosted = stack if stack is not None else model
            host = hosts.SAR(hosted, BATCH, n_classes=n_cls) if args.method == 'sar' else hosts.ETA(hosted, BATCH, n_classes=n_cls)
            logits = np.zeros((n_eval, 1000), np.float16); h = np.zeros((n_eval, model.embed_dim), np.float16)
            mix, routed = (np.full((n_eval, 1000), np.nan, np.float16), np.zeros(n_eval, bool)) if stack is not None else (None, None)
            for s in range(0, n_eval, BATCH):
                x = models.to_input(imgs[s:s + BATCH], dev)
                if al is not None and ARMS[arm]['tok']:
                    with torch.no_grad():
                        x = al.tokenizer(x)
                tap.h = None
                out = host(x)
                logits[s:s + BATCH] = out.half().cpu().numpy()
                if stack is not None:
                    h[s:s + BATCH] = stack.h.half().cpu().numpy(); mix[s:s + BATCH] = stack.mix.half().cpu().numpy(); routed[s:s + BATCH] = stack.routed
                else:
                    h[s:s + BATCH] = tap.h.half().cpu().numpy()
            tap.remove()
            if mask is not None:
                mask.remove()
            if al is not None:
                al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'
            np.savez(path, index=idx, logits=logits, h=h, **({'labels': labels, 'class_labels': np.asarray(natural.class_labels)} if labels is not None else {}),
                     **({'mix': mix, 'routed': routed} if stack is not None else {}))
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve()), 'eta_updated_samples': host.n_updated_samples(),
                                                     'sar_resets': getattr(host, 'resets', 0),
                                                     **({'routed_fraction': float(routed.mean())} if stack is not None else {})}
            del model, host
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
