"""Episodic FOA runs (GPU; no labels): official FOA (fetched into third_party/foa, see its SOURCE.md; non-commercial
licence, research evaluation only) on ViT-B/16 per design cell, with and without TokAlign in front. Needs timm >= 0.9
(the runs used timm 1.0.15) for FOA's own model code. FOA's source statistics come from 3,000 clean images (split
'source'); FOA is reset per cell. 2,000 images per cell (split 'design', 31 batches) or 6,000 (split 'design_extended',
94 batches). Manifest <out-dir>/foa_<split>_vitb.json. (Research name: scripts/spa/probe_foa.py.)

    python scripts/episodic/run_foa.py --n-eval 2000 --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL \\
        --imagenet-c $IMAGENET_C --out-dir runs
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party' / 'foa'))

import timm  # noqa: E402

from tokalign import data, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, Aligner, load_source, set_fp32  # noqa: E402

CELLS = data.CELLS
TAG = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
ARMS = {'foa': False, 'tok+foa': True}


def build_vit(dev):
    vit = timm.create_model(TAG, pretrained=True).to(dev).eval()
    vit.requires_grad_(False)
    return vit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cells', choices=['sev5', 'all'], default='sev5')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--n-eval', type=int, default=N_EVAL, help='2000 or 6000 design images per cell')
    ap.add_argument('--stats', required=True, help='source statistics of ViT-B/16 (for TokAlign)')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    set_fp32()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cuda')
    from models.vpt import PromptViT  # noqa: E402  (third_party/foa)
    from tta_library.foa import FOA  # noqa: E402
    cells = [c for c in CELLS if c.endswith('-5')] if args.cells == 'sev5' else list(CELLS)
    cells = ['clean'] + cells
    n_eval = args.n_eval
    assert n_eval in (2000, 6000)
    if args.smoke:
        cells, n_eval = ['clean', 'gaussian_noise-5'], 128
    t0 = time.time()
    vit = build_vit(dev)
    src = load_source(args.stats, dev)
    net = PromptViT(vit, 3).to(dev)
    foa = FOA(net, fitness_lambda=0.4)
    foa.imagenet_mask = None
    index = data.ImageNetIndex(args.imagenet_val)
    src_idx = data.split_indices('source')[:3000 if not args.smoke else 256]
    clean = data.load_images(index, 'clean', src_idx, args.imagenet_c, args.cache_dir)
    foa.obtain_origin_stat([(models.to_input(clean[s:s + BATCH], dev), None) for s in range(0, len(clean), BATCH)])
    print(f'FOA source statistics from {len(clean)} images; {time.time() - t0:.0f}s', flush=True)
    split = 'design' if n_eval <= 2000 else 'design_extended'
    idx = data.split_indices(split)[:n_eval]
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'model': TAG, 'batch': BATCH, 'n_eval': n_eval, 'split': split, 'foa': {'num_prompts': 3, 'fitness_lambda': 0.4,
           'popsize': 27, 'source_images': len(clean)}, 'runs': {}}
    name = f'foa_{split}_vitb{"_smoke" if args.smoke else ""}'
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        for arm in args.arms:
            foa.reset()
            al = Aligner(vit, src, dict(tok=True, mean=(), cov=(), head='none'), dev) if ARMS[arm] else None
            logits = np.zeros((n_eval, 1000), np.float16)
            tc = time.time()
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                for s in range(0, n_eval, BATCH):
                    x = models.to_input(imgs[s:s + BATCH], dev)
                    if al is not None:
                        x = al.tokenizer(x)
                    out = foa(x)
                    if not torch.isfinite(out).all():
                        raise FloatingPointError(f'non-finite logits: {arm} {cell}')
                    logits[s:s + BATCH] = out.half().cpu().numpy()
            path = store / f'{name}_{arm}_{cell}.npz'
            np.savez(path, index=idx, logits=logits)
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve()), 'seconds': round(time.time() - tc, 1)}
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
