"""Episodic PEA runs (GPU; no labels): official PEA (fetched into third_party/pea, see its SOURCE.md; no licence,
research evaluation only) on ViT-B/16 per design cell, with and without TokAlign in front. Per cell: fresh PEA
statistics (the repository's each_shift_reset), the entropy-spike reset of its main_vit.py, its imagenet_c
hyper-parameters (momentum 0.02, alpha 0.2, gamma 4.0, entropy threshold 1.0, group = embed dim, update every batch),
views 'orig' only (`pea`, 2 forwards per batch) or orig + centre-crop 0.9 + h-flip (`pea_aug`, 6 forwards per batch).
PEA's source statistics: 3,000 clean images (split 'source', the same as ours) or 20,000 (split 'pea_source',
--n-src 20000 --tag _src20k). Manifest <out-dir>/pea_design_<model><tag>.json. (Research name: scripts/spa/probe_pea.py.)

    python scripts/episodic/run_pea.py --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tokalign import data, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, Aligner, load_source, set_fp32  # noqa: E402

CELLS = data.CELLS
ARMS = {'pea': (False, False), 'pea_aug': (False, True), 'tok+pea': (True, False), 'tok+pea_aug': (True, True)}
HP = dict(momentum=0.02, alpha=0.2, gamma=4.0, entropy_threshold=1.0)
PEA_COMMIT = '9c358ba682052862df0739fbbadb71f1b586614d'


def load_pea():
    spec = importlib.util.spec_from_file_location('tokalign_pea_vit', str(ROOT / 'third_party' / 'pea' / 'pea_vit.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['tokalign_pea_vit'] = mod
    spec.loader.exec_module(mod)
    return mod


class SrcIter:
    """Re-iterable source loader for PEA's statistics (it iterates once per block)."""

    def __init__(self, imgs, dev):
        self.imgs, self.dev = imgs, dev

    def __iter__(self):
        for s in range(0, len(self.imgs), BATCH):
            yield models.to_input(self.imgs[s:s + BATCH], self.dev), None


def entropy_mean(logits):
    p = torch.softmax(logits.float(), dim=1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--cells', nargs='*', default=None, help='subset of cells (default clean + 30)')
    ap.add_argument('--tag', default='', help='manifest suffix, e.g. _src20k')
    ap.add_argument('--src-split', default='source', help="clean images for PEA's source statistics: 'source' or 'pea_source'")
    ap.add_argument('--n-src', type=int, default=3000)
    ap.add_argument('--stats', required=True, help='our source statistics (for TokAlign)')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    pea_mod = load_pea()
    set_fp32()                                          # pea_vit.py switches TF32 on at import; our runs are FP32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cpu' if args.smoke else 'cuda')
    cells, n_eval, n_src = (['clean', 'gaussian_noise-5'], 128, 128) if args.smoke else (args.cells or ['clean'] + CELLS, N_EVAL, args.n_src)
    t0 = time.time()
    model = models.build(args.model).to(dev).eval()
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    clean = data.load_images(index, 'clean', data.split_indices(args.src_split)[:n_src], args.imagenet_c, args.cache_dir)
    with contextlib.redirect_stderr(io.StringIO()):
        means, vars_, sqrts = pea_mod.compute_source_stats_vit(model, SrcIter(clean, dev), dev, use_cls_only=False)
    print(f'PEA source statistics from {n_src} images: {time.time() - t0:.0f}s', flush=True)
    idx = data.split_indices('design')[:n_eval]
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'model': args.model, 'batch': BATCH, 'n_eval': n_eval, 'split': 'design', 'pea': {**HP, 'group_size': model.embed_dim,
           'update_every': 1, 'source_images': n_src, 'source_split': args.src_split, 'upstream': PEA_COMMIT}, 'runs': {}}
    name = f'pea_design_{args.model}{args.tag}{"_smoke" if args.smoke else ""}'
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        for arm in args.arms:
            use_tok, aug = ARMS[arm]
            stats = pea_mod.PEAStatsViT(**HP)
            pea = pea_mod.PEAViT(model, means, vars_, sqrts, stats, use_cls_only=False, group_size=model.embed_dim, update_every=1).to(dev).eval()
            al = Aligner(model, src, dict(tok=True, mean=(), cov=(), head='none'), dev) if use_tok else None
            logits = np.zeros((n_eval, 1000), np.float16); resets = 0; tc = time.time()
            with torch.no_grad():
                for s in range(0, n_eval, BATCH):
                    x = models.to_input(imgs[s:s + BATCH], dev)
                    if al is not None:
                        x = al.tokenizer(x)
                    out, _ = pea_mod.pea_vit_infer(model, pea, x, dev, weight_use_cls_only=True, use_augmentation=aug, n_aug_max=2,
                                                   center_crop_ratio=0.9, do_hflip=True, include_original=True, microbatch_variants=1)
                    if not torch.isfinite(out).all():
                        raise FloatingPointError(f'non-finite logits: {arm} {cell}')
                    logits[s:s + BATCH] = out.half().cpu().numpy()
                    ent = entropy_mean(out)
                    if stats.entropy_spike(ent):
                        stats = stats.reset(); pea.stats = stats; pea.reset_state(); resets += 1
                    stats.update_entropy(ent)
            if al is not None:
                al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'
            np.savez(path, index=idx, logits=logits)
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve()), 'resets': resets, 'seconds': round(time.time() - tc, 1)}
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
