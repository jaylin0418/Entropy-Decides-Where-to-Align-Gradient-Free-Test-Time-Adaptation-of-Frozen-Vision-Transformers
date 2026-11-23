"""Wall-clock of the other gradient-free methods (GPU; no labels): CUDA-synchronised time per batch of 64 of NEO, PEA (the
original view only, and with the three augmented views) and FOA on the six design timing cells of time_batches.py, 2,000
images per cell, the first two batches of each cell dropped. NEO is timed in the same run as the reference, so that
score_timing.py gives every arm's time relative to it. Logits are saved so that the same script scores the arms. Needs the
official PEA and FOA code (third_party/*/SOURCE.md), timm >= 0.9 and cma for FOA, so it runs in the FOA environment
(PY_TIMM1 of reproduce_paper.sh). PEA's and FOA's source statistics come from the 3,000 clean source images, PEA with the
imagenet_c hyper-parameters of its main_vit.py and its entropy-spike reset, FOA with its official configuration (3 prompts,
population 27, lambda 0.4), both reset per cell as in the episodic runs. Writes <out-dir>/timing_design_vitb_others.json.

    $PY_TIMM1 scripts/cost/time_others.py --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
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
from scripts.cost.time_batches import ARMS as TIMING_ARMS, CELLS  # noqa: E402
from scripts.episodic.run_foa import build_vit  # noqa: E402  (puts third_party/foa on the path)
from scripts.episodic.run_pea import HP, SrcIter, entropy_mean, load_pea  # noqa: E402
from tokalign import data, models  # noqa: E402
from tokalign.aligner import BATCH, Aligner, load_source, set_fp32  # noqa: E402

ARMS = ('neo', 'pea', 'pea_aug', 'foa')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='one cell, 128 images (GPU)')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--n-src', type=int, default=3000)
    ap.add_argument('--stats', required=True, help='source statistics of ViT-B/16 (for NEO)')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    ap.add_argument('--repeats', type=int, default=1, help='passes over the arms per cell (the order reversed on every second pass), batches pooled')
    args = ap.parse_args()
    from models.vpt import PromptViT  # noqa: E402  (third_party/foa)
    from tta_library.foa import FOA  # noqa: E402
    pea_mod = load_pea()
    set_fp32()                                          # pea_vit.py switches TF32 on at import; the runs are FP32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cuda')
    cells, n_eval, n_src = (CELLS[:1], 128, 256) if args.smoke else (CELLS, 2000, args.n_src)
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    clean = data.load_images(index, 'clean', data.split_indices('source')[:n_src], args.imagenet_c, args.cache_dir)
    t0 = time.time()
    model = models.build('vitb').to(dev).eval()
    with contextlib.redirect_stderr(io.StringIO()):
        means, vars_, sqrts = pea_mod.compute_source_stats_vit(model, SrcIter(clean, dev), dev, use_cls_only=False)
    del model
    vit = build_vit(dev)
    foa = FOA(PromptViT(vit, 3).to(dev), fitness_lambda=0.4)
    foa.imagenet_mask = None
    foa.obtain_origin_stat([(models.to_input(clean[s:s + BATCH], dev), None) for s in range(0, len(clean), BATCH)])
    print(f'PEA and FOA source statistics from {len(clean)} images: {time.time() - t0:.0f}s', flush=True)
    idx = data.split_indices('design')[:n_eval]
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'batch': BATCH, 'n_eval': n_eval, 'source_images': len(clean), 'pea': {**HP, 'update_every': 1}, 'foa': {'num_prompts': 3, 'fitness_lambda': 0.4, 'popsize': 27}, 'runs': {}}
    name = f'timing_design_vitb_others{"_smoke" if args.smoke else ""}'
    rec['repeats'] = args.repeats
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        pooled = {}
        for rep in range(args.repeats):
          for arm in (args.arms if rep % 2 == 0 else args.arms[::-1]):
            al = None
            if arm == 'neo':                                                        # the same NEO arm as time_batches.py
                model = models.build('vitb').to(dev).eval(); al = Aligner(model, src, TIMING_ARMS['neo'], dev)

                def step(x):
                    with torch.no_grad():
                        return model(x)
            elif arm in ('pea', 'pea_aug'):
                model = models.build('vitb').to(dev).eval()
                state = {'stats': pea_mod.PEAStatsViT(**HP)}
                pea = pea_mod.PEAViT(model, means, vars_, sqrts, state['stats'], use_cls_only=False, group_size=model.embed_dim, update_every=1).to(dev).eval()

                def step(x, aug=(arm == 'pea_aug')):
                    with torch.no_grad():
                        out, _ = pea_mod.pea_vit_infer(model, pea, x, dev, weight_use_cls_only=True, use_augmentation=aug, n_aug_max=2,
                                                       center_crop_ratio=0.9, do_hflip=True, include_original=True, microbatch_variants=1)
                        ent = entropy_mean(out)
                        if state['stats'].entropy_spike(ent):                       # PEA's own reset, part of its step
                            state['stats'] = state['stats'].reset(); pea.stats = state['stats']; pea.reset_state()
                        state['stats'].update_entropy(ent)
                    return out
            elif arm == 'foa':
                foa.reset()

                def step(x):
                    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                        return foa(x)
            else:
                raise KeyError(arm)
            logits = np.zeros((n_eval, 1000), np.float16); times = []
            for s in range(0, n_eval, BATCH):
                x = models.to_input(imgs[s:s + BATCH], dev); torch.cuda.synchronize(); t1 = time.perf_counter()
                out = step(x)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t1)
                if not torch.isfinite(out).all():
                    raise FloatingPointError(f'non-finite logits: {arm} {cell}')
                logits[s:s + BATCH] = out.half().cpu().numpy()
            if al is not None:
                al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'; np.savez(path, index=idx, logits=logits)
            pooled.setdefault(arm, []).extend(np.array(times[2:]) * 1000)
            ms = np.array(pooled[arm])
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve()), 'ms_per_batch_mean': float(ms.mean()), 'ms_per_batch_median': float(np.median(ms)),
                                                     'passes': rep + 1, 'ms_per_pass': [float(np.mean(ms[i * (len(ms) // (rep + 1)):(i + 1) * (len(ms) // (rep + 1))])) for i in range(rep + 1)]}
            print(f'{cell}/{arm} pass {rep + 1}: {np.mean(np.array(times[2:]) * 1000):.1f} ms/batch', flush=True)
            if arm != 'foa':
                del model
            torch.cuda.empty_cache()
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
