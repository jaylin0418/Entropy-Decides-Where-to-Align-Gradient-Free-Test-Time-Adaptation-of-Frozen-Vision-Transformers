"""Wall-clock benchmark (GPU; no labels): CUDA-synchronised time per batch of 64 of NEO, the gated stack (and the two-site stack) with the maps
refreshed every k batches (float64 or float32 eigendecompositions, float64 or float32 statistics), ETA and SAR, on six design
cells; the first two batches of each cell are dropped. With --repeats R every arm is timed R times per cell, the order of the arms
reversed on every second pass, and the batches of all passes are pooled, so that a busy spell of the shared GPU does not land on one arm. Arms rule_t<T>_b<A>[b<B>][_random]_k16_fp32_stats32 time the calibrated
second-opinion stack, one second opinion alone and the random-routing control (rule_arm). Logits are saved so the accuracy of every variant is scored
offline. Writes <out-dir>/timing_design_<model><tag>.json. (Research name: scripts/spa/bench_wallclock.py; the cost
table is its single run with --arms neo stack_k1 stack_k16_fp32 stack_k16_fp32_stats32 eta.)

    python scripts/cost/time_batches.py --arms neo stack_k1 stack_k16_fp32 stack_k16_fp32_stats32 eta \\
        --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data, hosts, models  # noqa: E402
from tokalign.aligner import BATCH, Aligner, branch_mixture, load_source, set_fp32  # noqa: E402

CELLS = ['gaussian_noise-5', 'defocus_blur-3', 'snow-5', 'contrast-5', 'pixelate-3', 'jpeg_compression-5']
BASE = dict(tok=True, mean=(), cov=(), headcov=True, gate={'spec': 0.4, 'mean': 0.3})
ARMS = {'neo': dict(tok=False, mean=(), cov=(), headcov=False),
        'stack_k1': dict(BASE, refresh_every=1), 'stack_k4': dict(BASE, refresh_every=4), 'stack_k16': dict(BASE, refresh_every=16),
        'stack_k1_fp32': dict(BASE, refresh_every=1, fp32_eig=True), 'stack_k16_fp32': dict(BASE, refresh_every=16, fp32_eig=True),
        'stack_k16_fp32_stats32': dict(BASE, refresh_every=16, fp32_eig=True, stats32=True),
        'eta': dict(tok=False, mean=(), cov=(), head='none', eta=True),
        'sar': dict(tok=False, mean=(), cov=(), head='none', sar=True),
        # the pixel map alone (TokAlign in front of NEO) at the refresh settings of the timed stacks
        'tok_k16_fp32_stats32': dict(tok=True, mean=(), cov=(), headcov=False, refresh_every=16, fp32_eig=True, stats32=True),
        # the two-site stack (pixel map, block-6 map, head covariance, gate) at the same refresh settings, and the two sites inside ETA
        'twosite_k1': dict(BASE, refresh_every=1, place=(5,)),
        'twosite_k16_fp32_stats32': dict(BASE, refresh_every=16, fp32_eig=True, stats32=True, place=(5,)),
        'eta+twosite': dict(tok=True, mean=(), cov=(), head='none', eta=True, place=(5,)),
        # the pair without the head covariance stage and gate (pixel map + block-6 map + NEO) and the branch mixture at the same
        # refresh settings: blocks 1 to 6 once, blocks 7 to 12 on both branches, entropy-weighted mixture (1.5 forward passes)
        'pair_k16_fp32_stats32': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), refresh_every=16, fp32_eig=True, stats32=True),
        'branch_k16_fp32_stats32': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), branch=True, refresh_every=16, fp32_eig=True, stats32=True),
        # early-exit branches at the same refresh settings (extra sites: block 9; blocks 5 and 9)
        'exit_b9_k16_fp32_stats32': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(8,), exit_theta=1.5, refresh_every=16, fp32_eig=True, stats32=True),
        'exit_b5b9_k16_fp32_stats32': dict(tok=True, mean=(), cov=(), headcov=False, place=(5,), exit_sites=(4, 8), exit_theta=1.5, refresh_every=16, fp32_eig=True, stats32=True)}


def rule_arm(name):
    """Timing arm of the rule-chosen second-opinion stack: rule_t<T>_b<A>b<B>_k16_fp32_stats32 (the settings of the other
    timed stacks: maps refreshed every 16 batches, float32 eigendecomposition, float32 statistics)."""
    from scripts.episodic.run_forward_only import rule_arm as episodic_rule_arm
    suffix = '_k16_fp32_stats32'
    if not name.endswith(suffix):
        raise KeyError(f'unknown arm {name}')
    a = episodic_rule_arm(name[:-len(suffix)])
    arm = dict(tok=True, mean=(), cov=(), headcov=False, place=a['place'], exit_sites=a['exit_sites'], exit_theta=a['exit_theta'], refresh_every=16, fp32_eig=True, stats32=True)
    if 'exit_random' in a:                                         # the random-routing control keeps its fraction
        arm['exit_random'] = a['exit_random']
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--tag', default='')
    ap.add_argument('--stats', required=True)
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    ap.add_argument('--repeats', type=int, default=1, help='passes over the arms per cell (the order reversed on every second pass), batches pooled')
    args = ap.parse_args()
    for a in args.arms:
        if a not in ARMS:
            ARMS[a] = rule_arm(a)
    set_fp32(); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cpu' if args.smoke else 'cuda')
    cells, n_eval = (CELLS[:1], 128) if args.smoke else (CELLS, 2000)
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices('design')[:n_eval]
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'batch': BATCH, 'n_eval': n_eval, 'repeats': args.repeats, 'runs': {}}
    name = f'timing_design_{args.model}{args.tag}{"_smoke" if args.smoke else ""}'
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        pooled = {}
        for rep in range(args.repeats):
          for arm in (args.arms if rep % 2 == 0 else args.arms[::-1]):
            cfg = ARMS[arm]
            model = models.build(args.model).to(dev); al = Aligner(model, src, cfg, dev)
            host = hosts.ETA(model, BATCH) if cfg.get('eta') else hosts.SAR(model, BATCH) if cfg.get('sar') else None
            logits = np.zeros((n_eval, 1000), np.float16); times = []
            for s in range(0, n_eval, BATCH):
                x = models.to_input(imgs[s:s + BATCH], dev); sync(); t0 = time.perf_counter()
                if cfg.get('exit_sites'):
                    out = al.forward_exit(model, x)
                elif host is not None:
                    if cfg['tok']:
                        with torch.no_grad():
                            x = al.tokenizer(x)
                    out = host(x)
                else:
                    if cfg['tok']:
                        with torch.no_grad():
                            x = al.tokenizer(x)
                    with torch.no_grad():
                        out = model(x)
                        if cfg.get('branch'):
                            out = branch_mixture(out)                 # the mixture is part of the timed step
                sync(); times.append(time.perf_counter() - t0)
                logits[s:s + BATCH] = out.half().cpu().numpy()
            al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'; np.savez(path, index=idx, logits=logits)
            pooled.setdefault(arm, []).extend(np.array(times[2:]) * 1000)
            ms = np.array(pooled[arm])
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve()), 'ms_per_batch_mean': float(ms.mean()), 'ms_per_batch_median': float(np.median(ms)),
                                                     'passes': rep + 1, 'ms_per_pass': [float(np.mean(ms[i * (len(ms) // (rep + 1)):(i + 1) * (len(ms) // (rep + 1))])) for i in range(rep + 1)]}
            print(f'{cell}/{arm} pass {rep + 1}: {np.mean(np.array(times[2:]) * 1000):.1f} ms/batch', flush=True)
            del model, host
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')


if __name__ == '__main__':
    main()
