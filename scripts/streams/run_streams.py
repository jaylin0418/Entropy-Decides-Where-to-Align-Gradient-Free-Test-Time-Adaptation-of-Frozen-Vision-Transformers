"""Stream runs (GPU; no labels): one model, one host and one set of running statistics for a whole stream, EMA
statistics, the spectral gate, no resets. Streams: continual (cont5, cont3, alt), clean-return, clean, recurring
(recur0-2) and per-batch mixed. Arms: NEO-EMA, the ungated and gated stack, ETA or SAR (one instance per stream) with
and without gated TokAlign, the per-image tokenizer variant, and the main-path arms (block-6 site added, gated). Records logits, head input and the gate flags;
manifest <out-dir>/streams_<split>_<model><tag>.json. (Research name: scripts/spa/probe_streams.py.)

    python scripts/streams/run_streams.py --split heldout --stats stats/vitb.pt --imagenet-val $IMAGENET_VAL \\
        --imagenet-c $IMAGENET_C --out-dir runs
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data, hosts, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, Aligner, load_source, set_fp32  # noqa: E402

ORDER = list(data.CORRUPTIONS)
GATE = {'spec': 0.4, 'mean': 0.3}
EMA = dict(ema_tok=0.2, ema_head=0.5, ema_headcov=0.1)
ARMS = {
    'neo': dict(tok=False, mean=(), cov=(), headcov=False, eta=False, **EMA),
    'stack': dict(tok=True, mean=(), cov=(), headcov=True, eta=False, **EMA),
    'stack_gated': dict(tok=True, mean=(), cov=(), headcov=True, eta=False, gate=GATE, **EMA),
    'eta': dict(tok=False, mean=(), cov=(), head='none', eta=True, **EMA),
    'tok+eta_gated': dict(tok=True, mean=(), cov=(), head='none', eta=True, gate=GATE, **EMA),
    'sar': dict(tok=False, mean=(), cov=(), head='none', eta='sar', **EMA),
    'tok+sar_gated': dict(tok=True, mean=(), cov=(), head='none', eta='sar', gate=GATE, **EMA),
    'tokimg+headcov_gated': dict(tok=True, tok_mode='image', mean=(), cov=(), headcov=True, eta=False, gate=GATE, **EMA),
    # main-path arms: TokAlign's map also on the patch tokens after block 6 (0-based 5), same EMA, same gate
    'twosite_gated': dict(tok=True, mean=(), cov=(), headcov=True, eta=False, gate=GATE, place=(5,), **EMA),
    'twosite+eta_gated': dict(tok=True, mean=(), cov=(), head='none', eta=True, gate=GATE, place=(5,), **EMA),
    # hierarchical (within/between-image) alignment at block 6 behind the pooled pixel map, gated, EMA statistics
    'hier_gated': dict(tok=True, mean=(), cov=(), headcov=True, eta=False, gate=GATE, place=(5,), place_mode='hier', **EMA),
    'hier+eta_gated': dict(tok=True, mean=(), cov=(), head='none', eta=True, gate=GATE, place=(5,), place_mode='hier', **EMA),
    # v2: the pair without the head stage, gated, and the cascade (tokalign.aligner.forward_exit), gated, EMA statistics
    'pair_gated': dict(tok=True, mean=(), cov=(), headcov=False, eta=False, gate=GATE, place=(5,), **EMA),
    'exit_b5b9_gated': dict(tok=True, mean=(), cov=(), headcov=False, eta=False, gate=GATE, place=(5,), exit_sites=(4, 8), exit_theta=1.5, **EMA),
}


def streams(smoke, n_seg=N_EVAL):
    """Each stream is a list of (cell, row_start, row_end) batches; segment streams are contiguous, 'mixed' interleaves
    batches of all 30 cells at random (seed 0), 'recur<k>' are random orders of the 30 cells (seeds 0-2)."""
    std = [f'{c}-{s}' for c in ORDER for s in data.SEVERITIES]
    S = {'cont5': [f'{c}-5' for c in ORDER], 'cont3': [f'{c}-3' for c in ORDER],
         'alt': [f'{c}-{s}' for c, s in zip(ORDER, itertools.cycle([5, 3]))]}
    rng = np.random.default_rng(0)
    five = [std[i] for i in rng.choice(len(std), 5, replace=False)]
    S['cleanret'] = [x for c in five for x in ('clean', c)]
    S['clean'] = ['clean']
    for seed in range(3):
        S[f'recur{seed}'] = [std[i] for i in np.random.default_rng(seed).permutation(len(std))]
    if smoke:
        S = {'cleanret': ['clean', 'gaussian_noise-5', 'clean'], 'mixed': None}
    B = {k: [(c, s, min(s + BATCH, n_seg)) for c in v for s in range(0, n_seg, BATCH)] for k, v in S.items() if v is not None}
    allb = [(c, s, min(s + BATCH, n_seg)) for c in (std if not smoke else ['clean', 'gaussian_noise-5']) for s in range(0, n_seg, BATCH)]
    B['mixed'] = [allb[i] for i in np.random.default_rng(0).permutation(len(allb))]
    return B


def rule_arm(name):
    """Stream arm of the rule-chosen cascade, named as in run_forward_only.py (rule_t<T>_b<A>b<B>_gated), with the
    stream statistics (EMA) and the gate."""
    from scripts.episodic.run_forward_only import rule_arm as episodic_rule_arm
    a = episodic_rule_arm(name)
    return dict(tok=True, mean=(), cov=(), headcov=False, eta=False, gate=GATE, place=a['place'], exit_sites=a['exit_sites'], exit_theta=a['exit_theta'], **EMA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3', 'heldout4'], default='design')
    ap.add_argument('--tag', default='', help='manifest suffix, e.g. _sar')
    ap.add_argument('--arms', nargs='*', default=[a for a in ARMS if 'sar' not in a and 'tokimg' not in a and 'twosite' not in a])
    ap.add_argument('--streams', nargs='*', default=['cont5', 'cont3', 'alt', 'cleanret', 'clean'])
    ap.add_argument('--stats', required=True)
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    for a in args.arms:
        if a not in ARMS:
            ARMS[a] = rule_arm(a)
    set_fp32()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cpu' if args.smoke else 'cuda')
    n_seg = 128 if args.smoke else N_EVAL
    t0 = time.time()
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    idx_seg = data.split_indices(args.split)[:n_seg]
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    rec = {'model': args.model, 'batch': BATCH, 'n_seg': n_seg, 'split': args.split, 'stats': args.stats, 'gate': GATE, 'ema': EMA,
           'arms': {a: {k: v for k, v in ARMS[a].items() if k != 'gate'} for a in args.arms}, 'runs': {}}
    name = f'streams_{args.split}_{args.model}{args.tag}{"_smoke" if args.smoke else ""}'
    all_streams = streams(args.smoke, n_seg)
    for sname in (['cleanret', 'mixed'] if args.smoke else args.streams):
        batches = all_streams[sname]
        segs = [c for i, (c, s, e) in enumerate(batches) if i == 0 or batches[i - 1][0] != c]   # contiguous runs (segments)
        for arm in args.arms:
            cfg = ARMS[arm]
            model = models.build(args.model).to(dev)
            tap = models.HeadInput(model)
            al = Aligner(model, src, cfg, dev)
            eta = (hosts.SAR(model, BATCH) if cfg['eta'] == 'sar' else hosts.ETA(model, BATCH)) if cfg['eta'] else None
            n_total = sum(e - s for _, s, e in batches)
            logits = np.zeros((n_total, 1000), np.float16); h = np.zeros((n_total, model.embed_dim), np.float16)
            index_all = np.zeros(n_total, np.int64); cell_of = segs; batch_cells = []
            pos = 0
            caches = {}
            for cell, s, e in batches:
                if cell not in caches:
                    caches[cell] = data.load_images(index, cell, idx_seg, args.imagenet_c, args.cache_dir)
                imgs = caches[cell]
                batch_cells.append(cell)
                x = models.to_input(imgs[s:e], dev)
                tap.h = None
                if cfg.get('exit_sites'):                         # cascade: the aligner runs the model itself
                    out = al.forward_exit(model, x); tap.h = al.last_h
                else:
                    if cfg['tok']:
                        with torch.no_grad():
                            x = al.tokenizer(x)
                    if eta is not None:
                        out = eta(x)
                    else:
                        with torch.no_grad():
                            out = model(x)
                n = len(out)
                logits[pos:pos + n] = out.half().cpu().numpy(); h[pos:pos + n] = tap.h.half().cpu().numpy()
                index_all[pos:pos + n] = idx_seg[s:s + n]; pos += n
            tap.remove(); al.remove()
            path = store / f'{name}_{sname}_{arm}.npz'
            np.savez(path, index=index_all, logits=logits, h=h, segments=np.array(cell_of), n_seg=n_seg, batch_cells=np.array(batch_cells),
                     gate=np.array(al.gate_log, dtype=np.float64) if al.gate_log else np.zeros((0, 3)))
            rec['runs'].setdefault(sname, {})[arm] = {'path': str(path.resolve()), 'eta_updated_samples': eta.n_updated_samples() if eta else 0}
            if cfg.get('exit_sites'):
                rec['runs'][sname][arm]['routed_fraction'] = float(np.mean(al.exit_log))
            print(f'{sname}/{arm}: {time.time() - t0:.0f}s', flush=True)
            del model, eta
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
