"""Stream evaluation: one host and one set of running statistics for the whole stream, EMA statistics, no resets.

Streams (2,000 images per segment unless --n):
  cont5, cont3, alt   the 15 corruptions in sequence at severity 5 / 3 / alternating 5,3,...
  cleanret            clean and five random corruption cells alternating (seed 0)
  clean               the clean cell only
  recur0..recur2      the 30 cells in a random order (seeds 0-2)
  mixed               every batch drawn from a random cell (seed 0)
Arms:
  neo           frozen model, NEO-EMA(0.5)
  stack         frozen model, TokAlign + NEO + head covariance (ungated)
  stack_gated   the same with the spectral gate (the paper's forward-only method)
  eta, sar      the host alone; reported raw and with post-hoc NEO-EMA
  tok+eta, tok+sar   gated TokAlign in front of the host; raw and with NEO-EMA

    python scripts/stream.py --model vitb --split heldout --stream cont5 --arms neo stack_gated eta tok+eta ...
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_images  # noqa: E402
from tokalign import data, models  # noqa: E402
from tokalign.hosts import HOSTS  # noqa: E402
from tokalign.runner import BATCH, batches_of, run_stream, seed_everything  # noqa: E402
from tokalign.stages import HeadStage, Stack, StageConfig  # noqa: E402

ARMS = {'neo': ('frozen', set(), False, ()), 'stack': ('frozen', {'tok'}, True, ()), 'stack_gated': ('frozen', {'tok', 'gate'}, True, ()),
        'eta': ('eta', set(), False, ()), 'tok+eta': ('eta', {'tok', 'gate'}, False, ()),
        'sar': ('sar', set(), False, ()), 'tok+sar': ('sar', {'tok', 'gate'}, False, ()),
        # two-site arms: the block-6 site added (needs patch_mu / patch_cov in the statistics file)
        'twosite_gated': ('frozen', {'tok', 'gate'}, True, (6,)), 'twosite+eta_gated': ('eta', {'tok', 'gate'}, False, (6,))}


def stream_batches(name: str, n: int):
    order = list(data.CORRUPTIONS)
    if name in ('cont5', 'cont3'):
        return batches_of([f'{c}-{name[-1]}' for c in order], n)
    if name == 'alt':
        return batches_of([f'{c}-{s}' for c, s in zip(order, itertools.cycle([5, 3]))], n)
    if name == 'clean':
        return batches_of(['clean'], n)
    if name == 'cleanret':
        five = [data.CELLS[i] for i in np.random.default_rng(0).choice(len(data.CELLS), 5, replace=False)]
        return batches_of([x for c in five for x in ('clean', c)], n)
    if name.startswith('recur'):
        return batches_of([data.CELLS[i] for i in np.random.default_rng(int(name[-1])).permutation(len(data.CELLS))], n)
    if name == 'mixed':
        allb = batches_of(data.CELLS, n)
        return [allb[i] for i in np.random.default_rng(0).permutation(len(allb))]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3'], default='design')
    ap.add_argument('--stream', required=True)
    ap.add_argument('--arms', nargs='*', default=['neo', 'stack_gated', 'eta', 'tok+eta'])
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--stats', required=True)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    seed_everything()
    dev = torch.device(args.device)
    src = torch.load(args.stats, map_location=dev)
    cfg = StageConfig(ema_tok=0.2, ema_neo=0.5, ema_headcov=0.1, refresh_every=16 if args.fast else 1, fp32=args.fast)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices(args.split)[:args.n]
    y_all = index.labels(idx)
    batches = stream_batches(args.stream, len(idx))
    images = {c: load_images(index, c, idx, args.imagenet_c, args.cache_dir) for c in sorted({b[0] for b in batches})}
    y = np.concatenate([y_all[s:e] for _, s, e in batches])
    res = {}
    for arm in args.arms:
        host_name, stages, cov, place = ARMS[arm]
        model = models.build(args.model).to(dev)
        host = HOSTS[host_name](model, BATCH)
        stack = Stack(src, stages, cfg, model.embed_dim, dev, model=model, place=place) if stages else None
        heads = {'neo': HeadStage(src, cfg, model.embed_dim, dev, with_cov=cov)}
        preds, gate, sec = run_stream(model, host, stack, heads, batches, images, dev)
        if stack is not None:
            stack.remove()
        out = {'acc': float((preds['neo'] == y).mean() * 100), 'gate_on': float(gate.mean()), 'seconds': sec}
        if host_name != 'frozen':
            out['acc_raw'] = float((preds['raw'] == y).mean() * 100)
        res[arm] = out
        print(arm, {k: round(v, 2) for k, v in out.items()}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'args': vars(args), 'results': res}, indent=1))


if __name__ == '__main__':
    main()
