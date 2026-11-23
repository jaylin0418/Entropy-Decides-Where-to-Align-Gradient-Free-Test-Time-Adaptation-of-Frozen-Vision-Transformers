"""Episodic evaluation (each ImageNet-C cell is its own stream; statistics and host reset per cell).

One run gives three numbers per cell: the host's raw predictions, with NEO, and with NEO + head covariance stage.
With --tok the tokenizer stage (TokAlign) runs in front of the host; --place 6 adds the same alignment on the patch
tokens after block 6 (the two-site method, with stats from compute_source_stats.py --blocks); with --gate the spectral
gate decides per batch whether the covariance stages run.

    # Table 1, ViT-B, forward-only rows (NEO, +hc, +tok, stack) on the held-out split
    python scripts/evaluate.py --model vitb --split heldout --host frozen         --stats stats/vitb.pt ... --out results/vitb_frozen.json
    python scripts/evaluate.py --model vitb --split heldout --host frozen --tok   --stats stats/vitb.pt ... --out results/vitb_frozen_tok.json
    # ETA rows
    python scripts/evaluate.py --model vitb --split heldout --host eta            ...
    python scripts/evaluate.py --model vitb --split heldout --host eta --tok      ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign import data, models  # noqa: E402
from tokalign.hosts import HOSTS  # noqa: E402
from tokalign.runner import BATCH, batches_of, run_stream, seed_everything  # noqa: E402
from tokalign.stages import HeadStage, Stack, StageConfig  # noqa: E402


def load_images(index, cell, idx, imagenet_c, cache_dir):
    return data.load_images(index, cell, idx, imagenet_c, cache_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3'], default='design')
    ap.add_argument('--host', choices=list(HOSTS), default='frozen')
    ap.add_argument('--tok', action='store_true', help='run the tokenizer stage in front of the host')
    ap.add_argument('--place', type=int, nargs='*', default=[], help='blocks (1-based) that also get the alignment, e.g. 6 (two-site)')
    ap.add_argument('--gate', action='store_true', help='spectral gate for the covariance stages')
    ap.add_argument('--fast', action='store_true', help='maps refreshed every 16 batches, float32 statistics')
    ap.add_argument('--stats', required=True, help='source statistics from compute_source_stats.py')
    ap.add_argument('--cells', nargs='*', default=None, help="default: 'clean' and the 30 corruption cells")
    ap.add_argument('--n', type=int, default=2000, help='images per cell')
    ap.add_argument('--cache-dir', default=None, help='optional: cache decoded cells as .npy')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    seed_everything()
    dev = torch.device(args.device)
    src = torch.load(args.stats, map_location=dev)
    cfg = StageConfig(refresh_every=16 if args.fast else 1, fp32=args.fast)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices(args.split)[:args.n]
    y = index.labels(idx)
    cells = args.cells or ['clean'] + data.CELLS
    arm = ({'tok'} if args.tok else set()) | ({'gate'} if args.gate else set())
    res = {}
    for cell in cells:
        imgs = load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        model = models.build(args.model).to(dev)
        host = HOSTS[args.host](model, BATCH)
        stack = Stack(src, arm, cfg, model.embed_dim, dev, model=model, place=args.place) if (arm or args.place) else None
        heads = {'neo': HeadStage(src, cfg, model.embed_dim, dev, with_cov=False),
                 'headcov': HeadStage(src, cfg, model.embed_dim, dev, with_cov=True)}
        preds, gate, sec = run_stream(model, host, stack, heads, batches_of([cell], len(idx)), {cell: imgs}, dev)
        if stack is not None:
            stack.remove()
        res[cell] = {k: float((v == y).mean() * 100) for k, v in preds.items()}
        res[cell]['gate_on'] = float(gate.mean())
        print(cell, {k: round(v, 2) for k, v in res[cell].items()}, f'{sec:.0f}s', flush=True)
    corrupted = [c for c in cells if c != 'clean']
    summary = {k: float(np.mean([res[c][k] for c in corrupted])) for k in ('raw', 'neo', 'headcov')} if corrupted else {}
    out = {'args': vars(args), 'mean_over_corruption_cells': summary, 'clean': res.get('clean'), 'per_cell': res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('mean over corruption cells:', {k: round(v, 2) for k, v in summary.items()})


if __name__ == '__main__':
    main()
