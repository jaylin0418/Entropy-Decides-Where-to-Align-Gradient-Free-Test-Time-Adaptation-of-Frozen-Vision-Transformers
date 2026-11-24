"""Scores a stream run (CPU; labels): per-stream and per-segment accuracy of every arm; ETA / SAR arms also with
NEO-EMA(0.5) applied to the recorded head input; the gate's on-fraction per segment; the stream bars of the design
study. (Research name: scripts/spa/score_streams.py, which wrote results/spa/spa4*.json.)

    python scripts/streams/score_streams.py --runs runs/streams_heldout_vitb.json --imagenet-val $IMAGENET_VAL \\
        --out results/scores/streams_heldout_vitb.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokalign import data, models  # noqa: E402

BATCH = 64
BAR_CONT, BAR_CLEAN, BAR_ETA = 2.0, 0.3, 1.0


def neo_ema_correct(h, W, b, y, alpha=0.5):
    corr, mu = np.empty(len(h), bool), None
    for st in range(0, len(h), BATCH):
        hb = h[st:st + BATCH].astype(np.float64); bm = hb.mean(0)
        mu = bm if mu is None else (1 - alpha) * mu + alpha * bm
        corr[st:st + BATCH] = ((hb - mu) @ W.T + b).argmax(1) == y[st:st + BATCH]
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = json.loads(Path(args.runs).read_text())
    index = data.ImageNetIndex(args.imagenet_val)
    W, b = models.head(runs['model'])
    res = {}
    for sname, arms in runs['runs'].items():
        res[sname] = {}
        for arm, info in arms.items():
            z = np.load(info['path']); y = index.labels(z['index']); n_seg = int(z['n_seg']); segs = list(z['segments'])
            variants = {arm: z['logits'].astype(np.float32).argmax(1) == y}
            if 'eta' in arm or 'sar' in arm:
                variants[arm + '+neo'] = neo_ema_correct(z['h'], W, b, y)
            gate = z['gate']
            bcells = list(z['batch_cells']) if 'batch_cells' in z.files else None
            for name, corr in variants.items():
                r = {'acc': float(corr.mean() * 100), 'seg': {}}
                if sname == 'mixed' and bcells is not None:
                    # per-cell accuracy: positions of each batch are consecutive in the stream
                    starts = np.cumsum([0] + [BATCH] * (len(bcells) - 1))
                    for c in sorted(set(bcells)):
                        sel = np.concatenate([np.arange(st, min(st + BATCH, len(corr))) for st, bc in zip(starts, bcells) if bc == c])
                        r['seg'][f'{c}@mixed'] = float(corr[sel].mean() * 100)
                else:
                    for i, c in enumerate(segs):
                        r['seg'][f'{c}@{i}'] = float(corr[i * n_seg:(i + 1) * n_seg].mean() * 100)
                r['clean_seg_acc'] = float(np.mean([v for k, v in r['seg'].items() if k.startswith('clean@')])) if any(c == 'clean' for c in segs) else None
                if len(gate):
                    nb = n_seg // BATCH + (n_seg % BATCH > 0)
                    r['gate_on_per_seg'] = {f'{c}@{i}': float(gate[i * nb:(i + 1) * nb, 2].mean()) for i, c in enumerate(segs)}
                res[sname][name] = r
    cont = [s for s in ('cont5', 'cont3', 'alt') if s in res]
    avg = lambda a, k='acc': float(np.mean([res[s][a][k] for s in cont])) if cont and all(a in res[s] for s in cont) else float('nan')
    names = sorted({a for s in res for a in res[s]})
    table = {a: {'cont': avg(a), 'cleanret': res.get('cleanret', {}).get(a, {}).get('acc'), 'cleanret_clean': res.get('cleanret', {}).get(a, {}).get('clean_seg_acc'),
                 'clean': res.get('clean', {}).get(a, {}).get('acc')} for a in names}
    out = {'table': table, 'streams': res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    f = lambda v: 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.2f}'
    L = ['# streams (EMA statistics, spectral gate)', '', '| arm | continual | cleanret | cleanret clean segs | clean |', '|---|---|---|---|---|']
    for a in names:
        t = table[a]
        L.append(f'| {a} | {f(t["cont"])} | {f(t["cleanret"])} | {f(t["cleanret_clean"])} | {f(t["clean"])} |')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
