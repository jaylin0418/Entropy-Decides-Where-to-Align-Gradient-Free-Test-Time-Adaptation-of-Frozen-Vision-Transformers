"""What shifts at each site (GPU; no labels): the within-image and the between-image parts of the patch-token statistics
at every site (the pixel patches and the output of each of the 12 blocks), per batch, for every cell of a split, with the
frozen model. For a batch of B images with T tokens each, the image means m_i and the deviations x_ip - m_i give
  within   log ratio of the trace of the deviations' covariance to the source within-image covariance, and the spec
           statistic (mean |log| variance ratio) along the source within-image eigenbasis;
  between  the same two for the image means against the source between-image covariance (spec over the top 32 source
           directions, as 64 image means cannot fill 768);
  shift    distance of the batch mean to the source mean over sqrt(tr Sigma_s), as the drift analysis.
Writes <out-dir>/decompose_by_depth_<split>_<model>.npz (arrays dec_<cell>_<site>, n_batches x 5: within_logtrace,
within_spec, between_logtrace, between_spec, shift) and a manifest; scripts/analysis/score_decompose_by_depth.py turns
them into per-cell means and z-scores against the clean cell.

    python scripts/analysis/decompose_by_depth.py --split design --stats stats/vitb_hier.pt --imagenet-val $IMAGENET_VAL \\
        --imagenet-c $IMAGENET_C --out-dir runs
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
from tokalign import data, models  # noqa: E402
from tokalign.aligner import BATCH, N_EVAL, load_source, set_fp32  # noqa: E402
from tokalign.stages import unfold  # noqa: E402

SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13))
TOP_BETWEEN = 32
COLS = ('within_logtrace', 'within_spec', 'between_logtrace', 'between_spec', 'shift')


class DecompLog:
    """Record-only hooks on the 12 blocks plus the pixel site from the patches passed to ``new_batch``."""

    def __init__(self, model, src: dict, device):
        self.handles, self.rows, self.seen, self.ref = [], {s: [] for s in SITES}, set(), {}
        self._add('pixel', src['tok_mu'], src['tok_cov_within'], src['tok_cov_between'])
        for k in range(12):
            self._add(f'b{k + 1}', src['patch_mu'][k], src['patch_cov_within'][k], src['patch_cov_between'][k])
            self.handles.append(model.blocks[k].register_forward_hook(
                lambda m, i, o, s=f'b{k + 1}': self._record(s, o[:, 1:])))

    def _add(self, site, mu, S_w, S_b):
        w_w, V_w = torch.linalg.eigh(S_w)
        w_b, V_b = torch.linalg.eigh(S_b)
        self.ref[site] = {'mu': mu, 'scale': (S_w + S_b).trace().sqrt(), 'tr_w': S_w.trace(), 'tr_b': S_b.trace(),
                          'V_w': V_w.T.float(), 'lam_w': w_w.clamp_min(1e-8 * w_w.max()),
                          'V_b': V_b.T[-TOP_BETWEEN:].float(), 'lam_b': w_b[-TOP_BETWEEN:].clamp_min(1e-8 * w_b.max())}

    @torch.no_grad()
    def measure(self, site, u):
        """u (B, T, d) float -> the five statistics of the batch."""
        r = self.ref[site]
        u = u.detach().float()
        m = u.mean(1)                                               # (B, d) image means
        e = (u - m[:, None, :]).reshape(-1, u.shape[-1])            # within-image deviations
        pw = ((e @ r['V_w'].T) ** 2).mean(0).double().clamp_min(1e-8 * r['lam_w'].max())
        mb = m - m.mean(0)
        pb = ((mb @ r['V_b'].T) ** 2).mean(0).double().clamp_min(1e-8 * r['lam_b'].max())
        return (float((pw.sum() / r['tr_w']).log()), float((pw / r['lam_w']).log().abs().mean()),
                float(((mb.double() ** 2).sum(1).mean() / r['tr_b']).log()), float((pb / r['lam_b']).log().abs().mean()),
                float((m.double().mean(0) - r['mu']).norm() / r['scale']))

    def _record(self, site, u):
        if site not in self.seen:
            self.seen.add(site)
            self.rows[site].append(self.measure(site, u))

    def new_batch(self, patches, batch):
        self.seen = set()
        self.rows['pixel'].append(self.measure('pixel', patches.reshape(batch, -1, patches.shape[-1])))

    def arrays(self):
        return {s: np.array(r, np.float64).reshape(-1, len(COLS)) for s, r in self.rows.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3'], default='design')
    ap.add_argument('--stats', required=True, help='source statistics with the within/between parts (compute_source_stats.py --blocks)')
    ap.add_argument('--cells', nargs='*', default=None, help="default: 'clean' and the 30 cells")
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    set_fp32()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cuda')
    cells = args.cells or ['clean'] + data.CELLS
    t0 = time.time()
    model = models.build(args.model).to(dev)
    src = load_source(args.stats, dev)
    index = data.ImageNetIndex(args.imagenet_val)
    idx = data.split_indices(args.split)[:N_EVAL]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f'decompose_by_depth_{args.split}_{args.model}'
    arrays, rec = {}, {'model': args.model, 'batch': BATCH, 'n_eval': N_EVAL, 'split': args.split, 'stats': args.stats,
                       'sites': SITES, 'columns': COLS, 'top_between': TOP_BETWEEN, 'cells': cells}
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        log = DecompLog(model, src, dev)
        with torch.no_grad():
            for s in range(0, N_EVAL, BATCH):
                x = models.to_input(imgs[s:s + BATCH], dev)
                log.new_batch(unfold(x), x.shape[0])
                model(x)
        log.remove()
        for site, a in log.arrays().items():
            arrays[f'dec_{cell}_{site}'] = a
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
    path = out_dir / f'{name}.npz'
    np.savez(path, **arrays)
    rec.update({'path': str(path.resolve()), 'wall_s': round(time.time() - t0, 1)})
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
