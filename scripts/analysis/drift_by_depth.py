"""Where the shift enters (GPU; no labels): per-batch drift of the patch tokens at every site (the pixel patches and the
output of each of the 12 blocks) against the source statistics, for every cell of a split, with the frozen model and,
in a second pass, with TokAlign in front (the residual drift after the pixel-site alignment). Two statistics per site
and batch, those of the spectral gate: spec = mean |log| ratio of the batch variance along the site's source eigenbasis
(the DCT basis at the pixel site) to the source variance; shift = distance of the batch mean to the source mean over
sqrt(tr Sigma_s). Writes <out-dir>/drift_by_depth_<split>_<model>.npz (arrays drift_<pass>_<cell>_<site>, n_batches x 2)
and the manifest <out-dir>/drift_by_depth_<split>_<model>.json; scripts/analysis/score_drift_by_depth.py turns them
into z-scores against the clean cell. (Research name: DriftLog of scripts/cvpr/run_ext.py.)

    python scripts/analysis/drift_by_depth.py --split design --stats stats/vitb_blocks.pt --imagenet-val $IMAGENET_VAL \\
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
from tokalign.aligner import BATCH, N_EVAL, Aligner, load_source, set_fp32  # noqa: E402
from tokalign.stages import dct_basis, unfold  # noqa: E402

SITES = ('pixel',) + tuple(f'b{k}' for k in range(1, 13))


class DriftLog:
    """Record-only hooks on the 12 blocks (registered after any alignment hook, so an aligned site is measured after its
    alignment) plus the pixel site measured from the unaligned patches passed to ``new_batch``."""

    def __init__(self, model, src: dict, device):
        self.handles, self.rows, self.seen = [], {s: [] for s in SITES}, set()
        self.basis, self.lam, self.mu, self.scale = {}, {}, {}, {}
        self._add('pixel', src['tok_mu'], src['tok_cov'], torch.tensor(dct_basis(), dtype=torch.float64, device=device))
        for k in range(12):
            S = src['patch_cov'][k]
            w, V = torch.linalg.eigh(S)
            self._add(f'b{k + 1}', src['patch_mu'][k], S, V.T)                      # rows = eigenvectors
            self.handles.append(model.blocks[k].register_forward_hook(
                lambda m, i, o, s=f'b{k + 1}': self._record(s, o[:, 1:].reshape(-1, o.shape[-1]))))

    def _add(self, site, mu, S, basis):
        var = torch.einsum('ij,jk,ik->i', basis, S, basis)
        self.basis[site], self.mu[site], self.scale[site] = basis, mu, S.trace().sqrt()
        self.lam[site] = var.clamp_min(1e-8 * var.max())

    @torch.no_grad()
    def measure(self, site, v):
        p = v.detach().float()                                                       # float32: a log-ratio statistic, not a map
        m = p.mean(0)
        power = (((p - m) @ self.basis[site].T.float()) ** 2).mean(0).double().clamp_min(1e-8 * self.lam[site].max())
        return float((power / self.lam[site]).log().abs().mean()), float((m.double() - self.mu[site]).norm() / self.scale[site])

    def _record(self, site, v):
        if site not in self.seen:                                                    # the first forward of the batch counts
            self.seen.add(site)
            self.rows[site].append(self.measure(site, v))

    def new_batch(self, patches):
        self.seen = set()
        self.rows['pixel'].append(self.measure('pixel', patches))

    def arrays(self):
        return {s: np.array(r, np.float64).reshape(-1, 2) for s, r in self.rows.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['vitb', 'vits'], default='vitb')
    ap.add_argument('--split', choices=['design', 'heldout', 'heldout2', 'heldout3'], default='design')
    ap.add_argument('--stats', required=True, help='source statistics with --blocks (patch_mu / patch_cov)')
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
    name = f'drift_by_depth_{args.split}_{args.model}'
    arrays, rec = {}, {'model': args.model, 'batch': BATCH, 'n_eval': N_EVAL, 'split': args.split, 'stats': args.stats,
                       'sites': SITES, 'passes': {'frozen': 'no alignment', 'tok': 'TokAlign in front (residual drift)'}, 'cells': cells}
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        for pas in ('frozen', 'tok'):
            al = Aligner(model, src, dict(tok=True, mean=(), cov=(), head='none'), dev) if pas == 'tok' else None
            log = DriftLog(model, src, dev)
            with torch.no_grad():
                for s in range(0, N_EVAL, BATCH):
                    x = models.to_input(imgs[s:s + BATCH], dev)
                    log.new_batch(unfold(x))
                    model(al.tokenizer(x) if al is not None else x)
            log.remove()
            if al is not None:
                al.remove()
            for site, a in log.arrays().items():
                arrays[f'drift_{pas}_{cell}_{site}'] = a
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
    path = out_dir / f'{name}.npz'
    np.savez(path, **arrays)
    rec.update({'path': str(path.resolve()), 'wall_s': round(time.time() - t0, 1)})
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
