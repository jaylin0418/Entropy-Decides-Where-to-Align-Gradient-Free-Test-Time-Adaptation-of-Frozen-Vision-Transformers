"""Multi-block alignment of the token statistics (GPU; no labels): re-centering (mb_*), whitening (mc_*) or LayerNorm
re-standardisation (ln_affine) of the patch tokens after all / the first six / the last six blocks, with NEO at the head,
one online pass per design cell (2,000 images; cumulative statistics, current batch included). A source pass over the
3,000 clean images of split 'source' gives per-block statistics, token-norm thresholds and the top massive channels;
the `neo` arm also records per-block shift diagnostics. Writes per-image logits under <out-dir>/logits/, the manifest
<out-dir>/block_alignment_design_vitb.json and the diagnostics <out-dir>/block_alignment_design_vitb_diag.json.
(Research name: scripts/regprobe/probe_a.py, Probe A of the research docs/icml_prereg_A.md; the papers' "multi-block
mean alignment" paragraph quotes its arm mb_all_all.)

    python scripts/block_alignment/run_block_alignment.py --imagenet-val $IMAGENET_VAL --imagenet-c $IMAGENET_C --out-dir runs
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
from tokalign.aligner import set_fp32  # noqa: E402

CELLS = data.CELLS
BATCH, N_EVAL, N_SRC, EPS = 64, 2000, 3000, 1e-5
TAU_FACTORS, TAU_MAIN, K_MASK = (3.0, 4.0, 6.0), 1, 8          # TAU_MAIN indexes TAU_FACTORS (= 4x)
N0_TOK, LAM = 4 * 196, 0.1                                      # covariance prior (tokens) and constant shrinkage, as the SPA tokenizer stage
SETS = {'all': tuple(range(12)), 'early': tuple(range(6)), 'late': tuple(range(6, 12))}
KINDS = ('mb_all', 'mb_content', 'mb_maskch', 'mc_content', 'ln_affine')
ARMS = {'neo': None, **{f'{k}_{s}': (k, SETS[s]) for k in KINDS for s in SETS}}
LN_N0 = 4 * 196                                                 # prior strength (tokens) of the per-channel LN-output moments


def sqrt_psd(S, inv=False):
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(EPS)
    return (V / w.sqrt()) @ V.T if inv else (V * w.sqrt()) @ V.T


class Running:
    """Cumulative mean and (if ``cov``) covariance of d-dim vectors (float64); the covariance accumulation is the costly part."""

    def __init__(self, d, dev, cov=True):
        self.s1, self.n, self.with_cov = torch.zeros(d, device=dev, dtype=torch.float64), 0, cov
        self.s2 = torch.zeros(d, d, device=dev, dtype=torch.float64) if cov else None

    def update(self, x):
        if len(x) == 0:
            return
        x = x.double(); self.s1 += x.sum(0); self.n += len(x)
        if self.with_cov:
            self.s2 += x.T @ x

    def mean(self):
        return self.s1 / self.n

    def cov(self):
        mu = self.mean()
        return self.s2 / self.n - torch.outer(mu, mu)


def source_pass(model, dev, n_src, imgs):
    """Two passes over the clean source images: (1) norms and channel maxima -> tau, massive channels; (2) statistics."""
    d = model.embed_dim
    norms = [[] for _ in range(12)]; chmax = torch.zeros(12, d, device=dev, dtype=torch.float64)
    def h1(l):
        def hook(m, i, out):
            p = out[:, 1:].reshape(-1, d)
            norms[l].append(p.norm(dim=1).float().cpu()); chmax[l] = torch.maximum(chmax[l], p.abs().amax(0).double())
        return hook
    handles = [model.blocks[l].register_forward_hook(h1(l)) for l in range(12)]
    with torch.no_grad():
        for s in range(0, n_src, BATCH):
            model(models.to_input(imgs[s:s + BATCH], dev))
    for h in handles:
        h.remove()
    med = torch.tensor([torch.cat(norms[l]).median() for l in range(12)], dtype=torch.float64)
    tau = torch.stack([med * f for f in TAU_FACTORS], 1).to(dev)                    # (12, 3)
    mass = chmax.topk(K_MASK, dim=1).indices                                          # (12, k)
    run_all, run_con = [Running(d, dev) for _ in range(12)], [Running(d, dev) for _ in range(12)]
    ln = torch.zeros(12, 2, 2, d, device=dev, dtype=torch.float64)                  # (block, ln1/ln2, sum/sumsq, channel)
    n_out = torch.zeros(12, len(TAU_FACTORS), device=dev, dtype=torch.float64); n_tok = 0
    def hln(l, j):
        def hook(m, i, out):
            o = out[:, 1:].reshape(-1, d).double(); ln[l, j, 0] += o.sum(0); ln[l, j, 1] += (o ** 2).sum(0)
        return hook
    def h2(l):
        def hook(m, i, out):
            p = out[:, 1:].reshape(-1, d); nr = p.norm(dim=1).double()
            run_all[l].update(p); run_con[l].update(p[nr <= tau[l, TAU_MAIN]])
            n_out[l] += (nr[:, None] > tau[l][None, :]).sum(0).double()
        return hook
    handles = [model.blocks[l].register_forward_hook(h2(l)) for l in range(12)]
    handles += [getattr(model.blocks[l], f'norm{j + 1}').register_forward_hook(hln(l, j)) for l in range(12) for j in range(2)]
    with torch.no_grad():
        for s in range(0, n_src, BATCH):
            model(models.to_input(imgs[s:s + BATCH], dev)); n_tok += len(imgs[s:s + BATCH]) * 196
    for h in handles:
        h.remove()
    ln_mu = ln[:, :, 0] / n_tok; ln_var = (ln[:, :, 1] / n_tok - ln_mu ** 2).clamp_min(1e-8)
    return {'ln_mu': ln_mu, 'ln_var': ln_var,'tau': tau, 'median_norm': med.to(dev), 'mass_ch': mass, 'frac_out_src': n_out / n_tok,
            'mu_all': [r.mean() for r in run_all], 'cov_all': [r.cov() for r in run_all],
            'mu_con': [r.mean() for r in run_con], 'cov_con': [r.cov() for r in run_con],
            'cov_con_half': [sqrt_psd(r.cov()) for r in run_con], 'n_images': n_src}


class Aligner:
    """Hooks on all 12 blocks (statistics; alignment at the arm's blocks) and NEO on the head input."""

    def __init__(self, model, src, arm, dev, diag=False):
        self.model, self.src, self.dev, self.diag = model, src, dev, diag
        self.kind, self.blocks = arm if arm else (None, ())
        d = model.embed_dim
        self.run_all = [Running(d, dev, cov=diag) for _ in range(12)]
        self.run_con = [Running(d, dev, cov=diag or (self.kind == 'mc_content' and l in self.blocks)) for l in range(12)]
        self.run_out = [Running(d, dev, cov=False) for _ in range(12)]
        self.n_out = torch.zeros(12, len(TAU_FACTORS), device=dev, dtype=torch.float64); self.n_tok = torch.zeros(12, device=dev, dtype=torch.float64)
        self.run_head = Running(d, dev)
        self.handles = [model.blocks[l].register_forward_hook(self._block_hook(l)) for l in range(12)]
        if self.kind == 'ln_affine':
            self.ln_s1 = torch.zeros(12, 2, d, device=dev, dtype=torch.float64); self.ln_s2 = torch.zeros_like(self.ln_s1); self.ln_n = torch.zeros(12, 2, device=dev, dtype=torch.float64)
            self.handles += [getattr(model.blocks[l], f'norm{j + 1}').register_forward_hook(self._ln_hook(l, j)) for l in self.blocks for j in range(2)]
        self.handles.append(model.norm.register_forward_hook(self._head_hook))

    def _ln_hook(self, l, j):
        """Closed-form Tent: per-channel mean/variance of the LN output over patch tokens (cumulative, source prior) matched to
        the source; equivalent to re-fitting the LN affine (gamma, beta). CLS untouched."""
        src = self.src
        def hook(module, inputs, out):
            B, T, d = out.shape
            o = out[:, 1:].reshape(-1, d).double()
            self.ln_s1[l, j] += o.sum(0); self.ln_s2[l, j] += (o ** 2).sum(0); self.ln_n[l, j] += len(o)
            n = self.ln_n[l, j]
            mu_t = (self.ln_s1[l, j] + LN_N0 * src['ln_mu'][l, j]) / (n + LN_N0)
            var_t = ((self.ln_s2[l, j] + LN_N0 * (src['ln_var'][l, j] + src['ln_mu'][l, j] ** 2)) / (n + LN_N0) - mu_t ** 2).clamp_min(1e-8)
            y = (o - mu_t) * (src['ln_var'][l, j] / var_t).sqrt() + src['ln_mu'][l, j]
            new = out.clone(); new[:, 1:] = y.float().reshape(B, T - 1, d)
            return new
        return hook

    def _block_hook(self, l):
        src = self.src
        def hook(module, inputs, out):
            B, T, d = out.shape
            p = out[:, 1:].reshape(B * (T - 1), d); nr = p.norm(dim=1).double()
            con = nr <= src['tau'][l, TAU_MAIN]
            self.run_all[l].update(p); self.run_con[l].update(p[con])
            self.n_tok[l] += len(p); self.n_out[l] += (nr[:, None] > src['tau'][l][None, :]).sum(0).double()
            if self.diag:
                self.run_out[l].update(p[~con])
            if l not in self.blocks or self.kind == 'ln_affine':
                return None
            y = p.double().clone()
            if self.kind == 'mb_all':
                y = y - (self.run_all[l].mean() - src['mu_all'][l])
            elif self.kind == 'mb_maskch':
                delta = self.run_all[l].mean() - src['mu_all'][l]; delta[src['mass_ch'][l]] = 0.0
                y = y - delta
            elif self.run_con[l].n > 0:
                r = self.run_con[l]
                if self.kind == 'mb_content':
                    y[con] = y[con] - (r.mean() - src['mu_con'][l])
                else:                                                     # mc_content
                    S = (r.n * r.cov() + N0_TOK * src['cov_con'][l]) / (r.n + N0_TOK)
                    S = (1 - LAM) * S + LAM * src['cov_con'][l]
                    y[con] = (y[con] - r.mean()) @ sqrt_psd(S, inv=True) @ src['cov_con_half'][l] + src['mu_con'][l]
            new = out.clone(); new[:, 1:] = y.float().reshape(B, T - 1, d)
            return new
        return hook

    def _head_hook(self, module, inputs, out):
        cls = out[:, 0]
        self.run_head.update(cls)
        new = out.clone(); new[:, 0] = (cls.double() - self.run_head.mean()).float()
        return new

    def diagnostics(self):
        """A1 per block: mean-shift decomposition into content / outlier tokens, channel concentration, covariance drift."""
        rows = []
        for l in range(12):
            s = self.src; ra, rc, ro = self.run_all[l], self.run_con[l], self.run_out[l]
            d_all = ra.mean() - s['mu_all'][l]; d_con = (rc.mean() - s['mu_con'][l]) if rc.n else torch.zeros_like(d_all)
            f = ro.n / max(ra.n, 1)
            contrib = f * (ro.mean() - s['mu_all'][l]) if ro.n else torch.zeros_like(d_all)
            sq = d_all ** 2; top = sq.topk(5).indices
            rows.append({'block': l, 'frac_out': (self.n_out[l] / self.n_tok[l]).tolist(), 'norm_d_all': float(d_all.norm()),
                         'ratio_con': float(d_con.norm() / d_all.norm().clamp_min(1e-12)), 'share_out': float(contrib.norm() ** 2 / sq.sum().clamp_min(1e-12)),
                         'top2_share': float(sq.topk(2).values.sum() / sq.sum().clamp_min(1e-12)), 'top5_share': float(sq.topk(5).values.sum() / sq.sum().clamp_min(1e-12)),
                         'top5_ch': top.tolist(), 'mass_ch': s['mass_ch'][l].tolist(),
                         'cov_rel_all': float((ra.cov() - s['cov_all'][l]).norm() / s['cov_all'][l].norm()),
                         'cov_rel_con': float((rc.cov() - s['cov_con'][l]).norm() / s['cov_con'][l].norm()) if rc.n > 1 else None})
        return rows

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--arms', nargs='*', default=list(ARMS))
    ap.add_argument('--imagenet-val', required=True)
    ap.add_argument('--imagenet-c', required=True)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out-dir', default='runs')
    args = ap.parse_args()
    set_fp32(); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cpu' if args.smoke else 'cuda')
    cells, n_eval, n_src = ['clean'] + CELLS, N_EVAL, N_SRC
    if args.smoke:
        cells, n_eval, n_src = ['clean', 'gaussian_noise-5'], 128, 128
    out_dir, store = Path(args.out_dir), Path(args.out_dir) / 'logits'
    store.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    index = data.ImageNetIndex(args.imagenet_val)
    model = models.build('vitb').to(dev)
    src = source_pass(model, dev, n_src, data.load_images(index, 'clean', data.split_indices('source')[:n_src], args.imagenet_c, args.cache_dir))
    print(f'source pass {time.time() - t0:.0f}s; median norms {[round(float(m), 1) for m in src["median_norm"]]}; '
          f'outlier frac at 4x {[round(float(f), 4) for f in src["frac_out_src"][:, TAU_MAIN]]}', flush=True)
    idx = data.split_indices('design')[:n_eval]
    name = f'block_alignment_design_vitb{"_smoke" if args.smoke else ""}'
    rec = {'model': 'vitb', 'batch': BATCH, 'n_eval': n_eval, 'n_src': n_src,
           'tau_factors': TAU_FACTORS, 'tau_main': TAU_FACTORS[TAU_MAIN], 'k_mask': K_MASK, 'n0_tok': N0_TOK, 'lam': LAM,
           'arms': {a: ({'kind': ARMS[a][0], 'blocks': ARMS[a][1]} if ARMS[a] else 'neo') for a in args.arms}, 'runs': {}}
    diag = {'src': {'median_norm': src['median_norm'].tolist(), 'frac_out_src': src['frac_out_src'].tolist(), 'mass_ch': src['mass_ch'].tolist()}, 'cells': {}}
    for cell in cells:
        imgs = data.load_images(index, cell, idx, args.imagenet_c, args.cache_dir)
        for arm in args.arms:
            al = Aligner(model, src, ARMS[arm], dev, diag=(arm == 'neo'))
            logits = np.zeros((n_eval, 1000), np.float16)
            with torch.no_grad():
                for s in range(0, n_eval, BATCH):
                    out = model(models.to_input(imgs[s:s + BATCH], dev))
                    if not torch.isfinite(out).all():
                        raise FloatingPointError(f'non-finite logits: {arm} {cell}')
                    logits[s:s + BATCH] = out.half().cpu().numpy()
            if arm == 'neo':
                diag['cells'][cell] = al.diagnostics()
            al.remove()
            path = store / f'{name}_{arm}_{cell}.npz'
            np.savez(path, index=idx, logits=logits)
            rec['runs'].setdefault(arm, {})[cell] = {'path': str(path.resolve())}
        print(f'{cell}: {time.time() - t0:.0f}s', flush=True)
        (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
        (out_dir / f'{name}_diag.json').write_text(json.dumps(diag, indent=1) + '\n')
    rec['wall_s'] = round(time.time() - t0, 1)
    (out_dir / f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'done in {rec["wall_s"]}s', flush=True)


if __name__ == '__main__':
    main()
