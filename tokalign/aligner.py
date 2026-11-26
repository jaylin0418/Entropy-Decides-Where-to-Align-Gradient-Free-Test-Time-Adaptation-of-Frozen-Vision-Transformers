"""The online alignment code that produced every TokAlign number in the papers (hook-based; one object per stream).

``stages.py`` is the same method as a small library API. This module is the implementation the paper's runs used,
kept with its computation unchanged so that the run scripts under ``scripts/`` reproduce the reported numbers exactly.
It also holds the variants used only in ablations and negative results: per-block token alignment, the per-image
DCT-diagonal tokenizer, map refresh every k batches and float32 statistics (cost table).

An arm is a dict: ``tok`` (TokAlign in front of the model), ``place`` (0-based blocks whose output patch tokens get
TokAlign's map: same shrinkage, same prior strength, same running statistics, CLS untouched; the main path is
``tok`` with ``place=(5,)``, block 6), ``mean`` / ``cov`` (blocks whose output tokens are re-centered / whitened toward
the source with the batch prior of the multi-block study), ``headcov`` (head covariance stage), ``head`` ('neo' or 'none'),
``gate`` ({'spec': 0.4, 'mean': 0.3} or absent), the EMA rates ``ema_tok`` / ``ema_head`` / ``ema_headcov`` (absent =
cumulative), ``lam_tok``, ``n0_tok_images``, ``n0_head_batches``, ``refresh_every``, ``fp32_eig``, ``stats32``,
``tok_mode`` ('image' = per-image DCT-diagonal tokenizer), ``patch_only`` and ``scale`` (block-mean diagnostics).

Hierarchical alignment (the within/between-image method): ``tok_mode`` / ``place_mode`` in {'hier', 'within', 'between'}
split every site's patch tokens into the image mean m_i and the within-image deviations x_ip - m_i and align the two
components with their own whitening-recoloring maps (source ``*_cov_between`` / ``*_cov_within`` from
compute_source_stats.py --blocks; priors ``n0_between_images`` images and ``n0_tok_images`` x T tokens; the same lam);
'within' / 'between' keep the other component's covariance as it is (the mean is always re-centered).

Branch mixture (``branch``: True with ``tok`` and ``place=(5,)``): after block 6 the batch is doubled, one copy with the
block-6 map and one without; both copies run through blocks 7 to 12 with their own NEO running mean (1.5 forward passes), and
``branch_mixture`` averages the two softmax outputs per image with entropy weights w_a ∝ exp(-H_a / tau), tau = 1.

Early-exit branches (``exit_sites``: 0-based blocks, with ``tok`` and ``place=(5,)``; ``exit_theta``): the trunk is the pair
(pixel map, block-6 map, NEO). Images whose trunk prediction entropy exceeds ``exit_theta`` also run the extra branches, one per
site in ``exit_sites``: for a site below block 6 the saved pre-map tokens of that block are mapped and blocks after it run
without the block-6 map; for a site above block 6 the unmapped block-6 tokens run on, are mapped at the site and continue.
Each branch has its own NEO mean; the routed images' outputs are mixed by ``branch_mixture`` weights over all branches.
Views control (``exit_views``: 2, no ``exit_sites``): the same routed images run the main path again on a horizontal flip and
on a 192-pixel center crop resized to 224, with the maps of the batch frozen and one NEO mean per view, and the three
predictions are mixed as the branches are: an ensemble of the main path over views instead of over placements.
The runner calls ``forward_exit`` instead of the hooked model (no hooks are installed for this arm).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .stages import dct_basis

BATCH, N_EVAL, P, EPS, LAM_TOK, N0_TOK_IMAGES, N0_BLOCK_BATCHES = 64, 2000, 16, 1e-5, 0.1, 4, 4
N0_BETWEEN_IMAGES = 64     # prior strength of the between-image covariance of the hierarchical alignment, in images
HIER_MODES = ('hier', 'within', 'between')


def set_fp32():
    """FP32 semantics: no TF32 in cuDNN convolutions or matmuls (the default TF32 changes results on recent GPUs)."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


BRANCH_TAU = 1.0


def branch_mixture(logits, tau=BRANCH_TAU, n_branches=2):
    """logits (n_branches * B, C) of the branches stacked along the batch (the same B images in each block of rows) -> log of
    the per-image entropy-weighted average of their softmax outputs, (B, C). argmax = the mixture's prediction."""
    B = logits.shape[0] // n_branches
    logp = torch.log_softmax(logits.double(), 1)
    p = logp.exp().reshape(n_branches, B, -1); H = -(p * logp.reshape(n_branches, B, -1)).sum(2)   # (n, B, C), (n, B)
    w = torch.softmax(-H / tau, 0)                                                                   # (n, B), sum to one per image
    mix = (w[:, :, None] * p).sum(0)
    return mix.clamp_min(1e-30).log().float()


def sqrt_psd(S, inv=False, dtype=None):
    if dtype is not None:
        S = S.to(dtype)
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(EPS)
    return (V / w.sqrt()) @ V.T if inv else (V * w.sqrt()) @ V.T


class Running:
    """Mean and covariance of d-dim vectors, cumulative (current batch included) or EMA with rate ``ema`` per batch;
    ``n`` is the sample count (cumulative) or the effective window (EMA: batch size / ema, capped by the total seen)."""

    def __init__(self, d, dev, ema=None, dtype=torch.float64):
        self.ema, self.total, self.dtype = ema, 0, dtype
        self.s1, self.s2, self.n = torch.zeros(d, device=dev, dtype=dtype), torch.zeros(d, d, device=dev, dtype=dtype), 0

    def update(self, x):                      # x (m, d)
        x = x.to(self.dtype)
        if self.ema is None:
            self.s1 += x.sum(0); self.s2 += x.T @ x; self.n += len(x)
            return
        bm, bs2 = x.mean(0), x.T @ x / len(x)
        a = 1.0 if self.total == 0 else self.ema
        self.s1 = (1 - a) * self.s1 + a * bm; self.s2 = (1 - a) * self.s2 + a * bs2
        self.total += len(x); self.n = int(min(self.total, len(x) / self.ema))

    def mean(self):
        return self.s1 / self.n if self.ema is None else self.s1

    def cov(self):
        mu = self.mean()
        return (self.s2 / self.n if self.ema is None else self.s2) - torch.outer(mu, mu)


class Aligner:
    """Holds the source statistics and the per-stream running statistics; installs hooks for one arm."""

    def __init__(self, model, src, arm, dev):
        self.model, self.src, self.arm, self.dev, self.handles = model, src, arm, dev, []
        d = model.embed_dim
        self.run_blocks = {l: Running(d, dev) for l in set(arm['mean']) | set(arm['cov'])}
        self.run_head = Running(d, dev, ema=arm.get('ema_head'))              # NEO mean (cumulative or NEO-EMA)
        self.run_headcov = Running(d, dev, ema=arm.get('ema_headcov'))        # head covariance stage
        self.stats_dtype = torch.float32 if arm.get('stats32') else torch.float64
        self.run_tok = Running(3 * P * P, dev, ema=arm.get('ema_tok'), dtype=self.stats_dtype)
        self.gate = arm.get('gate')                                            # {'spec': thr, 'mean': thr} or None
        self.on, self.gate_log = True, []
        self.k, self.eig_dtype = int(arm.get('refresh_every', 1)), (torch.float32 if arm.get('fp32_eig') else None)
        self.n_batches, self.tok_map, self.head_map = 0, None, None
        self.C = torch.tensor(dct_basis(), dtype=torch.float64, device=dev)
        self.P_s = ((self.C @ torch.linalg.cholesky(src['tok_cov'] + 1e-8 * torch.eye(len(self.C), device=dev, dtype=torch.float64))) ** 2).sum(1)
        self.sc = src['tok_cov'].trace().sqrt()
        self.src_tok_half = sqrt_psd(src['tok_cov'])
        self.src_blk_half = {l: sqrt_psd(src['blk_cov'][l]) for l in arm['cov']}
        self.src_head_half = sqrt_psd(src['head_cov']) if arm.get('headcov') else None
        self.place = tuple(arm.get('place', ()))
        self.run_place = {l: Running(d, dev, ema=arm.get('ema_tok'), dtype=self.stats_dtype) for l in self.place}
        self.src_place_half = {l: sqrt_psd(src['patch_cov'][l]) for l in self.place}
        self.place_map, self.place_calls = {}, {l: 0 for l in self.place}
        # hierarchical alignment: one Running over image means (n = images) and one over within-image deviations (n = tokens)
        self.tok_mode, self.place_mode = arm.get('tok_mode'), arm.get('place_mode')
        self.hier = {}
        if self.tok_mode in HIER_MODES:
            self.hier['tok'] = self._hier_state(src['tok_cov_between'], src['tok_cov_within'], 3 * P * P, dev, arm)
        if self.place_mode in HIER_MODES:
            for l in self.place:
                self.hier[l] = self._hier_state(src['patch_cov_between'][l], src['patch_cov_within'][l], d, dev, arm)
        for l in set(arm['mean']) | set(arm['cov']):
            self.handles.append(model.blocks[l].register_forward_hook(self._block_hook(l)))
        self.branch = bool(arm.get('branch'))
        self.run_head_b = Running(d, dev, ema=arm.get('ema_head')) if self.branch else None   # NEO mean of the mapped branch
        self.exit_sites, self.exit_theta = tuple(arm.get('exit_sites', ())), float(arm.get('exit_theta', 1.5))
        self.exit_random, self.exit_equal = arm.get('exit_random'), bool(arm.get('exit_equal'))   # controls: random routing at a fixed fraction; equal weights
        self.exit_rng = torch.Generator(device='cpu').manual_seed(int(arm.get('exit_seed', 0)))
        self.exit_views = int(arm.get('exit_views', 0))                # control: views of the main path instead of extra branches
        self.exit = bool(self.exit_sites) or self.exit_views > 0
        if self.exit:                                                  # early-exit branches: manual forward, no hooks
            for l in self.exit_sites:
                self.run_place[l] = Running(d, dev, ema=arm.get('ema_tok'), dtype=self.stats_dtype)
                self.src_place_half[l] = sqrt_psd(src['patch_cov'][l]); self.place_calls[l] = 0
            self.run_head_x = {l: Running(d, dev, ema=arm.get('ema_head')) for l in self.exit_sites}
            self.run_head_v = {v: Running(d, dev, ema=arm.get('ema_head')) for v in range(self.exit_views)}   # one NEO mean per view
            self.exit_log = []                                         # routed fraction per batch
            return
        for l in self.place:
            self.handles.append(model.blocks[l].register_forward_hook(self._branch_hook(l) if self.branch else self._place_hook(l)))
        if arm.get('head', 'neo') != 'none':
            self.handles.append(model.norm.register_forward_hook(self._head_hook))

    def _hier_state(self, S_b, S_w, d, dev, arm):
        return {'between': Running(d, dev, ema=arm.get('ema_tok'), dtype=self.stats_dtype),
                'within': Running(d, dev, ema=arm.get('ema_tok'), dtype=self.stats_dtype),
                'S_b': S_b, 'S_w': S_w, 'S_b_half': sqrt_psd(S_b), 'S_w_half': sqrt_psd(S_w), 'map': None, 'calls': 0}

    def _hier_align(self, u, st, mode, mu_s, calls_ok):
        """u (B, T, d): tokens of B images. Image means and within-image deviations are aligned with their own maps:
        y_ip = mu_s + (m_i - mu_t) M_B + (x_ip - m_i) M_W, with M_B / M_W the whitening-recoloring maps of the between-image
        covariance (prior n0_between_images images) and the within-image covariance (prior n0_tok_images x T tokens), both
        shrunk toward the source as the pooled map. mode 'within' sets M_B = I, 'between' sets M_W = I."""
        B, T, d = u.shape
        m = u.mean(1)                                            # (B, d) image means
        e = (u - m[:, None, :]).reshape(B * T, d)                # within-image deviations
        st['between'].update(m.detach()); st['within'].update(e.detach())
        st['calls'] += 1
        if st['map'] is None or ((st['calls'] - 1) % self.k) == 0:
            lam = self.arm.get('lam_tok', LAM_TOK)
            n0_b, n0_w = self.arm.get('n0_between_images', N0_BETWEEN_IMAGES), self.arm.get('n0_tok_images', N0_TOK_IMAGES) * T
            def shrink(run, S_s, n0):
                S = (run.n * run.cov().double() + n0 * S_s) / (run.n + n0)
                return (1 - lam) * S + lam * S_s
            eye = torch.eye(d, device=u.device, dtype=self.stats_dtype)
            M_b = (sqrt_psd(shrink(st['between'], st['S_b'], n0_b), inv=True, dtype=self.eig_dtype).double() @ st['S_b_half']).to(self.stats_dtype) if mode != 'within' else eye
            M_w = (sqrt_psd(shrink(st['within'], st['S_w'], n0_w), inv=True, dtype=self.eig_dtype).double() @ st['S_w_half']).to(self.stats_dtype) if mode != 'between' else eye
            st['map'] = (M_b, M_w, st['between'].mean().clone().to(self.stats_dtype), mu_s.to(self.stats_dtype))
        M_b, M_w, mu_t, mu_s = st['map']
        y = mu_s + (m.to(self.stats_dtype) - mu_t) @ M_b
        return (y[:, None, :] + (e.to(self.stats_dtype) @ M_w).reshape(B, T, d))

    def _align(self, x, run, src_mu, src_half, src_cov, n0, with_cov):
        """x (m, d) float32 -> aligned; running stats include x (updated first)."""
        run.update(x)
        mu = run.mean()
        y = x.double() - mu
        if with_cov:
            n = run.n
            S = (n * run.cov() + n0 * src_cov) / (n + n0)
            y = y @ sqrt_psd(S, inv=True) @ src_half
        return (y + src_mu).float()

    def _block_hook(self, l):
        def hook(module, inputs, out):
            B, T, d = out.shape
            with_cov = l in self.arm['cov']
            scale = self.arm.get('scale', 1.0)
            if self.arm.get('patch_only'):
                y = self._align(out[:, 1:].reshape(B * (T - 1), d), self.run_blocks[l], self.src['blk_mu'][l], self.src_blk_half.get(l),
                                self.src['blk_cov'][l], N0_BLOCK_BATCHES * BATCH * T, with_cov).reshape(B, T - 1, d)
                new = out.clone(); new[:, 1:] = out[:, 1:] + scale * (y - out[:, 1:])
                return new
            y = self._align(out.reshape(B * T, d), self.run_blocks[l], self.src['blk_mu'][l], self.src_blk_half.get(l),
                            self.src['blk_cov'][l], N0_BLOCK_BATCHES * BATCH * T, with_cov).reshape(B, T, d)
            return out + scale * (y - out)
        return hook

    def _place_hook(self, l):
        """TokAlign's map on the patch tokens after block l (statistics detached; a host's gradients flow through the map).
        Off when the gate is off (statistics then not updated), as the tokenizer stage."""
        def hook(module, inputs, out):
            if not self.on:
                return out
            return self._place_aligned(l, out)
        return hook

    def _branch_hook(self, l):
        """The branch: the unmapped tokens and the mapped tokens of the same batch, stacked along the batch dimension, so that
        blocks l+1 .. 12 run once on both (the head hook keeps one NEO mean per branch). Gate off: the mapped copy is the
        unmapped copy (the mixture then returns the unmapped prediction)."""
        def hook(module, inputs, out):
            if not self.on:
                return torch.cat([out, out], 0)
            return torch.cat([out, self._place_aligned(l, out)], 0)
        return hook

    def _place_aligned(self, l, out):
        """out (B, T+1, d) block output -> the same with the patch tokens sent through the running map of site l."""
        t = out[:, 1:]
        B, T, d = t.shape
        if l in self.hier:                                            # hierarchical (within/between-image) alignment
            y = self._hier_align(t, self.hier[l], self.place_mode, self.src['patch_mu'][l], True).to(out.dtype)
            return torch.cat([out[:, :1], y], 1)
        v = t.reshape(B * T, d)
        run = self.run_place[l]
        run.update(v.detach())
        self.place_calls[l] += 1
        if l not in self.place_map or ((self.place_calls[l] - 1) % self.k) == 0:
            n0 = self.arm.get('n0_tok_images', N0_TOK_IMAGES) * T; lam = self.arm.get('lam_tok', LAM_TOK)
            S = (run.n * run.cov().double() + n0 * self.src['patch_cov'][l]) / (run.n + n0)
            S = (1 - lam) * S + lam * self.src['patch_cov'][l]
            M = sqrt_psd(S, inv=True, dtype=self.eig_dtype).double() @ self.src_place_half[l]
            self.place_map[l] = (M.to(self.stats_dtype), run.mean().clone().to(self.stats_dtype), self.src['patch_mu'][l].to(self.stats_dtype))
        return self._place_apply(l, out)

    def _place_apply(self, l, out):
        """The current map of site l applied to the patch tokens of out (B, T+1, d) without touching the running statistics."""
        t = out[:, 1:]
        B, T, d = t.shape
        M, mu, mu_s = self.place_map[l]
        y = ((t.reshape(B * T, d).to(self.stats_dtype) - mu) @ M + mu_s).to(out.dtype).reshape(B, T, d)
        return torch.cat([out[:, :1], y], 1)

    def _head_hook(self, module, inputs, out):
        cls = out[:, 0]
        if self.branch:                                                # two branches, one NEO mean each
            B = cls.shape[0] // 2
            self.run_head.update(cls[:B]); self.run_head_b.update(cls[B:])
            y = torch.cat([(cls[:B].double() - self.run_head.mean()), (cls[B:].double() - self.run_head_b.mean())], 0).float()
            out = out.clone(); out[:, 0] = y
            return out
        self.run_head.update(cls)
        if self.arm.get('headcov'):
            self.run_headcov.update(cls)
        if self.arm.get('headcov') and self.on:
            run = self.run_headcov
            if self.head_map is None or (self.n_batches % self.k) == 0:
                n0h = self.arm.get('n0_head_batches', N0_BLOCK_BATCHES) * BATCH
                S = (run.n * run.cov() + n0h * self.src['head_cov']) / (run.n + n0h)
                self.head_map = (sqrt_psd(S, inv=True, dtype=self.eig_dtype).double() @ self.src_head_half, run.mean().clone())
            M, mu = self.head_map
            y = ((cls.double() - mu) @ M).float()
        else:                                                         # NEO: subtract the running mean, no source mean
            y = (cls.double() - self.run_head.mean()).float()
        out = out.clone(); out[:, 0] = y
        return out

    def drift(self, flat):
        """Model-free drift of a batch of patch vectors against the source: mean |log DCT-power ratio|, mean shift."""
        m = flat.double().mean(0)
        P = (((flat.double() - m) @ self.C.T) ** 2).mean(0)
        return float((P / self.P_s).log().abs().mean()), float((m - self.src['tok_mu']).norm() / self.sc)

    def tokenizer_image(self, x, n0=196, clip=(0.25, 4.0)):
        """Per-image DCT-diagonal filter (WR1's dct_match, per image): gains sqrt(P_s / P_img) with an equal-weight prior
        toward the source spectrum; mean correction per image. No stream statistics -> usable on per-batch mixed streams."""
        u = F.unfold(x, P, stride=P).transpose(1, 2)                  # (B, T, 768)
        B, T, d = u.shape
        m = u.double().mean(1, keepdim=True)                           # per-image mean patch
        z = (u.double() - m) @ self.C.T                                # DCT coefficients
        P_img = (z ** 2).mean(1)                                       # (B, 768)
        P_hat = (T * P_img + n0 * self.P_s) / (T + n0)
        g = (self.P_s / P_hat.clamp_min(1e-12)).sqrt().clamp(*clip)   # (B, 768)
        y = (z * g[:, None, :]) @ self.C + self.src['tok_mu']
        return F.fold(y.float().transpose(1, 2), x.shape[-2:], P, stride=P)

    def tokenizer(self, x):
        """x (B, 3, 224, 224) normalised -> patch-space CORAL with running target stats and source prior."""
        u = F.unfold(x, P, stride=P).transpose(1, 2)                  # (B, 196, 768)
        B, T, d = u.shape
        flat = u.reshape(B * T, d)
        if self.gate is not None:
            ds, dm = self.drift(flat)
            self.on = ds > self.gate['spec'] or dm > self.gate['mean']
            self.gate_log.append((ds, dm, self.on))
        self.run_tok.update(flat)
        self.n_batches += 1
        if not self.on:
            return x
        if self.arm.get('tok_mode') == 'image':
            return self.tokenizer_image(x)
        if 'tok' in self.hier:
            y = self._hier_align(u, self.hier['tok'], self.tok_mode, self.src['tok_mu'], True)
            return F.fold(y.float().transpose(1, 2), x.shape[-2:], P, stride=P)
        if self.tok_map is None or ((self.n_batches - 1) % self.k) == 0:
            n = self.run_tok.n
            n0 = self.arm.get('n0_tok_images', N0_TOK_IMAGES) * T; lam = self.arm.get('lam_tok', LAM_TOK)
            S = (n * self.run_tok.cov().double() + n0 * self.src['tok_cov']) / (n + n0)
            S = (1 - lam) * S + lam * self.src['tok_cov']          # constant shrinkage toward the source (lam_tok)
            M = sqrt_psd(S, inv=True, dtype=self.eig_dtype).double() @ self.src_tok_half
            self.tok_map = (M.to(self.stats_dtype), self.run_tok.mean().clone().to(self.stats_dtype), self.src['tok_mu'].to(self.stats_dtype))
        return self._tok_apply(x)

    def _tok_apply(self, x):
        """The current pixel map applied to x (B, 3, 224, 224) without touching the running statistics."""
        u = F.unfold(x, P, stride=P).transpose(1, 2)
        B, T, d = u.shape
        M, mu, mu_s = self.tok_map
        y = (u.reshape(B * T, d).to(self.stats_dtype) - mu) @ M + mu_s
        return F.fold(y.float().reshape(B, T, d).transpose(1, 2), x.shape[-2:], P, stride=P)

    def _tokens(self, model, x):
        """Patch embedding, class token and position embedding of timm's ViT (the part of forward_features before the blocks)."""
        t = model.patch_embed(x)
        cls = model.cls_token.expand(t.shape[0], -1, -1)
        return model.pos_drop(torch.cat((cls, t), 1) + model.pos_embed)

    def _head(self, model, t, run):
        """norm, CLS, NEO with the running mean ``run`` (updated first), classifier -> logits."""
        h = model.norm(t)[:, 0]
        if run is self.run_head:
            self._last_trunk_h = h.detach()
        run.update(h)
        return model.head(model.pre_logits((h.double() - run.mean()).float()))

    @torch.no_grad()
    def forward_exit(self, model, x):
        """One batch through the trunk (pixel map, blocks 1 to 6, block-6 map, blocks 7 to 12, NEO) and, for the images whose
        trunk entropy exceeds exit_theta, the extra branches of exit_sites; returns log-probabilities (B, C) whose argmax is
        the prediction (trunk rows are its log-softmax, routed rows the mixture). Gate off: the trunk without maps."""
        x_raw = x
        x = self.tokenizer(x)                                          # pixel map (and the gate, if any)
        t = self._tokens(model, x)
        main = self.place[0]                                           # the trunk's block (0-based); 5 for the pair of the report
        lo, hi = [l for l in self.exit_sites if l < main], [l for l in self.exit_sites if l > main]
        saved = {}
        for i, blk in enumerate(model.blocks):
            t = blk(t)
            if i in lo:
                saved[i] = t
            if i == main:
                saved[main] = t
                if self.on:
                    t = self._place_aligned(main, t)
        logits = self._head(model, t, self.run_head)
        self.last_h = self._last_trunk_h                                # head input of the trunk (stream records)
        out = torch.log_softmax(logits.double(), 1).float()
        if not self.on:
            self.exit_log.append(0.0)
            return out
        p = out.double().exp(); H = -(p * out.double()).sum(1)
        if self.exit_random is not None:                               # control: the same fraction of images, chosen at random
            S = (torch.rand(H.shape[0], generator=self.exit_rng) < float(self.exit_random)).to(H.device)
        else:
            S = H > self.exit_theta                                    # theta <= 0 routes every image (the full mixture)
        self.exit_log.append(float(S.float().mean()))
        if not bool(S.any()):
            return out
        branches = [logits[S]]
        for l in lo:                                                   # site below block 6: map the saved tokens, run on without the block-6 map
            u = self._place_aligned_all_then_subset(l, saved[l], S)
            for blk in model.blocks[l + 1:]:
                u = blk(u)
            branches.append(self._head(model, u, self.run_head_x[l]))
        for l in hi:                                                   # site above block 6: unmapped tokens run on, mapped at the site
            u = saved[main][S]
            for blk in model.blocks[main + 1:l + 1]:
                u = blk(u)
            u = self._place_aligned(l, u)
            for blk in model.blocks[l + 1:]:
                u = blk(u)
            branches.append(self._head(model, u, self.run_head_x[l]))
        for v in range(self.exit_views):                               # views control: the main path again on a view of the routed images, maps frozen
            u = self._tokens(model, self._tok_apply(self.view(x_raw[S], v)))
            for i, blk in enumerate(model.blocks):
                u = blk(u)
                if i == main:
                    u = self._place_apply(main, u)
            branches.append(self._head(model, u, self.run_head_v[v]))
        mix = branch_mixture(torch.cat(branches, 0), n_branches=len(branches), tau=(1e6 if self.exit_equal else BRANCH_TAU))   # tau -> inf: equal weights
        out = out.clone(); out[S] = mix
        return out

    @staticmethod
    def view(x, v):
        """View v of x (B, 3, 224, 224): 0 the horizontal flip, 1 the central 192 x 192 crop resized to 224 (bilinear)."""
        if v == 0:
            return x.flip(3)
        return F.interpolate(x[:, :, 16:208, 16:208], size=x.shape[-2:], mode='bilinear', align_corners=False)

    def _place_aligned_all_then_subset(self, l, t_all, S):
        """Statistics of site l from the whole batch (the saved tokens cost nothing more), the map applied to the subset."""
        y = self._place_aligned(l, t_all)
        return y[S]

    def remove(self):
        for h in self.handles:
            h.remove()


class HostStack(torch.nn.Module):
    """The cascade as the forward of a gradient host (ETA, SAR). The host's forward runs the main path with
    gradients (the pixel map is applied by the runner; the map after the base block here), so the host adapts its LayerNorm
    parameters on the main path's logits exactly as it does with the pair inside it. Alongside, without gradients and with the
    same current weights, the images whose NEO-corrected main-path entropy exceeds ``exit_theta`` run the extra branches of
    ``exit_sites`` and the three predictions are averaged by confidence (``branch_mixture``), as ``forward_exit`` does.
    After each batch: ``h`` the main path's head input (B, d), ``mix`` the log of the averaged probabilities for the routed
    rows (B, C; NaN elsewhere) and ``routed`` (B,) bool; the scorer uses ``mix`` for the routed rows and NEO for the rest."""

    def __init__(self, model, al):
        super().__init__()
        self.model, self.al = model, al
        self.h = self.mix = self.routed = None

    def forward(self, x):
        al, model = self.al, self.model
        main = al.place[0]
        lo, hi = [l for l in al.exit_sites if l < main], [l for l in al.exit_sites if l > main]
        t, saved = al._tokens(model, x), {}
        for i, blk in enumerate(model.blocks):
            t = blk(t)
            if i in lo:
                saved[i] = t.detach()
            if i == main:
                saved[main] = t.detach()
                if al.on:
                    t = al._place_aligned(main, t)
        h = model.norm(t)[:, 0]
        logits = model.head(model.pre_logits(h))
        self.h = h.detach()
        self._second_opinions(logits.detach(), h.detach(), saved, main, lo, hi)
        return logits

    @torch.no_grad()
    def _second_opinions(self, logits, h, saved, main, lo, hi):
        al, model = self.al, self.model
        al.run_head.update(h)                                          # NEO as the scorer applies it: cumulative mean, batch included
        base = model.head(model.pre_logits((h.double() - al.run_head.mean()).float()))
        out = torch.log_softmax(base.double(), 1); p = out.exp(); H = -(p * out).sum(1)
        S = H > al.exit_theta if al.on else torch.zeros_like(H, dtype=torch.bool)
        al.exit_log.append(float(S.float().mean()))
        self.routed, self.mix = S.cpu().numpy(), torch.full_like(logits, float('nan'))
        if not bool(S.any()):
            return
        branches = [base[S]]
        for l in lo:
            u = al._place_aligned_all_then_subset(l, saved[l], S)
            for blk in model.blocks[l + 1:]:
                u = blk(u)
            branches.append(al._head(model, u, al.run_head_x[l]))
        for l in hi:
            u = saved[main][S]
            for blk in model.blocks[main + 1:l + 1]:
                u = blk(u)
            u = al._place_aligned(l, u)
            for blk in model.blocks[l + 1:]:
                u = blk(u)
            branches.append(al._head(model, u, al.run_head_x[l]))
        self.mix[S] = branch_mixture(torch.cat(branches, 0), n_branches=len(branches))


def load_source(path, dev):
    """Source statistics written by scripts/compute_source_stats.py, as float64 tensors on ``dev``."""
    s = torch.load(path, map_location=dev)
    return {k: (v.to(dev).double() if torch.is_tensor(v) else [t.to(dev).double() for t in v]) for k, v in s.items()
            if torch.is_tensor(v) or isinstance(v, list)}
