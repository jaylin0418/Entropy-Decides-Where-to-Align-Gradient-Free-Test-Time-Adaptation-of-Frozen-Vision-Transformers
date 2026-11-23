"""TokAlign and the head stages: closed-form, label-free, online alignment for a frozen ViT.

* ``TokAlign``   whitening--recoloring of the 16x16x3 pixel patches toward clean statistics (Eq. 1-2 of the paper),
                 applied before the patch embedding.
* ``LayerAlign`` the same map on the patch tokens at the output of one block (block 6 in the two-site method), as a
                 forward hook; CLS untouched, statistics detached so a host's gradients flow through the map.
* ``HierLayerAlign`` the hierarchical map at a block: the token covariance is split into the between-image part (the
                 covariance of the image means) and the within-image part (the covariance of the deviations from the
                 image mean), and each part gets its own whitening-recoloring toward the source part (Eq. hier).
* ``HeadStage``  NEO re-centering of the head input, optionally followed by a covariance alignment with the source
                 covariance as shrinkage prior.
* ``SpectralGate`` model-free drift test on the batch's patches; when it is off only NEO runs.
* ``Stack``      the stages together, one call per batch (Algorithm 1); ``place=(6,)`` adds the block-6 site.

All statistics are running (cumulative within a stream, or EMA per batch); the current batch is included before it is
corrected, as in NEO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

P = 16                     # patch size
D_PATCH = 3 * P * P        # 768
EPS = 1e-5


def sqrt_psd(S: torch.Tensor, inv: bool = False, dtype=None) -> torch.Tensor:
    """Symmetric (inverse) square root of a PSD matrix; eigenvalues floored at EPS."""
    if dtype is not None:
        S = S.to(dtype)
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(EPS)
    return (V / w.sqrt()) @ V.T if inv else (V * w.sqrt()) @ V.T


class Running:
    """Mean and covariance of d-dim vectors: cumulative, or EMA with rate ``ema`` per batch.

    ``n`` is the sample count (cumulative) or the effective window (EMA: batch size / ema, capped by the total seen)."""

    def __init__(self, d: int, device, ema: float | None = None, dtype=torch.float64):
        self.ema, self.total, self.dtype, self.n = ema, 0, dtype, 0
        self.s1 = torch.zeros(d, device=device, dtype=dtype)
        self.s2 = torch.zeros(d, d, device=device, dtype=dtype)

    def update(self, x: torch.Tensor) -> None:
        x = x.to(self.dtype)
        if self.ema is None:
            self.s1 += x.sum(0); self.s2 += x.T @ x; self.n += len(x)
            return
        a = 1.0 if self.total == 0 else self.ema
        self.s1 = (1 - a) * self.s1 + a * x.mean(0)
        self.s2 = (1 - a) * self.s2 + a * (x.T @ x / len(x))
        self.total += len(x); self.n = int(min(self.total, len(x) / self.ema))

    def mean(self) -> torch.Tensor:
        return self.s1 / self.n if self.ema is None else self.s1

    def cov(self) -> torch.Tensor:
        mu = self.mean()
        return (self.s2 / self.n if self.ema is None else self.s2) - torch.outer(mu, mu)


def unfold(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, 224, 224) -> (B * 196, 768) patch vectors in (c, i, j) order."""
    u = F.unfold(x, P, stride=P).transpose(1, 2)
    return u.reshape(-1, u.shape[-1])


def fold(patches: torch.Tensor, batch: int, hw) -> torch.Tensor:
    return F.fold(patches.reshape(batch, -1, D_PATCH).transpose(1, 2), hw, P, stride=P)


@dataclass
class StageConfig:
    lam: float = 0.1                 # constant shrinkage of the patch covariance toward the source
    n0_tok_images: int = 4           # prior strength of the patch statistics, in images (x196 patches)
    n0_head: int = 256               # prior strength of the head covariance, in samples
    ema_tok: float | None = None     # None = cumulative (episodic); 0.2 on streams
    ema_neo: float | None = None     # None = cumulative NEO; 0.5 = NEO-EMA on streams
    ema_headcov: float | None = None  # None = cumulative; 0.1 on streams
    n0_between_images: int = 64      # prior strength of the between-image covariance (hierarchical map), in images
    refresh_every: int = 1           # recompute the maps every k batches (16 in the fast setting)
    fp32: bool = False               # float32 statistics and eigendecompositions (fast setting)
    gate_spec: float = 0.4
    gate_mean: float = 0.3


class TokAlign:
    def __init__(self, src: dict, cfg: StageConfig, device):
        self.cfg, self.dev = cfg, device
        self.dtype = torch.float32 if cfg.fp32 else torch.float64
        self.mu_s = src['tok_mu'].to(device, torch.float64)
        self.S_s = src['tok_cov'].to(device, torch.float64)
        self.S_s_half = sqrt_psd(self.S_s)
        self.run = Running(D_PATCH, device, ema=cfg.ema_tok, dtype=self.dtype)
        self.map, self.calls = None, 0

    def update(self, patches: torch.Tensor) -> None:
        self.run.update(patches)
        self.calls += 1

    def apply(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if self.map is None or (self.calls - 1) % cfg.refresh_every == 0:
            n, n0 = self.run.n, cfg.n0_tok_images * (patches.shape[0] // x.shape[0])
            S = (n * self.run.cov().double() + n0 * self.S_s) / (n + n0)
            S = (1 - cfg.lam) * S + cfg.lam * self.S_s
            M = sqrt_psd(S, inv=True, dtype=torch.float32 if cfg.fp32 else None).double() @ self.S_s_half
            self.map = (M.to(self.dtype), self.run.mean().clone().to(self.dtype), self.mu_s.to(self.dtype))
        M, m, mu_s = self.map
        y = (patches.to(self.dtype) - m) @ M + mu_s
        return fold(y.float(), x.shape[0], x.shape[-2:])


class LayerAlign:
    """TokAlign's alignment on the patch tokens after block ``k`` (1-based): running mean and covariance of the tokens
    (cumulative, or EMA ``ema_tok``), shrinkage toward the source statistics of that block (``patch_mu`` / ``patch_cov``
    from compute_source_stats.py --blocks, prior ``n0_tok_images`` images, constant ``lam``), whitening-recoloring
    recomputed every ``refresh_every`` batches. Registered as a forward hook; ``on`` follows the gate."""

    def __init__(self, model, k: int, src: dict, cfg: StageConfig, device):
        self.cfg, self.dev, self.on, self.calls, self.map = cfg, device, True, 0, None
        self.dtype = torch.float32 if cfg.fp32 else torch.float64
        self.mu_s = src['patch_mu'][k - 1].to(device, torch.float64)
        self.S_s = src['patch_cov'][k - 1].to(device, torch.float64)
        self.S_s_half = sqrt_psd(self.S_s)
        self.run = Running(self.mu_s.numel(), device, ema=cfg.ema_tok, dtype=self.dtype)
        self.handle = model.blocks[k - 1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        if not self.on:
            return out
        cfg, t = self.cfg, out[:, 1:]
        B, T, d = t.shape
        v = t.reshape(B * T, d)
        self.run.update(v.detach())
        self.calls += 1
        if self.map is None or (self.calls - 1) % cfg.refresh_every == 0:
            n, n0 = self.run.n, cfg.n0_tok_images * T
            S = (n * self.run.cov().double() + n0 * self.S_s) / (n + n0)
            S = (1 - cfg.lam) * S + cfg.lam * self.S_s
            M = sqrt_psd(S, inv=True, dtype=torch.float32 if cfg.fp32 else None).double() @ self.S_s_half
            self.map = (M.to(self.dtype), self.run.mean().clone().to(self.dtype), self.mu_s.to(self.dtype))
        M, m, mu_s = self.map
        y = ((v.to(self.dtype) - m) @ M + mu_s).to(out.dtype).reshape(B, T, d)
        return torch.cat([out[:, :1], y], 1)

    def remove(self):
        self.handle.remove()


class HierLayerAlign(LayerAlign):
    """The hierarchical map at block ``k``: with image means m_i and deviations e_ip = t_ip - m_i, the tokens are mapped
    to mu_s + (m_i - m_t) M_B + e_ip M_W with M_B = S_B^{-1/2} (S_B^s)^{1/2} from a running covariance over image means
    (prior ``n0_between_images`` images of ``patch_cov_between``) and M_W = S_W^{-1/2} (S_W^s)^{1/2} from a running
    covariance over the deviations (prior ``n0_tok_images`` x T tokens of ``patch_cov_within``), both shrunk as the pooled
    map. ``mode`` 'hier' uses both maps, 'within' only M_W (M_B = I), 'between' only M_B (M_W = I)."""

    def __init__(self, model, k: int, src: dict, cfg: StageConfig, device, mode: str = 'hier'):
        super().__init__(model, k, src, cfg, device)
        assert mode in ('hier', 'within', 'between'), mode
        self.mode = mode
        self.S_b = src['patch_cov_between'][k - 1].to(device, torch.float64)
        self.S_w = src['patch_cov_within'][k - 1].to(device, torch.float64)
        self.S_b_half, self.S_w_half = sqrt_psd(self.S_b), sqrt_psd(self.S_w)
        self.run_b = Running(self.mu_s.numel(), device, ema=cfg.ema_tok, dtype=self.dtype)   # over image means
        self.run_w = Running(self.mu_s.numel(), device, ema=cfg.ema_tok, dtype=self.dtype)   # over deviations

    def _shrunk_map(self, run: Running, S_s: torch.Tensor, S_s_half: torch.Tensor, n0: int) -> torch.Tensor:
        cfg = self.cfg
        S = (run.n * run.cov().double() + n0 * S_s) / (run.n + n0)
        S = (1 - cfg.lam) * S + cfg.lam * S_s
        return (sqrt_psd(S, inv=True, dtype=torch.float32 if cfg.fp32 else None).double() @ S_s_half).to(self.dtype)

    def _hook(self, module, inputs, out):
        if not self.on:
            return out
        cfg, t = self.cfg, out[:, 1:]
        B, T, d = t.shape
        m = t.mean(1)
        e = (t - m[:, None, :]).reshape(B * T, d)
        self.run_b.update(m.detach()); self.run_w.update(e.detach())
        self.calls += 1
        if self.map is None or (self.calls - 1) % cfg.refresh_every == 0:
            eye = torch.eye(d, device=t.device, dtype=self.dtype)
            M_b = self._shrunk_map(self.run_b, self.S_b, self.S_b_half, cfg.n0_between_images) if self.mode != 'within' else eye
            M_w = self._shrunk_map(self.run_w, self.S_w, self.S_w_half, cfg.n0_tok_images * T) if self.mode != 'between' else eye
            self.map = (M_b, M_w, self.run_b.mean().clone().to(self.dtype), self.mu_s.to(self.dtype))
        M_b, M_w, m_t, mu_s = self.map
        y = mu_s + (m.to(self.dtype) - m_t) @ M_b
        y = (y[:, None, :] + (e.to(self.dtype) @ M_w).reshape(B, T, d)).to(out.dtype)
        return torch.cat([out[:, :1], y], 1)


class HeadStage:
    """NEO on the head input h (always), then the source-prior covariance alignment (when ``cov`` is on)."""

    def __init__(self, src: dict | None, cfg: StageConfig, d: int, device, with_cov: bool):
        self.cfg, self.with_cov = cfg, with_cov
        self.neo = Running(d, device, ema=cfg.ema_neo)
        if with_cov:
            self.S_s = src['head_cov'].to(device, torch.float64)
            self.S_s_half = sqrt_psd(self.S_s)
            self.run = Running(d, device, ema=cfg.ema_headcov)
            self.map, self.calls = None, 0

    def __call__(self, h: torch.Tensor, cov_on: bool = True) -> torch.Tensor:
        h64 = h.double()
        self.neo.update(h64)
        if self.with_cov:
            self.run.update(h64)
            self.calls += 1
            if cov_on:
                if self.map is None or self.calls % self.cfg.refresh_every == 0 or self.cfg.refresh_every == 1:
                    n, n0 = self.run.n, self.cfg.n0_head
                    S = (n * self.run.cov() + n0 * self.S_s) / (n + n0)
                    self.map = (sqrt_psd(S, inv=True, dtype=torch.float32 if self.cfg.fp32 else None).double() @ self.S_s_half,
                                self.run.mean().clone())
                M, mu = self.map
                return ((h64 - mu) @ M).float()
        return (h64 - self.neo.mean()).float()


class SpectralGate:
    """Model-free shift test: mean |log| ratio of the batch's patch DCT power to the source spectrum, and the
    normalized patch-mean shift. On if either exceeds its threshold."""

    def __init__(self, src: dict, cfg: StageConfig, device):
        self.cfg = cfg
        C = torch.tensor(dct_basis(), dtype=torch.float64, device=device)
        self.C = C
        S_s = src['tok_cov'].to(device, torch.float64)
        self.P_s = torch.einsum('ij,jk,ik->i', C, S_s, C).clamp_min(1e-12)   # diag(C S_s C^T)
        self.mu_s = src['tok_mu'].to(device, torch.float64)
        self.scale = S_s.trace().sqrt()
        self.log = []

    def __call__(self, patches: torch.Tensor) -> bool:
        p = patches.double()
        m = p.mean(0)
        power = (((p - m) @ self.C.T) ** 2).mean(0)
        spec = float((power / self.P_s).log().abs().mean())
        shift = float((m - self.mu_s).norm() / self.scale)
        on = spec > self.cfg.gate_spec or shift > self.cfg.gate_mean
        self.log.append((spec, shift, on))
        return on


def dct_basis(n: int = P) -> np.ndarray:
    """Orthonormal 2-D DCT-II for (c, i, j)-ordered patch vectors (block-diagonal over channels)."""
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    D = np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    D[0] /= np.sqrt(2)
    C2 = np.kron(D, D)
    C = np.zeros((3 * n * n, 3 * n * n))
    for c in range(3):
        C[c * n * n:(c + 1) * n * n, c * n * n:(c + 1) * n * n] = C2
    return C


class Stack:
    """One object per stream. ``tokenizer(x)`` maps a normalized batch before the model; ``head(h)`` maps the recorded
    head input after it. ``arm`` is a set over {'tok', 'headcov', 'gate'}; NEO always runs in ``head``. ``place`` lists
    the blocks (1-based) whose output patch tokens get the same alignment as hooks on ``model`` (the two-site method:
    ``'tok' in arm`` and ``place=(6,)``); they follow the gate. ``place_mode='hier'`` uses the hierarchical map at
    those blocks (``HierLayerAlign``; 'within' / 'between' for its one-part variants), ``None`` the pooled map."""

    def __init__(self, src: dict, arm: set, cfg: StageConfig, d: int, device, model=None, place=(), place_mode=None):
        self.arm = arm
        self.tok = TokAlign(src, cfg, device) if 'tok' in arm else None
        self.gate = SpectralGate(src, cfg, device) if 'gate' in arm else None
        self.headst = HeadStage(src, cfg, d, device, with_cov='headcov' in arm)
        self.layers = [HierLayerAlign(model, k, src, cfg, device, place_mode) if place_mode else LayerAlign(model, k, src, cfg, device)
                       for k in place]
        self.on = True

    def remove(self):
        for la in self.layers:
            la.remove()

    @torch.no_grad()
    def tokenizer(self, x: torch.Tensor) -> torch.Tensor:
        if self.tok is None and self.gate is None:
            return x
        patches = unfold(x)
        if self.gate is not None:
            self.on = self.gate(patches)
            for la in self.layers:
                la.on = self.on
        if self.tok is None:
            return x
        self.tok.update(patches)
        return self.tok.apply(x, patches) if self.on else x

    @torch.no_grad()
    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.headst(h, cov_on=self.on)
