"""Run one stream of batches through a host with the stages, recording per-image predictions of every head variant.

A stream is a list of (cell, start, end) batches over per-cell image arrays. The tokenizer stage (with its gate) acts
on the input; the host produces logits; the head variants re-map the recorded head input with the model's own linear
head, so one pass yields the host's raw predictions and those with NEO or with the head covariance stage.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import models
from .stages import HeadStage, Stack, StageConfig

BATCH = 64


def run_stream(model, host, stack: Stack | None, head_variants: dict[str, HeadStage], batches, images: dict, device):
    hook = models.HeadInput(model)
    n = sum(e - s for _, s, e in batches)
    preds = {'raw': np.empty(n, np.int64), **{k: np.empty(n, np.int64) for k in head_variants}}
    gate_log, pos, t0 = [], 0, time.time()
    for cell, s, e in batches:
        x = models.to_input(images[cell][s:e], device)
        on = True
        if stack is not None:
            x = stack.tokenizer(x)
            on = stack.on
        gate_log.append(on)
        logits = host(x)
        h = hook.h
        k = e - s
        preds['raw'][pos:pos + k] = logits.argmax(1).cpu().numpy()
        with torch.no_grad():
            for name, hs in head_variants.items():
                preds[name][pos:pos + k] = model.head(hs(h, cov_on=on)).argmax(1).cpu().numpy()
        pos += k
    hook.remove()
    return preds, np.array(gate_log), time.time() - t0


def batches_of(cells, n_per_cell: int, batch: int = BATCH):
    return [(c, s, min(s + batch, n_per_cell)) for c in cells for s in range(0, n_per_cell, batch)]


def seed_everything():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(0)


__all__ = ['run_stream', 'batches_of', 'seed_everything', 'StageConfig', 'Stack', 'HeadStage', 'BATCH']
