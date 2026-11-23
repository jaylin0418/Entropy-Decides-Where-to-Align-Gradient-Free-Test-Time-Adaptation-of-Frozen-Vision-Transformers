"""Frozen timm ViTs with the official augreg (ImageNet-21k -> ImageNet-1k) weights, plus a hook that records the head
input (the CLS token after the final LayerNorm). Tested with timm 0.4.12, whose default weights for these names are the
augreg checkpoints used in the paper; with timm >= 0.9 the explicit ``.augreg_in21k_ft_in1k`` tag is used."""
from __future__ import annotations

import timm
import torch

# One entry per model: the timm architecture, the weight tag for timm >= 0.9, the input normalization and the resize before the
# 224 center crop of clean and natural images (the crop fraction of the weights' evaluation transform: 248 = 0.9 for the augreg
# ViTs, 256 = 0.875 for DeiT). ImageNet-C files are already 224 x 224 and are used as they are.
CONFIG = {
    'vitb': dict(arch='vit_base_patch16_224', tag='.augreg_in21k_ft_in1k', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), resize=248),
    'vits': dict(arch='vit_small_patch16_224', tag='.augreg_in21k_ft_in1k', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), resize=248),
    'vitl': dict(arch='vit_large_patch16_224', tag='.augreg_in21k_ft_in1k', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), resize=248),
    'deitb': dict(arch='deit_base_patch16_224', tag='.fb_in1k', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), resize=256),
    'deits': dict(arch='deit_small_patch16_224', tag='.fb_in1k', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), resize=256),
}
NAMES = {k: v['arch'] for k, v in CONFIG.items()}
MEAN = STD = (0.5, 0.5, 0.5)                                      # the augreg normalization (vitb, vits, vitl)
_CURRENT = 'vitb'                                                 # the model last built: to_input and clean_resize default to it


def build(name: str = 'vitb') -> torch.nn.Module:
    global _CURRENT
    cfg = CONFIG[name]
    arch = cfg['arch']
    major, minor = (int(v) for v in timm.__version__.split('.')[:2])
    if (major, minor) >= (0, 9):
        arch += cfg['tag']
    model = timm.create_model(arch, pretrained=True)
    model.requires_grad_(False)
    _CURRENT = name
    return model.eval()


def clean_resize(name: str | None = None) -> int:
    """The resize before the 224 center crop for clean and natural images, of the model given or of the model last built."""
    return CONFIG[name or _CURRENT]['resize']


def to_input(batch_uint8, device, name: str | None = None) -> torch.Tensor:
    """uint8 (N, 224, 224, 3) -> normalized float (N, 3, 224, 224), with the normalization of the model given or last built."""
    cfg = CONFIG[name or _CURRENT]
    x = torch.as_tensor(batch_uint8).to(device).permute(0, 3, 1, 2).float().div(255)
    m = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
    s = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
    return (x - m) / s


def restrict_classes(model: torch.nn.Module, labels):
    """Forward hook on ``model.head`` that sets the logits of every class not in ``labels`` to a large negative value, the usual
    protocol for ImageNet-R and ImageNet-A (200 of the 1,000 classes). Returns the hook handle (``handle.remove()`` lifts it).
    Softmax, entropy and the prediction then range over the kept classes only."""
    keep = torch.zeros(model.head.out_features, dtype=torch.bool)
    keep[torch.as_tensor(sorted(set(int(l) for l in labels)))] = True

    def hook(module, inputs, out):
        return out.masked_fill(~keep.to(out.device), -1e4)
    return model.head.register_forward_hook(hook)


class HeadInput:
    """Forward hook on ``model.norm`` that keeps the CLS token of the last forward (read-only)."""

    def __init__(self, model):
        self.h = None
        self.handle = model.norm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        self.h = out[:, 0].detach()

    def remove(self):
        self.handle.remove()


def head(name: str = 'vitb'):
    """The classifier head (W, b) as float64 arrays, for scoring recorded head inputs offline ('vitb' / 'vits' or the
    timm name, as written in a run manifest)."""
    m = build({v: k for k, v in NAMES.items()}.get(name, name))
    return m.head.weight.detach().numpy().astype('float64'), m.head.bias.detach().numpy().astype('float64')
