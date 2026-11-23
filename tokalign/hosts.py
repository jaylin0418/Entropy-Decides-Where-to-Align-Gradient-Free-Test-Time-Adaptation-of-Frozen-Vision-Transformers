"""Test-time adaptation hosts the stages are stacked on. Each host takes a normalized batch and returns logits computed
by the model's forward on that batch (the head input is recorded by ``models.HeadInput`` during the same forward).

* ``Frozen``  the source model (NEO and the head stages are then applied to the recorded head input)
* ``ETA``     EATA without the Fisher term (official SAR-repository ``eata.py``, vendored unmodified, BSD-3)
* ``SAR``     official ``sar.py`` + ``sam.py`` (vendored unmodified, BSD-3)
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

THIRD = Path(__file__).resolve().parents[1] / 'third_party' / 'sar'


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Frozen:
    def __init__(self, model, batch: int):
        self.model = model

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x)


class ETA:
    """lr 0.001 at batch 64, SGD momentum 0.9, LayerNorm affine parameters, e_margin ln(C)/2 - 1 for C classes, d_margin 0.05."""

    def __init__(self, model, batch: int, lr_per_64: float = 0.001, n_classes: int = 1000):
        eata = _module('tokalign_sar_eata', THIRD / 'eata.py')
        self.model = eata.configure_model(model)
        params, _ = eata.collect_params(self.model)
        opt = torch.optim.SGD(params, lr_per_64 / 64 * batch, momentum=0.9)
        self.adapted = eata.EATA(self.model, opt, fishers=None, e_margin=math.log(n_classes) / 2 - 1, d_margin=0.05)

    def __call__(self, x):
        return self.adapted(x).detach()

    def n_updated_samples(self) -> int:
        return int(self.adapted.num_samples_update_2)


class SAR:
    """SAM (rho 0.05) over SGD lr 0.001 at batch 64, reliable margin 0.4 ln(C) for C classes, model recovery at EMA < 0.2."""

    def __init__(self, model, batch: int, lr_per_64: float = 0.001, n_classes: int = 1000):
        sam = _module('tokalign_sar_sam', THIRD / 'sam.py')
        sar = _module('tokalign_sar_sar', THIRD / 'sar.py')
        self.model = sar.configure_model(model)
        params, _ = sar.collect_params(self.model)
        opt = sam.SAM(params, torch.optim.SGD, lr=lr_per_64 / 64 * batch, momentum=0.9)
        self.adapted = sar.SAR(self.model, opt, margin_e0=0.4 * math.log(n_classes))
        self.resets = 0

    def __call__(self, x):
        before = self.adapted.ema
        out = self.adapted(x).detach()
        if before is not None and self.adapted.ema is None:      # SAR's model-recovery reset fired on this batch
            self.resets += 1
        return out

    def n_updated_samples(self) -> int:
        return -1


HOSTS = {'frozen': Frozen, 'eta': ETA, 'sar': SAR}
