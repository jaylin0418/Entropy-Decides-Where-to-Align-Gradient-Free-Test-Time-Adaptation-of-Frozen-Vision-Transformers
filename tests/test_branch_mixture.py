"""The branch mixture of tokalign.aligner (torch, used by the runs) equals the mixture of scripts/analysis/site_mixture.py
(numpy, used to evaluate the saved logits), and behaves as stated: equal entropies give the plain average, a confident
branch dominates, and identical branches return the branch itself. Model-free (CPU)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokalign.aligner import branch_mixture  # noqa: E402

spec = importlib.util.spec_from_file_location('site_mixture', ROOT / 'scripts' / 'analysis' / 'site_mixture.py')
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)


def test_torch_and_numpy_mixtures_agree():
    rng = np.random.default_rng(0)
    la, lb = rng.normal(size=(64, 1000)) * 3, rng.normal(size=(64, 1000)) * 3
    torch_mix = branch_mixture(torch.tensor(np.concatenate([la, lb]), dtype=torch.float32)).exp().numpy()
    P = {'a': sm.softmax(la.astype(np.float32)), 'b': sm.softmax(lb.astype(np.float32))}
    numpy_mix = sm.mixture(P, ('a', 'b'))
    assert np.allclose(torch_mix, numpy_mix, atol=1e-5)
    assert np.allclose(torch_mix.sum(1), 1.0, atol=1e-5)


def test_mixture_weights_behave():
    la = torch.zeros(4, 10); lb = torch.zeros(4, 10)
    la[:, 0], lb[:, 1] = 2.0, 2.0                                    # equal entropies: the plain average of the two softmaxes
    mix = branch_mixture(torch.cat([la, lb])).exp()
    assert torch.allclose(mix, (torch.softmax(la, 1) + torch.softmax(lb, 1)) / 2, atol=1e-6)
    lb[:, 1] = 12.0                                                  # branch b far more confident: it dominates
    assert branch_mixture(torch.cat([la, lb])).argmax(1).eq(1).all()
    same = branch_mixture(torch.cat([la, la])).exp()                 # identical branches: the branch itself
    assert torch.allclose(same, torch.softmax(la, 1), atol=1e-6)


def test_three_branch_mixture_matches_numpy():
    rng = np.random.default_rng(1)
    L = [rng.normal(size=(32, 50)) * 2 for _ in range(3)]
    torch_mix = branch_mixture(torch.tensor(np.concatenate(L), dtype=torch.float32), n_branches=3).exp().numpy()
    P = {str(i): sm.softmax(l.astype(np.float32)) for i, l in enumerate(L)}
    assert np.allclose(torch_mix, sm.mixture(P, ('0', '1', '2')), atol=1e-5)
