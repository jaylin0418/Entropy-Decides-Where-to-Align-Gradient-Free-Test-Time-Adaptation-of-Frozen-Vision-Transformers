"""The index splits are disjoint where the documentation says they are, and heldout4 is a fresh 2,000-image slice."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign import data  # noqa: E402


def test_evaluation_splits_are_disjoint_and_sized():
    names = ['design', 'heldout', 'heldout2', 'heldout3', 'heldout4', 'source', 'source_extended', 'pea_source']
    idx = {n: data.split_indices(n) for n in names}
    for n in ('design', 'heldout', 'heldout2', 'heldout3', 'heldout4'):
        assert len(idx[n]) == 2000 and len(set(idx[n])) == 2000
        for m in names:
            if m != n and not (n == 'source' and m == 'source_extended'):
                assert not set(idx[n]) & set(idx[m]), (n, m)
    assert set(idx['source']) <= set(idx['source_extended'])


def test_heldout4_is_the_next_block_of_the_permutation():
    perm = np.random.default_rng(data.SEED).permutation(data.N_VAL)
    assert (data.split_indices('heldout4') == perm[42000:44000]).all()
