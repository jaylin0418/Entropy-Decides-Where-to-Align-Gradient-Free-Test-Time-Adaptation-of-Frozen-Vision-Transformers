"""Per-model configuration, class restriction and the natural-dataset index (tokalign/models.py, tokalign/data.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokalign import data, models  # noqa: E402


def test_every_model_has_a_normalization_and_a_resize():
    for name, cfg in models.CONFIG.items():
        assert set(cfg) == {'arch', 'tag', 'mean', 'std', 'resize'} and cfg['resize'] >= 224 and len(cfg['mean']) == 3
    x = np.full((1, 224, 224, 3), 255, np.uint8)
    assert torch.allclose(models.to_input(x, 'cpu', 'vitb'), torch.ones(1, 3, 224, 224))
    d = models.to_input(x, 'cpu', 'deitb')[0, :, 0, 0]
    assert torch.allclose(d, (1 - torch.tensor(models.CONFIG['deitb']['mean'])) / torch.tensor(models.CONFIG['deitb']['std']))
    assert models.clean_resize('deitb') == 256 and models.clean_resize('vits') == 248


def test_restrict_classes_masks_the_other_logits():
    head = torch.nn.Linear(4, 10)
    m = torch.nn.Module(); m.head = head
    handle = models.restrict_classes(m, [1, 3])
    out = m.head(torch.zeros(2, 4))
    assert (out[:, [1, 3]] > -1e3).all() and (out[:, [0, 2, 4, 5, 6, 7, 8, 9]] == -1e4).all()
    handle.remove()
    assert (m.head(torch.zeros(1, 4)) > -1e3).all()


def test_natural_index_splits_and_labels(tmp_path):
    val = tmp_path / 'val'; nat = tmp_path / 'nat'
    wn = [f'n{k:08d}' for k in range(1000)]
    for w in wn:
        (val / w).mkdir(parents=True); (val / w / 'ILSVRC2012_val_00000001.JPEG').write_bytes(b'')
    old = data.N_CELL, data.N_DESIGN, data.N_HELD
    data.N_CELL, data.N_DESIGN, data.N_HELD = 4, 2, 1
    try:
        for w in wn[:3]:
            for j in range(5):
                (nat / w).mkdir(parents=True, exist_ok=True); (nat / w / f'{j}.jpg').write_bytes(b'')
        index = data.ImageNetIndex(str(val)); ni = data.NaturalIndex(str(nat), index, 'r')
        assert len(ni.files) == 15 and ni.class_labels == [0, 1, 2]
        assert ni.cells('design') == ['r-d1', 'r-d2'] and ni.cells('heldout') == ['r-h1']
        assert ni.cell_slice('r-d2') == slice(4, 8) and ni.cell_slice('r-h1') == slice(8, 12)
        assert len(ni.labels_of('r-d1')) == 4 and set(ni.labels_of('r-h1')) <= {0, 1, 2}
        again = data.NaturalIndex(str(nat), index, 'r')
        assert [f.name for f in again.files] == [f.name for f in ni.files]       # the shuffle is fixed by the seed
        with pytest.raises(ValueError):
            ni.cell_slice('sketch-d1')
    finally:
        data.N_CELL, data.N_DESIGN, data.N_HELD = old
    assert data.is_natural_cell('a-h2') and not data.is_natural_cell('clean')
