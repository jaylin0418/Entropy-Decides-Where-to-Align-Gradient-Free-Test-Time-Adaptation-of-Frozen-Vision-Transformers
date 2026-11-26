"""ImageNet / ImageNet-C access by validation index, with the paper's fixed splits.

Expected layouts (standard):
  ImageNet val, class folders:  <imagenet_val>/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG   (e.g. after valprep.sh)
  ImageNet-C:                   <imagenet_c>/<corruption>/<severity>/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG
Labels are the position of the wnid in sorted order (torchvision ImageFolder order = the models' class order).
Clean images go through resize 248 (bicubic) + center crop 224 (the augreg crop 0.9); ImageNet-C files are used as is.

Splits: a fixed permutation of the 50,000 validation indices (seed 20261019).
  design   perm[2000:4000]    all design choices (2,000 images per cell)
  source   perm[46000:49000]  3,000 clean images for the source statistics (labels never used)
  heldout  perm[36000:38000]  ViT-B/16 pre-registered held-out test
  heldout2 perm[38000:40000]  ViT-S/16 pre-registered held-out test
  heldout3 perm[40000:42000]  pre-registered held-out test of the main path (both backbones); never
                              decoded or passed through a model before docs/heldout_test_plan_two_site.md
  heldout4 perm[42000:44000]  reserved for the pre-registered test of the cascade; never decoded or passed
                              through a model before its plan is committed
Further index sets used by single experiments:
  source_extended  perm[46000:50000]  the source images plus 1,000 more: patch statistics of the filter study
                                      (4,000 clean images)
  design_extended  perm[2000:4000] then perm[46000:50000]  6,000 images per cell: FOA on 6,000-image streams and the
                                      head-input dumps of the source-eigenbasis study
  pea_source       perm[10000:30000]  20,000 clean images for PEA's own source statistics (never evaluated)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SEED, N_VAL = 20261019, 50000
SPLITS = {'design': (2000, 4000), 'source': (46000, 49000), 'heldout': (36000, 38000), 'heldout2': (38000, 40000), 'heldout3': (40000, 42000),
          'heldout4': (42000, 44000),
          'source_extended': (46000, 50000), 'pea_source': (10000, 30000)}
CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression')
SEVERITIES = (3, 5)
CELLS = [f'{c}-{s}' for c in CORRUPTIONS for s in SEVERITIES]
# ImageNet-C-bar (Mintun et al., 2021): ten corruptions disjoint from ImageNet-C, same folder layout; the report uses severities 3 and 5
CBAR_CORRUPTIONS = ('blue_noise_sample', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout', 'cocentric_sine_waves',
                    'inverse_sparkles', 'perlin_noise', 'plasma_noise', 'single_frequency_greyscale', 'sparkles')
CBAR_CELLS = [f'{c}-{s}' for c in CBAR_CORRUPTIONS for s in SEVERITIES]


# Natural-shift datasets in class-folder layout (ImageNet-R: 200 classes, 30,000 images; ImageNet-Sketch: 1,000 classes,
# 50,889 images). Images are shuffled once with the seed above; the first N_CELL * N_DESIGN form the design cells and the next
# N_CELL * N_HELD the held-out cells, each of N_CELL images, named '<name>-d1'.. and '<name>-h1'..; labels are the position of
# the wnid in the 1,000-class order of ImageNet val. Images go through the clean transform (resize, center crop 224).
NATURAL = ('r', 'sketch', 'a')
N_CELL, N_DESIGN, N_HELD = 2000, 3, 3


class NaturalIndex:
    def __init__(self, root: str, imagenet_index: 'ImageNetIndex', name: str):
        if name not in NATURAL:
            raise ValueError(f'unknown natural dataset {name}')
        root = Path(root)
        wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
        files, labels = [], []
        for w in wnids:
            for f in sorted((root / w).iterdir()):
                if f.is_file():
                    files.append(f); labels.append(imagenet_index.label_of_wnid[w])
        perm = np.random.default_rng(SEED).permutation(len(files))
        self.name, self.root, self.wnids = name, root, wnids
        self.files = [files[i] for i in perm]
        self.labels = np.asarray(labels, np.int64)[perm]
        self.class_labels = sorted({imagenet_index.label_of_wnid[w] for w in wnids})
        if len(self.files) < N_CELL * (N_DESIGN + N_HELD):
            raise ValueError(f'{root}: {len(self.files)} images, fewer than the {N_CELL * (N_DESIGN + N_HELD)} the splits need')

    def cells(self, split: str):
        if split == 'design':
            return [f'{self.name}-d{k}' for k in range(1, N_DESIGN + 1)]
        if split == 'heldout':
            return [f'{self.name}-h{k}' for k in range(1, N_HELD + 1)]
        raise ValueError(f'natural datasets have the splits design and heldout, not {split}')

    def cell_slice(self, cell: str) -> slice:
        name, part = cell.rsplit('-', 1)
        if name != self.name or part[0] not in 'dh' or not part[1:].isdigit():
            raise ValueError(f'{cell} is not a cell of {self.name}')
        k = int(part[1:]) - 1
        start = (k if part[0] == 'd' else N_DESIGN + k) * N_CELL
        return slice(start, start + N_CELL)

    def labels_of(self, cell: str) -> np.ndarray:
        return self.labels[self.cell_slice(cell)]


def is_natural_cell(cell: str) -> bool:
    name, _, part = cell.rpartition('-')
    return name in NATURAL and len(part) >= 2 and part[0] in 'dh' and part[1:].isdigit()


def load_natural(nindex: NaturalIndex, cell: str, cache_dir: str | None = None, clean_resize: int = 248, workers: int = 16) -> np.ndarray:
    """uint8 (N_CELL, 224, 224, 3) images of one natural cell, cached as <cache_dir>/<cell>_r<resize>.npy."""
    f = Path(cache_dir) / f'{cell}_r{clean_resize}.npy' if cache_dir else None
    if f is not None and f.exists():
        return np.load(f, mmap_mode='r')
    paths = nindex.files[nindex.cell_slice(cell)]
    with ThreadPoolExecutor(workers) as ex:
        a = np.stack(list(ex.map(lambda p: _load(p, True, clean_resize), paths)))
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True); np.save(f, a)
    return a


def split_indices(split: str) -> np.ndarray:
    if split == 'design_extended':
        return np.concatenate([split_indices('design'), split_indices('source_extended')])
    a, b = SPLITS[split]
    return np.random.default_rng(SEED).permutation(N_VAL)[a:b]


def val_name(i: int) -> str:
    return f'ILSVRC2012_val_{i + 1:08d}.JPEG'


class ImageNetIndex:
    """Maps a validation index to its wnid and label from the class-folder layout of ImageNet val."""

    def __init__(self, imagenet_val: str):
        root = Path(imagenet_val)
        self.wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
        if len(self.wnids) != 1000:
            raise ValueError(f'{root}: expected 1000 class folders, found {len(self.wnids)}')
        self.root, self.wnid_of = root, {}
        for w in self.wnids:
            for f in (root / w).iterdir():
                self.wnid_of[f.name] = w
        self.label_of_wnid = {w: k for k, w in enumerate(self.wnids)}

    def label(self, i: int) -> int:
        return self.label_of_wnid[self.wnid_of[val_name(i)]]

    def labels(self, idx) -> np.ndarray:
        return np.array([self.label(int(i)) for i in idx], np.int64)


def _load(path: Path, clean: bool, clean_resize: int = 248) -> np.ndarray:
    from PIL import Image
    from torchvision import transforms
    with Image.open(path) as im:
        im = im.convert('RGB')
        if clean:
            im = transforms.CenterCrop(224)(transforms.Resize(clean_resize, interpolation=transforms.InterpolationMode.BICUBIC)(im))
        if not clean and im.size != (224, 224) and min(im.size) >= 224:     # ImageNet-C-bar images are stored larger than 224
            im = transforms.CenterCrop(224)(im)
        a = np.asarray(im, dtype=np.uint8)
    if a.shape != (224, 224, 3):
        raise ValueError(f'{path}: shape {a.shape}')
    return a


def load_cell(index: ImageNetIndex, cell: str, idx, imagenet_c: str | None = None, workers: int = 16,
              clean_resize: int = 248) -> np.ndarray:
    """uint8 (N, 224, 224, 3) images of one cell ('clean' or '<corruption>-<severity>') for validation indices idx.
    Clean images are resized to ``clean_resize`` (bicubic) and center-cropped to 224 (248 = crop 0.9 of the augreg ViTs;
    236 = crop 0.95 of FlexiViT)."""
    if cell == 'clean':
        paths = [index.root / index.wnid_of[val_name(int(i))] / val_name(int(i)) for i in idx]
    else:
        corr, sev = cell.rsplit('-', 1)
        paths = [Path(imagenet_c) / corr / sev / index.wnid_of[val_name(int(i))] / val_name(int(i)) for i in idx]
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda p: _load(p, cell == 'clean', clean_resize), paths)))


def load_images(index: ImageNetIndex, cell: str, idx, imagenet_c: str | None = None, cache_dir: str | None = None,
                clean_resize: int = 248) -> np.ndarray:
    """``load_cell``, optionally cached as ``<cache_dir>/<cell>_<first index>_<n>.npy`` (read memory-mapped)."""
    if not cache_dir:
        return load_cell(index, cell, idx, imagenet_c, clean_resize=clean_resize)
    tag = '' if cell != 'clean' or clean_resize == 248 else f'_r{clean_resize}'
    f = Path(cache_dir) / f'{cell}{tag}_{int(idx[0])}_{len(idx)}.npy'
    if f.exists():
        return np.load(f, mmap_mode='r')
    a = load_cell(index, cell, idx, imagenet_c, clean_resize=clean_resize)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, a)
    return a
