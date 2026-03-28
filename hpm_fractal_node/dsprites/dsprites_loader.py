"""
dSprites data loader.

Downloads the dSprites dataset on first use and caches it locally.
Downsamples 64x64 to 16x16 via block averaging and returns float vectors.

dSprites generative factors:
    0  shape       3 values  (0=square, 1=ellipse, 2=heart)
    1  scale       6 values  (0.5 to 1.0)
    2  orientation 40 values (0 to 2pi)
    3  pos_x       32 values (0 to 1)
    4  pos_y       32 values (0 to 1)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
_CACHE_DIR = Path(__file__).parents[2] / "data"
_CACHE_FILE = _CACHE_DIR / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

SHAPE_NAMES = {0: "square", 1: "ellipse", 2: "heart"}
D = 256   # 16x16 flattened
GRID = 16


def _download() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dSprites (~150MB) to {_CACHE_FILE} ...")
    urllib.request.urlretrieve(_DSPRITES_URL, _CACHE_FILE)
    print("Download complete.")


def _downsample_block(imgs: np.ndarray) -> np.ndarray:
    """
    Downsample (N, 64, 64) uint8 binary images to (N, 16, 16) float32
    by averaging each non-overlapping 4x4 block.
    """
    N = imgs.shape[0]
    x = imgs.reshape(N, 16, 4, 16, 4).mean(axis=(2, 4))
    return x.astype(np.float32)


def load_dsprites(
    n_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dSprites, downsample to 16x16, return flat float vectors.

    Parameters
    ----------
    n_samples : int or None
        Number of samples to return. None = all 737,280.
    seed : int
        RNG seed for sampling when n_samples is specified.

    Returns
    -------
    images : ndarray, shape (N, 256)
        Float32 pixel vectors in [0, 1]. Each value is mean of a 4x4 block.
    latents : ndarray, shape (N, 5)
        Integer factor indices: [shape, scale, orientation, pos_x, pos_y].
        shape: 0=square, 1=ellipse, 2=heart
    """
    if not _CACHE_FILE.exists():
        _download()

    data = np.load(str(_CACHE_FILE))
    imgs_full = data["imgs"]               # (737280, 64, 64) uint8
    latents_classes = data["latents_classes"]  # (737280, 6) int; col 0 is colour (always 0)

    latents = latents_classes[:, 1:].astype(np.int32)  # (N, 5): skip colour

    if n_samples is not None and n_samples < len(imgs_full):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(imgs_full), size=n_samples, replace=False)
        idx.sort()
        imgs_full = imgs_full[idx]
        latents = latents[idx]

    imgs_16 = _downsample_block(imgs_full)        # (N, 16, 16)
    vectors = imgs_16.reshape(len(imgs_16), -1)   # (N, 256)

    return vectors, latents


def factor_names() -> list[str]:
    return ["shape", "scale", "orientation", "pos_x", "pos_y"]


def shape_name(idx: int) -> str:
    """0=square, 1=ellipse, 2=heart."""
    return SHAPE_NAMES.get(idx, f"shape_{idx}")
