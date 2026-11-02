from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def load_2d(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = np.asarray(tifffile.imread(path))
    else:
        image = np.asarray(Image.open(path).convert("F"))
    image = np.squeeze(image)
    while image.ndim > 2:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Imagem 2D esperada em {path}; shape recebido: {image.shape}")
    return image.astype(np.float32)


def normalize(image: np.ndarray, low: float = 1.0, high: float = 99.7) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A imagem não contém pixels finitos.")
    lo, hi = np.percentile(image[finite], (low, high))
    if hi <= lo:
        return np.zeros_like(image)
    work = np.nan_to_num(image, nan=lo, posinf=hi, neginf=lo)
    return np.clip((work - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    gray = np.round(255.0 * normalize(image)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)

