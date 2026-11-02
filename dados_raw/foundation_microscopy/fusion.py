from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.transform import resize

from .imaging import load_2d, normalize


def build_composite(paths: list[Path], register: bool = False) -> np.ndarray:
    """Build an RGB early-fusion image from two to four registered modalities."""
    if not 2 <= len(paths) <= 4:
        raise ValueError("A fusão requer entre duas e quatro modalidades.")
    images = [normalize(load_2d(path)) for path in paths]
    reference_shape = images[0].shape
    images = [
        image if image.shape == reference_shape else resize(
            image, reference_shape, preserve_range=True, anti_aliasing=True,
        ).astype(np.float32)
        for image in images
    ]
    if register:
        from scipy import ndimage as ndi
        from skimage.registration import phase_cross_correlation

        reference = images[0]
        aligned = [reference]
        for image in images[1:]:
            shift, _error, _phase = phase_cross_correlation(reference, image, upsample_factor=10)
            aligned.append(ndi.shift(image, shift=shift, order=1, mode="nearest"))
        images = aligned

    if len(images) == 2:
        channels = (images[0], images[1], 0.5 * (images[0] + images[1]))
    elif len(images) == 3:
        channels = tuple(images)
    else:
        # With four physical modalities and a three-channel foundation model,
        # preserve the first two and average the final two chemical contrasts.
        channels = (images[0], images[1], 0.5 * (images[2] + images[3]))
    return np.round(255.0 * np.stack(channels, axis=-1)).astype(np.uint8)

