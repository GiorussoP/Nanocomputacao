from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .imaging import load_2d, normalize


def segment_classical(image: np.ndarray, min_size: int = 32) -> np.ndarray:
    """Smoke-test backend; scientific runs should use micro-sam or annotations."""
    from skimage.filters import threshold_otsu
    from scipy import ndimage as ndi

    work = normalize(image)
    threshold = threshold_otsu(work)
    mask = work >= threshold
    labels, count = ndi.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    mask = mask & (sizes[labels] >= min_size)
    holes, count = ndi.label(~mask)
    sizes = np.bincount(holes.ravel(), minlength=count + 1)
    fill = (holes > 0) & (sizes[holes] < min_size)
    mask |= fill
    return mask.astype(np.uint32)


def segment_microsam(
    image: np.ndarray,
    python_executable: Path,
    bridge: Path,
    model_type: str = "vit_b_lm",
    mode: str = "apg",
    min_size: int = 25,
) -> np.ndarray:
    if not python_executable.is_file():
        raise FileNotFoundError(f"Python do micro-sam não encontrado: {python_executable}")
    if not bridge.is_file():
        raise FileNotFoundError(f"Bridge do micro-sam não encontrado: {bridge}")
    with tempfile.TemporaryDirectory(prefix="foundation_microsam_") as temp:
        temp_path = Path(temp)
        input_path = temp_path / "input.npy"
        output_path = temp_path / "output.npz"
        np.save(input_path, image, allow_pickle=False)
        subprocess.run(
            [
                str(python_executable), "-s", str(bridge),
                "--input", str(input_path), "--output", str(output_path),
                "--model-type", model_type, "--mode", mode,
                "--min-size", str(min_size),
            ],
            check=True,
        )
        with np.load(output_path, allow_pickle=False) as result:
            return result["instances"].astype(np.uint32)


def save_segmentation(labels: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, np.asarray(labels, dtype=np.uint32), compression="zlib")


def run_one(
    input_path: Path,
    output_path: Path,
    backend: str,
    microsam_python: Path,
    microsam_bridge: Path,
    model_type: str,
    mode: str,
    min_size: int,
) -> None:
    image = load_2d(input_path)
    if backend == "classical":
        labels = segment_classical(image, min_size=min_size)
    elif backend == "microsam":
        labels = segment_microsam(
            image, microsam_python, microsam_bridge,
            model_type=model_type, mode=mode, min_size=min_size,
        )
    else:
        raise ValueError(f"Backend desconhecido: {backend}")
    save_segmentation(labels, output_path)


def run_array(
    image: np.ndarray,
    output_path: Path,
    backend: str,
    microsam_python: Path,
    microsam_bridge: Path,
    model_type: str,
    mode: str,
    min_size: int,
) -> None:
    if backend == "classical":
        if image.ndim == 3:
            image = image.mean(axis=-1)
        labels = segment_classical(image, min_size=min_size)
    elif backend == "microsam":
        labels = segment_microsam(
            image, microsam_python, microsam_bridge,
            model_type=model_type, mode=mode, min_size=min_size,
        )
    else:
        raise ValueError(f"Backend desconhecido: {backend}")
    save_segmentation(labels, output_path)

