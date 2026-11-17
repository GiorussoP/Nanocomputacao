import argparse
from pathlib import Path

import numpy as np


def _to_uint8_2d(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A imagem não contém pixels finitos.")

    values = image[finite]
    low, high = np.percentile(values, (1.0, 99.7))
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    work = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    return np.clip((work - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_b_lm")
    parser.add_argument("--mode", choices=("amg", "ais", "apg"), default="apg")
    parser.add_argument("--min-size", type=int, default=25)
    args = parser.parse_args()

    from micro_sam.automatic_segmentation import (
        automatic_instance_segmentation,
        get_predictor_and_segmenter,
    )

    image = np.load(args.input, allow_pickle=False)
    if image.ndim == 2:
        prepared = _to_uint8_2d(image)
    elif image.ndim == 3 and image.shape[-1] == 3:
        prepared = np.stack([_to_uint8_2d(image[..., c]) for c in range(3)], axis=-1)
    else:
        raise ValueError(f"Imagem 2D ou RGB esperada; shape recebido: {image.shape}")

    print(f"Rodando micro-SAM {args.model_type} no modo {args.mode}...")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        segmentation_mode=args.mode,
    )
    instances = automatic_instance_segmentation(
        predictor,
        segmenter,
        prepared,
        ndim=2,
        min_size=args.min_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, instances=np.asarray(instances, dtype=np.uint32))


if __name__ == "__main__":
    main()