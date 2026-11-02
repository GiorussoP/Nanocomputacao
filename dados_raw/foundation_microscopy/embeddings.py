from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import load_2d, to_rgb_uint8


class DinoV2Embedder:
    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()

    def __call__(self, path: Path) -> np.ndarray:
        torch = self.torch
        rgb = to_rgb_uint8(load_2d(path))
        image = Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std)[None].to(self.device)
        with torch.inference_mode():
            feature = self.model(tensor)
        return feature.detach().float().cpu().numpy().reshape(-1)


def save_embeddings(
    output: Path,
    features: list[np.ndarray],
    image_ids: list[str],
    paths: list[str],
    groups: list[str],
    modalities: list[str],
    pair_keys: list[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features).astype(np.float32),
        image_ids=np.asarray(image_ids),
        paths=np.asarray(paths),
        groups=np.asarray(groups),
        modalities=np.asarray(modalities),
        pair_keys=np.asarray(pair_keys),
    )


def fuse_embeddings(
    source: Path,
    output: Path,
    required_modalities: tuple[str, ...],
    exact_only: bool = False,
) -> int:
    with np.load(source, allow_pickle=False) as data:
        features = data["features"]
        pair_keys = data["pair_keys"].astype(str)
        modalities = data["modalities"].astype(str)
        groups = data["groups"].astype(str)

    rows: list[np.ndarray] = []
    keys: list[str] = []
    row_groups: list[str] = []
    for key in sorted(set(pair_keys) - {""}):
        if exact_only and not key.startswith("exact:"):
            continue
        indices = np.flatnonzero(pair_keys == key)
        selected = []
        for modality in required_modalities:
            candidates = indices[modalities[indices] == modality]
            if not len(candidates):
                break
            selected.append(features[candidates].mean(axis=0))
        if len(selected) != len(required_modalities):
            continue
        rows.append(np.concatenate(selected))
        keys.append(key)
        row_groups.append(groups[indices[0]])
    if not rows:
        raise ValueError("Nenhum grupo contém todas as modalidades solicitadas.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(rows).astype(np.float32),
        pair_keys=np.asarray(keys),
        groups=np.asarray(row_groups),
        modalities=np.asarray(["+".join(required_modalities)] * len(rows)),
    )
    return len(rows)

