from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def cluster_embeddings(source: Path, output: Path, clusters: int, by_modality: bool) -> int:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    with np.load(source, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        modalities = data["modalities"].astype(str)
        ids = data["image_ids"].astype(str) if "image_ids" in data else data["pair_keys"].astype(str)
        groups = data["groups"].astype(str)
        paths = data["paths"].astype(str) if "paths" in data else np.asarray([""] * len(ids))

    labels = np.full(len(features), -1, dtype=np.int32)
    pc = np.zeros((len(features), 2), dtype=np.float32)
    partitions = sorted(set(modalities)) if by_modality else ["ALL"]
    for partition in partitions:
        indices = np.arange(len(features)) if partition == "ALL" else np.flatnonzero(modalities == partition)
        if len(indices) < 2:
            continue
        scaled = StandardScaler().fit_transform(features[indices])
        components = min(2, len(indices), scaled.shape[1])
        projected = PCA(n_components=components, random_state=0).fit_transform(scaled)
        pc[indices, :components] = projected
        k = min(clusters, len(indices))
        labels[indices] = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(scaled)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("id", "path", "group", "modality", "cluster", "pc1", "pc2"),
        )
        writer.writeheader()
        for index in range(len(features)):
            writer.writerow({
                "id": ids[index], "path": paths[index], "group": groups[index],
                "modality": modalities[index], "cluster": int(labels[index]),
                "pc1": float(pc[index, 0]), "pc2": float(pc[index, 1]),
            })
    return len(features)

