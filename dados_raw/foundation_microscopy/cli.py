from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import cluster_embeddings
from .embeddings import DinoV2Embedder, fuse_embeddings, save_embeddings
from .fusion import build_composite
from .inventory import MODALITIES, read_csv, scan, summarize, write_csv
from .segmentation import run_array, run_one


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _modalities(raw: str) -> set[str]:
    return set(_modality_list(raw))


def _modality_list(raw: str) -> list[str]:
    return list(dict.fromkeys(item.strip().upper() for item in raw.split(",") if item.strip()))


def _filtered(records, group: str | None, modalities: set[str], limit: int | None):
    selected = [
        record for record in records
        if (not group or record.group.casefold() == group.casefold())
        and record.modality in modalities
    ]
    return selected[:limit] if limit else selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foundation-microscopy")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="Indexa imagens e modalidades.")
    inventory.add_argument("--root", type=Path, default=PROJECT_ROOT / "raw/IMAGENS")
    inventory.add_argument("--output", type=Path, default=PROJECT_ROOT / "metadata/foundation_images.csv")

    segment = commands.add_parser("segment", help="Segmenta imagens separadamente por modalidade.")
    segment.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/foundation_images.csv")
    segment.add_argument("--output", type=Path, default=PROJECT_ROOT / "processed/foundation_microscopy/segmentations")
    segment.add_argument("--group")
    segment.add_argument("--modalities", default=",".join(MODALITIES))
    segment.add_argument("--backend", choices=("microsam", "classical"), default="microsam")
    segment.add_argument("--microsam-python", type=Path, default=PROJECT_ROOT / ".microsam-env/bin/python")
    segment.add_argument("--microsam-bridge", type=Path, default=PROJECT_ROOT / "models/microsam_bridge.py")
    segment.add_argument("--model-type", default="vit_b_lm")
    segment.add_argument("--mode", choices=("amg", "ais", "apg"), default="apg")
    segment.add_argument("--min-size", type=int, default=25)
    segment.add_argument("--limit", type=int)

    fused_segment = commands.add_parser(
        "segment-fused", help="Segmenta uma composição de modalidades pareadas.",
    )
    fused_segment.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/foundation_images.csv")
    fused_segment.add_argument("--output", type=Path, default=PROJECT_ROOT / "processed/foundation_microscopy/fused_segmentations")
    fused_segment.add_argument("--group")
    fused_segment.add_argument("--modalities", default="SRS,CARS")
    fused_segment.add_argument("--backend", choices=("microsam", "classical"), default="microsam")
    fused_segment.add_argument("--microsam-python", type=Path, default=PROJECT_ROOT / ".microsam-env/bin/python")
    fused_segment.add_argument("--microsam-bridge", type=Path, default=PROJECT_ROOT / "models/microsam_bridge.py")
    fused_segment.add_argument("--model-type", default="vit_b_lm")
    fused_segment.add_argument("--mode", choices=("amg", "ais", "apg"), default="apg")
    fused_segment.add_argument("--min-size", type=int, default=25)
    fused_segment.add_argument("--allow-candidate", action="store_true")
    fused_segment.add_argument("--register", action="store_true")
    fused_segment.add_argument("--limit", type=int)

    embed = commands.add_parser("embed", help="Extrai embeddings DINOv2.")
    embed.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/foundation_images.csv")
    embed.add_argument("--output", type=Path, default=PROJECT_ROOT / "processed/foundation_microscopy/embeddings.npz")
    embed.add_argument("--group")
    embed.add_argument("--modalities", default=",".join(MODALITIES))
    embed.add_argument("--model", default="dinov2_vits14")
    embed.add_argument("--device")
    embed.add_argument("--limit", type=int)

    fuse = commands.add_parser("fuse", help="Concatena embeddings de aquisições multimodais pareadas.")
    fuse.add_argument("--input", type=Path, required=True)
    fuse.add_argument("--output", type=Path, required=True)
    fuse.add_argument("--modalities", default="SHG,TPEF,CARS,SRS")
    fuse.add_argument("--exact-only", action="store_true")

    cluster = commands.add_parser("cluster", help="Agrupa embeddings com PCA + k-means.")
    cluster.add_argument("--input", type=Path, required=True)
    cluster.add_argument("--output", type=Path, required=True)
    cluster.add_argument("--clusters", type=int, default=5)
    cluster.add_argument("--by-modality", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        records = scan(args.root)
        write_csv(records, args.output)
        print(summarize(records))
        print(f"\n{len(records)} imagens gravadas em {args.output}")
        return
    if args.command == "segment":
        records = _filtered(read_csv(args.manifest), args.group, _modalities(args.modalities), args.limit)
        for index, record in enumerate(records, start=1):
            output = args.output / record.group / record.modality / f"{record.image_id}.tif"
            print(f"[{index}/{len(records)}] {record.modality}: {record.relative_path}")
            run_one(
                Path(record.path), output, args.backend, args.microsam_python,
                args.microsam_bridge, args.model_type, args.mode, args.min_size,
            )
        return
    if args.command == "segment-fused":
        modalities = _modality_list(args.modalities)
        records = _filtered(read_csv(args.manifest), args.group, set(modalities), None)
        pair_keys = sorted({record.pair_key for record in records if record.pair_key})
        completed = 0
        skipped = 0
        for pair_key in pair_keys:
            if pair_key.startswith("candidate:") and not args.allow_candidate:
                skipped += 1
                continue
            members = [record for record in records if record.pair_key == pair_key]
            ordered = []
            for modality in modalities:
                matches = [record for record in members if record.modality == modality]
                if len(matches) != 1:
                    ordered = []
                    break
                ordered.append(matches[0])
            if not ordered:
                skipped += 1
                continue
            composite = build_composite([Path(record.path) for record in ordered], register=args.register)
            safe_key = pair_key.replace(":", "_")
            output = args.output / ordered[0].group / "+".join(modalities) / f"{safe_key}.tif"
            print(f"[{completed + 1}] {'+'.join(modalities)}: {pair_key}")
            run_array(
                composite, output, args.backend, args.microsam_python,
                args.microsam_bridge, args.model_type, args.mode, args.min_size,
            )
            completed += 1
            if args.limit and completed >= args.limit:
                break
        print(f"Concluídos: {completed}; ignorados/não pareáveis: {skipped}")
        return
    if args.command == "embed":
        records = _filtered(read_csv(args.manifest), args.group, _modalities(args.modalities), args.limit)
        embedder = DinoV2Embedder(args.model, args.device)
        features = []
        for index, record in enumerate(records, start=1):
            print(f"[{index}/{len(records)}] {record.modality}: {record.relative_path}")
            features.append(embedder(Path(record.path)))
        save_embeddings(
            args.output, features, [item.image_id for item in records],
            [item.path for item in records], [item.group for item in records],
            [item.modality for item in records], [item.pair_key for item in records],
        )
        return
    if args.command == "fuse":
        count = fuse_embeddings(args.input, args.output, tuple(_modality_list(args.modalities)), args.exact_only)
        print(f"{count} vetores multimodais gravados em {args.output}")
        return
    if args.command == "cluster":
        count = cluster_embeddings(args.input, args.output, args.clusters, args.by_modality)
        print(f"{count} amostras agrupadas em {args.output}")


if __name__ == "__main__":
    main()

