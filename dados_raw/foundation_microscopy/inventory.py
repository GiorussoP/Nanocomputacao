from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MODALITIES = ("SHG", "TPEF", "CARS", "SRS")
DATE_PATTERN = re.compile(r"\b\d{2}-\d{2}(?:-\d{4})?\b")
ACQUISITION_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*[_ ]?acq", re.IGNORECASE)
LEADING_SAMPLE_PATTERN = re.compile(r"^(\d{1,2})(?:[._-]|\s)")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    relative_path: str
    group: str
    subgroup: str
    modality: str
    channel: str
    acquisition: str
    sample_id: str
    pair_key: str
    pair_quality: str


def _channel(text: str) -> str:
    upper = text.upper()
    if "PMT1" in upper or "[PMT1]" in upper or "_C0" in upper:
        return "PMT1"
    if "PMT2" in upper or "[PMT2]" in upper or "_C1" in upper:
        return "PMT2"
    return ""


def detect_modality(path: Path, group: str) -> tuple[str, str]:
    text = str(path).upper()
    channel = _channel(text)
    if "SRS-CARS" in text:
        if channel == "PMT1":
            return "CARS", channel
        if channel == "PMT2":
            return "SRS", channel
        return "UNKNOWN", channel
    if "SHG" in text or group.upper() == "ESCALA":
        return "SHG", channel
    if "TPEF" in text or "TPFL" in text or "TPEH" in text:
        return "TPEF", channel
    if "CARS" in text:
        return "CARS", channel
    if "SRS" in text:
        return "SRS", channel
    return "UNKNOWN", channel


def _subgroup(relative: Path) -> str:
    parts = relative.parts
    if len(parts) < 2:
        return ""
    for part in parts[1:-1]:
        if DATE_PATTERN.fullmatch(part):
            return part
    return parts[1] if len(parts) > 2 else ""


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).zfill(2) if match else ""


def _pairing(relative: Path, modality: str, channel: str) -> tuple[str, str]:
    context = " ".join(relative.parts)
    acquisition = _first_match(ACQUISITION_PATTERN, context)
    sample_id = _first_match(LEADING_SAMPLE_PATTERN, relative.parent.name)
    group = relative.parts[0]
    subgroup = _subgroup(relative)

    # C0/C1 in the same SRS-CARS directory are simultaneous channels.
    if "SRS-CARS" in context.upper() and channel:
        key_source = str(relative.parent)
        return f"exact:{hashlib.sha1(key_source.encode()).hexdigest()[:16]}", "exact"

    # Other modalities are only candidate pairs: acquisition labels do not
    # prove that fields of view are co-registered.
    if acquisition or sample_id:
        key_source = "|".join((group, subgroup, sample_id, acquisition))
        return f"candidate:{hashlib.sha1(key_source.encode()).hexdigest()[:16]}", "candidate"
    return "", "unpaired"


def scan(root: Path) -> list[ImageRecord]:
    root = root.resolve()
    records: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        rel_to_root = path.relative_to(root)                # BOCA/1.tif  (p/ lógica interna)
        try:
            rel_to_project = path.relative_to(PROJECT_ROOT)
        except ValueError:
            # root fora do projeto: mantém relativo ao root (comportamento antigo)
            rel_to_project = rel_to_root

        group = rel_to_root.parts[0] if len(rel_to_root.parts) > 1 else root.name
        # ↑ continua sendo "BOCA" — não muda!

        modality, channel = detect_modality(rel_to_root, group)
        context = " ".join(rel_to_root.parts)
        acquisition = _first_match(ACQUISITION_PATTERN, context)
        sample_id = _first_match(LEADING_SAMPLE_PATTERN, rel_to_root.parent.name)
        pair_key, pair_quality = _pairing(rel_to_root, modality, channel)

        relative_text = rel_to_project.as_posix()           # ← o que vai pro .npz

        records.append(
            ImageRecord(
                image_id=hashlib.sha1(rel_to_root.as_posix().encode()).hexdigest()[:16],
                path=str(path),
                relative_path=relative_text,                # raw/IMAGENS/BOCA/1.tif
                group=group,                                # continua BOCA
                subgroup=_subgroup(rel_to_root),            # idem
                modality=modality,
                channel=channel,
                acquisition=acquisition,
                sample_id=sample_id,
                pair_key=pair_key,
                pair_quality=pair_quality,
            )
        )
    return records


def write_csv(records: list[ImageRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ImageRecord.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_csv(path: Path) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ImageRecord(**row) for row in csv.DictReader(handle)]


def summarize(records: list[ImageRecord]) -> str:
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_group[record.group][record.modality] += 1
    lines = ["group\tSHG\tTPEF\tCARS\tSRS\tUNKNOWN\ttotal"]
    for group in sorted(by_group):
        counts = by_group[group]
        values = [counts[name] for name in (*MODALITIES, "UNKNOWN")]
        lines.append("\t".join([group, *map(str, values), str(sum(values))]))
    return "\n".join(lines)

