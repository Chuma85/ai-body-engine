from __future__ import annotations

import argparse
import csv
from pathlib import Path


SYNTHETIC_ROOT = Path("data/synthetic")
ACTIVE_DATASET_NAME = "phase_3t"
ARCHIVE_DIR_NAME = "_archived_old_mannequin"
ACTIVE_DATASET_PATH = SYNTHETIC_ROOT / ACTIVE_DATASET_NAME
ARCHIVE_DATASET_PATH = SYNTHETIC_ROOT / ARCHIVE_DIR_NAME


def dataset_counts(dataset_path: str | Path = ACTIVE_DATASET_PATH) -> dict[str, int | str | bool]:
    path = Path(dataset_path)
    labels_path = path / "labels" / "labels.csv"
    manifest_path = path / "manifest.csv"
    image_count = len(list((path / "images").rglob("*.png"))) if (path / "images").exists() else 0
    label_count = count_label_rows(labels_path) if labels_path.exists() else 0
    return {
        "path": str(path),
        "exists": path.exists(),
        "labels_path": str(labels_path),
        "labels_exists": labels_path.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "label_rows": label_count,
        "image_count": image_count,
    }


def count_label_rows(labels_path: Path) -> int:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return sum(1 for _ in csv.DictReader(labels_file))


def top_level_synthetic_entries(root: str | Path = SYNTHETIC_ROOT) -> list[str]:
    path = Path(root)
    if not path.exists():
        return []
    return sorted(child.name for child in path.iterdir())


def format_active_dataset_report(counts: dict[str, int | str | bool]) -> str:
    return "\n".join(
        [
            f"Active dataset path: {ACTIVE_DATASET_PATH}",
            f"Archived dataset folder: {ARCHIVE_DATASET_PATH}",
            f"Active dataset exists: {counts['exists']}",
            f"Labels path: {counts['labels_path']}",
            f"Labels rows: {counts['label_rows']}",
            f"Image count: {counts['image_count']}",
            "Warning: archived datasets are preserved for historical comparison only and are not used for active training.",
            "Training must use an explicit dataset path or default only to data/synthetic/phase_3t.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the active synthetic dataset and archived dataset quarantine.")
    parser.add_argument("--dataset", default=str(ACTIVE_DATASET_PATH), help="Dataset to summarize; defaults to active phase_3t.")
    args = parser.parse_args(argv)

    counts = dataset_counts(args.dataset)
    print(format_active_dataset_report(counts))
    if not counts["exists"]:
        return 1
    if not counts["labels_exists"]:
        return 1
    if int(counts["image_count"]) <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
