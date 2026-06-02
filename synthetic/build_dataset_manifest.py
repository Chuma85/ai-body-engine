from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Any

from synthetic.validate_synthetic_dataset import _read_label_rows, validate_dataset

MANIFEST_COLUMNS = [
    "sample_id",
    "front_image_path",
    "side_image_path",
    "back_image_path",
    "has_front",
    "has_side",
    "has_back",
    "capture_views",
    "minimum_scan_views",
    "enhanced_scan_views",
    "renderer_mode",
    "render_source",
    "is_smoke_dataset",
    "is_training_candidate",
    "quality_tier",
    "label_row_index",
    "dataset_split",
]
DEFAULT_SPLIT_SEED = 42
MINIMUM_SCAN_VIEWS = "front,side"
ENHANCED_SCAN_VIEWS = "front,side,back"


def build_dataset_manifest(
    dataset: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    require_back: bool = False,
    require_realistic: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset)
    validation = validate_dataset(dataset_root, require_back=require_back, require_realistic=require_realistic)
    if not validation["valid"]:
        return {
            "valid": False,
            "manifest_path": str(dataset_root / "manifest.csv"),
            "row_count": 0,
            "split_counts": {},
            "errors": [*validation["errors"]],
        }

    labels_path = dataset_root / "labels" / "labels.csv"
    parse_result: dict[str, Any] = {"errors": [], "warnings": []}
    label_rows = _read_label_rows(labels_path, parse_result)
    if parse_result["errors"]:
        return {
            "valid": False,
            "manifest_path": str(dataset_root / "manifest.csv"),
            "row_count": 0,
            "split_counts": {},
            "errors": parse_result["errors"],
        }

    manifest_rows = _manifest_rows(dataset_root, label_rows, split_seed, require_back=require_back)
    manifest_path = dataset_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts: dict[str, int] = {}
    for row in manifest_rows:
        split = row["dataset_split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    return {
        "valid": True,
        "manifest_path": str(manifest_path),
        "row_count": len(manifest_rows),
        "split_counts": split_counts,
        "errors": [],
    }


def _manifest_rows(dataset_root: Path, label_rows: list[dict[str, str]], split_seed: int, require_back: bool = False) -> list[dict[str, str]]:
    rows_by_sample = {row["sample_id"]: (index, row) for index, row in enumerate(label_rows)}
    sample_ids = sorted(rows_by_sample)
    shuffled_sample_ids = [*sample_ids]
    random.Random(split_seed).shuffle(shuffled_sample_ids)
    splits = _assign_splits(shuffled_sample_ids)

    manifest_rows = []
    for sample_id in sample_ids:
        label_row_index, label_row = rows_by_sample[sample_id]
        back_path = dataset_root / "images" / "back" / f"{sample_id}_back.png"
        has_back = require_back or _truthy(label_row.get("has_back")) or back_path.exists()
        capture_views = ENHANCED_SCAN_VIEWS if has_back else MINIMUM_SCAN_VIEWS
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "front_image_path": (dataset_root / "images" / "front" / f"{sample_id}_front.png").as_posix(),
                "side_image_path": (dataset_root / "images" / "side" / f"{sample_id}_side.png").as_posix(),
                "back_image_path": back_path.as_posix() if has_back else "",
                "has_front": "true",
                "has_side": "true",
                "has_back": "true" if has_back else "false",
                "capture_views": capture_views,
                "minimum_scan_views": MINIMUM_SCAN_VIEWS,
                "enhanced_scan_views": ENHANCED_SCAN_VIEWS,
                "renderer_mode": label_row.get("renderer_mode", ""),
                "render_source": label_row.get("render_source", ""),
                "is_smoke_dataset": label_row.get("is_smoke_dataset", ""),
                "is_training_candidate": label_row.get("is_training_candidate", ""),
                "quality_tier": label_row.get("quality_tier", ""),
                "label_row_index": str(label_row_index),
                "dataset_split": splits[sample_id],
            }
        )

    return manifest_rows


def _assign_splits(sample_ids: list[str]) -> dict[str, str]:
    count = len(sample_ids)
    train_count = round(count * 0.80)
    val_count = round(count * 0.10)
    if count >= 3:
        train_count = min(train_count, count - 2)
        val_count = max(1, min(val_count, count - train_count - 1))
    test_count = count - train_count - val_count

    split_by_sample: dict[str, str] = {}
    for index, sample_id in enumerate(sample_ids):
        if index < train_count:
            split_by_sample[sample_id] = "train"
        elif index < train_count + val_count:
            split_by_sample[sample_id] = "val"
        else:
            split_by_sample[sample_id] = "test"

    if test_count < 0:
        raise ValueError("Invalid split counts")

    return split_by_sample


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_manifest_report(result: dict[str, Any]) -> str:
    lines = [
        f"Manifest: {result['manifest_path']}",
        f"Valid: {result['valid']}",
        f"Rows: {result['row_count']}",
    ]
    if result["split_counts"]:
        lines.append("Splits:")
        for split in ("train", "val", "test"):
            lines.append(f"- {split}: {result['split_counts'].get(split, 0)}")
    if result["errors"]:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result["errors"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic manifest for a validated synthetic dataset.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root, such as data/synthetic/phase_2k.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--require-back", action="store_true")
    parser.add_argument("--require-realistic", action="store_true")
    args = parser.parse_args(argv)

    result = build_dataset_manifest(
        args.dataset,
        split_seed=args.split_seed,
        require_back=args.require_back,
        require_realistic=args.require_realistic,
    )
    print(format_manifest_report(result))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
