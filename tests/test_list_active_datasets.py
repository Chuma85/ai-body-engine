from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.list_active_datasets import (
    ACTIVE_DATASET_PATH,
    ARCHIVE_DATASET_PATH,
    dataset_counts,
    format_active_dataset_report,
    top_level_synthetic_entries,
)


def test_dataset_counts_reports_labels_and_images(tmp_path: Path) -> None:
    dataset = write_active_dataset(tmp_path)

    counts = dataset_counts(dataset)

    assert counts["exists"] is True
    assert counts["labels_exists"] is True
    assert counts["manifest_exists"] is True
    assert counts["label_rows"] == 2
    assert counts["image_count"] == 4


def test_active_dataset_report_warns_about_archive() -> None:
    report = format_active_dataset_report(
        {
            "path": str(ACTIVE_DATASET_PATH),
            "exists": True,
            "labels_path": str(ACTIVE_DATASET_PATH / "labels" / "labels.csv"),
            "labels_exists": True,
            "manifest_path": str(ACTIVE_DATASET_PATH / "manifest.csv"),
            "manifest_exists": True,
            "label_rows": 1000,
            "image_count": 2000,
        }
    )

    assert str(ACTIVE_DATASET_PATH) in report
    assert str(ARCHIVE_DATASET_PATH) in report
    assert "not used for active training" in report


def test_top_level_entries_can_confirm_archive_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "data" / "synthetic"
    (root / "phase_3t").mkdir(parents=True)
    (root / "_archived_old_mannequin").mkdir()
    (root / "ACTIVE_DATASET.md").write_text("active", encoding="utf-8")

    assert top_level_synthetic_entries(root) == ["ACTIVE_DATASET.md", "_archived_old_mannequin", "phase_3t"]


def write_active_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "data" / "synthetic" / "phase_3t"
    front_dir = dataset / "images" / "front"
    side_dir = dataset / "images" / "side"
    labels_dir = dataset / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for index in range(1, 3):
        sample_id = f"sample_{index:06d}"
        Image.new("RGB", (8, 8), (20, 20, 20)).save(front_dir / f"{sample_id}_front.png")
        Image.new("RGB", (8, 8), (40, 40, 40)).save(side_dir / f"{sample_id}_side.png")
    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerow({"sample_id": "sample_000001"})
        writer.writerow({"sample_id": "sample_000002"})
    (dataset / "manifest.csv").write_text("sample_id\nsample_000001\nsample_000002\n", encoding="utf-8")
    return dataset
