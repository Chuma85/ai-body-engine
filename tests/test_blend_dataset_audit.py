from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from synthetic.blender.blend_dataset import BLEND_LABEL_COLUMNS
from synthetic.blender.blend_dataset_audit import (
    IMPORTANT_MEASUREMENT_COLUMNS,
    audit_blend_dataset,
    validate_label_schema,
)


def test_audit_fails_cleanly_when_dataset_folder_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset folder does not exist"):
        audit_blend_dataset(tmp_path / "missing", tmp_path / "audit")


def test_audit_fails_cleanly_when_labels_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "metadata.json").write_text("{}\n", encoding="utf-8")
    (dataset / "images").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing labels.csv"):
        audit_blend_dataset(dataset, tmp_path / "audit")


def test_label_schema_validation_reports_missing_columns() -> None:
    result = validate_label_schema(["sample_id", "front_image"])

    assert result["valid"] is False
    assert "side_image" in result["missing_columns"]
    assert result["errors"]


def test_audit_detects_label_variation(tmp_path: Path) -> None:
    dataset = write_fake_blend_dataset(tmp_path, sample_count=4, vary_labels=True)

    report = audit_blend_dataset(dataset, tmp_path / "audit", expected_samples=4, strict=True)

    assert report["passed"] is True
    assert report["label_audit"]["variation_exists"] is True
    assert report["label_audit"]["identical_columns"] == []


def test_audit_flags_identical_measurement_columns_in_strict_mode(tmp_path: Path) -> None:
    dataset = write_fake_blend_dataset(tmp_path, sample_count=4, vary_labels=False)

    report = audit_blend_dataset(dataset, tmp_path / "audit", expected_samples=4, strict=True)

    assert report["passed"] is False
    assert set(IMPORTANT_MEASUREMENT_COLUMNS) == set(report["label_audit"]["identical_columns"])
    assert any("no variation" in failure for failure in report["strict_failures"])


def test_audit_writes_requested_outputs_for_small_fake_dataset(tmp_path: Path) -> None:
    dataset = write_fake_blend_dataset(tmp_path, sample_count=3)
    output_dir = tmp_path / "audit"

    report = audit_blend_dataset(dataset, output_dir, expected_samples=3, max_contact_sheet_samples=2)

    for filename in (
        "audit_report.json",
        "audit_summary.md",
        "sample_contact_sheet.png",
        "label_distribution_summary.csv",
        "flagged_samples.csv",
    ):
        assert (output_dir / filename).exists()
    assert report["usable"] is True
    assert report["view_sanity"]["passed"] is True
    assert report["flagged_sample_count"] == 0


def test_strict_audit_fails_near_identical_views(tmp_path: Path) -> None:
    dataset = write_fake_blend_dataset(tmp_path, sample_count=2, identical_views=True)

    report = audit_blend_dataset(dataset, tmp_path / "audit", expected_samples=2, strict=True)

    assert report["passed"] is False
    assert report["view_sanity"]["passed"] is False
    assert any("near-identical" in failure for failure in report["strict_failures"])


def write_fake_blend_dataset(
    tmp_path: Path,
    sample_count: int,
    vary_labels: bool = True,
    identical_views: bool = False,
) -> Path:
    dataset = tmp_path / "phase_3h_blend"
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True)
    metadata = {
        "generator_version": "phase_3h_blend_dataset_v1",
        "source_blend_file": "assets/body_meshes/base_body_scene.blend",
        "camera_set": "FrontCam,SideCam,BackCam",
        "sample_count": sample_count,
        "seed": 42,
        "synthetic_labels": True,
        "real_world_validated": False,
        "variation_source": "shape_keys_safe_range",
        "shape_key_count": 10,
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        for index in range(1, sample_count + 1):
            sample_id = f"sample_{index:06d}"
            image_names = {
                "front": f"images/{sample_id}_front.png",
                "side": f"images/{sample_id}_side.png",
                "back": f"images/{sample_id}_back.png",
            }
            write_view_images(dataset, image_names, identical_views)
            offset = index if vary_labels else 0
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "front_image": image_names["front"],
                    "side_image": image_names["side"],
                    "back_image": image_names["back"],
                    "height_cm": str(170 + offset),
                    "chest_cm": str(90 + offset),
                    "waist_cm": str(75 + offset),
                    "hip_cm": str(95 + offset),
                    "shoulder_cm": str(42 + offset),
                    "inseam_cm": str(78 + offset),
                    "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                    "variation_source": "shape_keys_safe_range",
                    "camera_set": "FrontCam,SideCam,BackCam",
                    "seed": "42",
                    "label_source": "existing_synthetic_label_generator",
                    "synthetic_labels": "true",
                    "real_world_validated": "false",
                }
            )
    return dataset


def write_view_images(dataset: Path, image_names: dict[str, str], identical_views: bool) -> None:
    if identical_views:
        for path in image_names.values():
            write_body_image(dataset / path, body_color=(40, 40, 40), rect=(44, 16, 84, 108))
        return
    write_body_image(dataset / image_names["front"], body_color=(40, 40, 40), rect=(42, 12, 86, 112))
    write_body_image(dataset / image_names["side"], body_color=(55, 55, 55), rect=(54, 12, 76, 112))
    write_body_image(dataset / image_names["back"], body_color=(70, 70, 70), rect=(38, 12, 90, 112))


def write_body_image(path: Path, body_color: tuple[int, int, int], rect: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (128, 128), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, fill=body_color)
    image.save(path)
