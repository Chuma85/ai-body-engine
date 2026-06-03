from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

from scripts.generate_blend_dataset import generate_blend_dataset
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_SOURCE_MODE,
    build_blend_blender_command,
    camera_set_name,
    validate_blend_file_exists,
    validate_generated_blend_dataset,
)


def test_blend_dataset_command_includes_source_blend_file_and_cameras(tmp_path) -> None:
    blend_file = tmp_path / "base_body_scene.blend"
    output_dir = tmp_path / "phase_3h_blend"

    command = build_blend_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_blend_dataset.py",
        blend_file=str(blend_file),
        output_dir=str(output_dir),
        samples=3,
        seed=42,
    )

    assert command[:5] == [
        "blender",
        "--background",
        "--factory-startup",
        "--python",
        "synthetic/blender/scripts/render_blend_dataset.py",
    ]
    assert command[command.index("--source") + 1] == DEFAULT_SOURCE_MODE
    assert command[command.index("--blend-file") + 1] == str(blend_file)
    assert command[command.index("--out") + 1] == str(output_dir)
    assert command[command.index("--samples") + 1] == "3"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--front-camera") + 1] == DEFAULT_CAMERA_NAMES["front"]
    assert command[command.index("--side-camera") + 1] == DEFAULT_CAMERA_NAMES["side"]
    assert command[command.index("--back-camera") + 1] == DEFAULT_CAMERA_NAMES["back"]


def test_blend_dataset_dry_run_validates_config_without_blender(tmp_path) -> None:
    blend_file = tmp_path / "base_body_scene.blend"
    blend_file.write_bytes(b"BLENDER")

    result = generate_blend_dataset(
        blend_file=str(blend_file),
        out=str(tmp_path / "out"),
        samples=3,
        seed=99,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["mode"] == DEFAULT_SOURCE_MODE
    assert result["sample_count"] == 3
    assert result["seed"] == 99
    assert "--blend-file" in result["command"]
    assert str(blend_file.resolve()) in result["command"]


def test_blend_dataset_dry_run_allows_existing_output(tmp_path) -> None:
    blend_file = tmp_path / "base_body_scene.blend"
    output_dir = tmp_path / "out"
    blend_file.write_bytes(b"BLENDER")
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    result = generate_blend_dataset(
        blend_file=str(blend_file),
        out=str(output_dir),
        samples=1,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert (output_dir / "existing.txt").exists()


def test_missing_blend_file_fails_with_helpful_message(tmp_path) -> None:
    missing = tmp_path / "missing.blend"

    with pytest.raises(FileNotFoundError, match="Missing Blender .blend file"):
        validate_blend_file_exists(missing)


def test_validate_blend_dataset_output_structure_and_schema(tmp_path) -> None:
    output_dir = tmp_path / "phase_3h_blend"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text('{"synthetic_labels": true}\n', encoding="utf-8")

    rows = []
    for index in range(1, 4):
        sample_id = f"sample_{index:06d}"
        front_image = f"images/{sample_id}_front.png"
        side_image = f"images/{sample_id}_side.png"
        back_image = f"images/{sample_id}_back.png"
        for image in (front_image, side_image, back_image):
            (output_dir / image).write_bytes(b"png")
        rows.append(
            {
                "sample_id": sample_id,
                "front_image": front_image,
                "side_image": side_image,
                "back_image": back_image,
                "height_cm": "180.0",
                "chest_cm": "100.0",
                "waist_cm": "82.0",
                "hip_cm": "101.0",
                "shoulder_cm": "45.0",
                "inseam_cm": "82.0",
                "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                "variation_source": "static_blend_mesh",
                "camera_set": camera_set_name(),
                "seed": "42",
                "label_source": "existing_synthetic_label_generator",
                "synthetic_labels": "true",
                "real_world_validated": "false",
            }
        )

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    result = validate_generated_blend_dataset(output_dir, expected_samples=3)

    assert result["valid"] is True
    assert result["label_row_count"] == 3
    assert result["missing_image_paths"] == []


def test_validate_blend_dataset_rejects_row_count_mismatch(tmp_path) -> None:
    output_dir = tmp_path / "phase_3h_blend"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()

    result = validate_generated_blend_dataset(output_dir, expected_samples=3)

    assert result["valid"] is False
    assert any("row count" in error for error in result["errors"])


def test_render_blend_dataset_imports_without_bpy() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_blend_dataset")

    assert hasattr(module, "parse_args")
    assert hasattr(module, "find_required_cameras")
    assert hasattr(module, "blend_label_row")
