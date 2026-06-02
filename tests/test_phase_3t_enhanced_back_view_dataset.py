from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.generate_phase_3t_enhanced_back_view import (
    DEFAULT_OUTPUT_DIR,
    REALISTIC_MODE,
    SMOKE_MODE,
    build_realistic_blender_command,
    generate_enhanced_dataset,
    generate_realistic_blender_dataset,
    sample_alignment_summary,
)
from synthetic.blender.utils.render_config import load_render_config
from synthetic.generator.generate_dataset import generate_dataset
from synthetic.validate_synthetic_dataset import validate_dataset


ENHANCED_CONFIG_PATH = "synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json"


def test_phase_3t_enhanced_config_targets_back_view_dataset() -> None:
    config = load_render_config(ENHANCED_CONFIG_PATH)

    assert config.generator_version == "phase_3t_b2_enhanced_back_view_v1"
    assert config.output_dir == DEFAULT_OUTPUT_DIR
    assert config.views == ["front", "side", "back"]
    assert config.sample_count == 1000
    assert config.base_mesh is not None
    assert config.base_mesh["asset_path"] == "assets/body_meshes/base_human_rigged.fbx"
    assert config.mesh_deformation is not None
    assert config.mesh_deformation["enabled"] is True
    assert config.render_realism is not None
    assert config.render_realism["enabled"] is True


def test_realistic_blender_mode_is_default_generation_route() -> None:
    result = generate_realistic_blender_dataset(
        output_dir=DEFAULT_OUTPUT_DIR,
        sample_count=5,
        overwrite=True,
        dry_run=True,
    )

    assert result["mode"] == REALISTIC_MODE
    assert Path(result["output_dir"]) == Path(DEFAULT_OUTPUT_DIR)
    assert "--config" in result["command"]
    assert "synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json" in result["command"]
    assert "--output" in result["command"]
    assert Path(result["command"][result["command"].index("--output") + 1]) == Path(DEFAULT_OUTPUT_DIR)
    assert "--num-samples" in result["command"]
    assert "5" in result["command"]


def test_realistic_blender_command_includes_front_side_back_config_and_output() -> None:
    command = build_realistic_blender_command(
        blender_executable="blender",
        config_path=ENHANCED_CONFIG_PATH,
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        output_dir=DEFAULT_OUTPUT_DIR,
        sample_count=1000,
    )

    assert command[:4] == ["blender", "--background", "--python", "synthetic/blender/scripts/render_parametric_body.py"]
    assert command[command.index("--config") + 1] == ENHANCED_CONFIG_PATH
    assert command[command.index("--output") + 1] == DEFAULT_OUTPUT_DIR
    assert command[command.index("--num-samples") + 1] == "1000"


def test_enhanced_wrapper_generates_smoke_dataset_with_aligned_views(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced"

    result = generate_enhanced_dataset(
        output_dir=output_dir,
        sample_count=3,
        width=96,
        height=144,
        seed=123,
    )

    assert result["mode"] == SMOKE_MODE
    assert (output_dir / "images" / "front").exists()
    assert (output_dir / "images" / "side").exists()
    assert (output_dir / "images" / "back").exists()
    assert (output_dir / "labels" / "labels.csv").exists()
    assert (output_dir / "manifest.csv").exists()
    assert result["alignment"]["aligned_sample_count"] == 3
    assert sample_alignment_summary(output_dir)["missing_back"] == []

    for sample_id in ("sample_000001", "sample_000002", "sample_000003"):
        assert (output_dir / "images" / "front" / f"{sample_id}_front.png").exists()
        assert (output_dir / "images" / "side" / f"{sample_id}_side.png").exists()
        assert (output_dir / "images" / "back" / f"{sample_id}_back.png").exists()


def test_enhanced_wrapper_manifest_includes_back_view_metadata(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced"

    generate_enhanced_dataset(output_dir=output_dir, sample_count=2, width=96, height=144)

    with (output_dir / "manifest.csv").open("r", newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    assert len(rows) == 2
    assert all(row["back_image_path"].endswith("_back.png") for row in rows)
    assert all(row["has_back"] == "true" for row in rows)
    assert all(row["capture_views"] == "front,side,back" for row in rows)
    assert all(row["renderer_mode"] == "lightweight_smoke" for row in rows)
    assert all(row["render_source"] == "python_silhouette_placeholder" for row in rows)
    assert all(row["quality_tier"] == "smoke_only" for row in rows)


def test_enhanced_wrapper_requires_overwrite_for_existing_output(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_enhanced_dataset(output_dir=output_dir, sample_count=1, width=64, height=96)

    result = generate_enhanced_dataset(
        output_dir=output_dir,
        sample_count=1,
        width=64,
        height=96,
        overwrite=True,
    )

    assert not (output_dir / "existing.txt").exists()
    assert result["sample_count"] == 1
    assert (output_dir / "images" / "back" / "sample_000001_back.png").exists()


def test_realistic_validation_rejects_lightweight_smoke_output(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced_smoke"

    generate_enhanced_dataset(output_dir=output_dir, sample_count=2, width=96, height=144)

    result = validate_dataset(output_dir, require_back=True, require_realistic=True)

    assert result["valid"] is False
    assert result["non_realistic_label_rows"] == ["sample_000001", "sample_000002"]


def test_realistic_validation_accepts_blender_training_candidate_metadata(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced_realistic"

    generate_enhanced_dataset(output_dir=output_dir, sample_count=2, width=96, height=144)
    _rewrite_labels_as_blender_training_candidate(output_dir / "labels" / "labels.csv")

    result = validate_dataset(output_dir, require_back=True, require_realistic=True)

    assert result["valid"] is True
    assert result["non_realistic_label_rows"] == []


def test_legacy_front_side_dataset_still_valid_without_back_view(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_legacy"

    generate_dataset(count=2, output_dir=str(output_dir), width=96, height=144, include_back_view=False)

    result = validate_dataset(output_dir)

    assert result["valid"] is True
    assert result["back_image_count"] == 0
    assert any("Back view is optional" in warning for warning in result["warnings"])


def _rewrite_labels_as_blender_training_candidate(labels_path: Path) -> None:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    for row in rows:
        row["renderer_mode"] = "base_mesh"
        row["render_source"] = "blender_body_mesh"
        row["is_smoke_dataset"] = "False"
        row["is_training_candidate"] = "True"
        row["quality_tier"] = "training_candidate"

    with labels_path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
