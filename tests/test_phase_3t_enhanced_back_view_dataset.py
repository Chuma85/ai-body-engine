from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.generate_phase_3t_enhanced_back_view import (
    DEFAULT_OUTPUT_DIR,
    generate_enhanced_dataset,
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


def test_enhanced_wrapper_generates_smoke_dataset_with_aligned_views(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_enhanced"

    result = generate_enhanced_dataset(
        output_dir=output_dir,
        sample_count=3,
        width=96,
        height=144,
        seed=123,
    )

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


def test_legacy_front_side_dataset_still_valid_without_back_view(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_legacy"

    generate_dataset(count=2, output_dir=str(output_dir), width=96, height=144, include_back_view=False)

    result = validate_dataset(output_dir)

    assert result["valid"] is True
    assert result["back_image_count"] == 0
    assert any("Back view is optional" in warning for warning in result["warnings"])
