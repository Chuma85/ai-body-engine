from __future__ import annotations

import csv
from pathlib import Path

import pytest

from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.validate_synthetic_dataset import validate_dataset
from training.experiments.benchmark_back_view_shoulder import (
    build_feature_sets,
    generate_controlled_back_view_dataset,
    run_back_view_shoulder_benchmark,
)
from training.features.back_view_features import extract_back_view_features
from training.datasets.synthetic_body_dataset import SyntheticBodyDataset


def test_back_view_manifest_entries_are_valid_and_aligned(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "phase_5l"
    generate_controlled_back_view_dataset(dataset, sample_count=24, seed=11)
    monkeypatch.chdir(tmp_path)

    result = build_dataset_manifest(dataset, require_back=True)
    rows = _read_manifest(dataset / "manifest.csv")

    assert result["valid"] is True
    assert all(row["back_image_path"] for row in rows)
    assert all(Path(row["front_image_path"]).stem.removesuffix("_front") == row["sample_id"] for row in rows)
    assert all(Path(row["side_image_path"]).stem.removesuffix("_side") == row["sample_id"] for row in rows)
    assert all(Path(row["back_image_path"]).stem.removesuffix("_back") == row["sample_id"] for row in rows)


def test_back_view_feature_extractor_is_deterministic(tmp_path) -> None:
    dataset = tmp_path / "phase_5l"
    generate_controlled_back_view_dataset(dataset, sample_count=3, seed=12)
    back_path = dataset / "images" / "back" / "sample_000001_back.png"

    first = extract_back_view_features(back_path)
    second = extract_back_view_features(back_path)

    assert first == second
    assert first["back_shoulder_width_proxy"] > 0.0
    assert first["back_across_back_width_proxy"] > 0.0
    assert first["back_torso_width_band_29"] > 0.0


def test_missing_back_view_is_handled_clearly(tmp_path) -> None:
    dataset = tmp_path / "phase_5l"
    generate_controlled_back_view_dataset(dataset, sample_count=3, seed=13)
    back_path = dataset / "images" / "back" / "sample_000001_back.png"
    back_path.unlink()

    validation = validate_dataset(dataset, require_back=True)

    assert validation["valid"] is False
    assert any("label_rows_missing_image_pairs" in error for error in validation["errors"])
    with pytest.raises(FileNotFoundError, match="Missing back image"):
        extract_back_view_features(back_path)


def test_front_side_baseline_remains_available(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "phase_5l"
    generate_controlled_back_view_dataset(dataset, sample_count=24, seed=14)
    monkeypatch.chdir(tmp_path)
    samples_by_split = {
        split: list(SyntheticBodyDataset(dataset, split=split))
        for split in ("train", "val", "test")
    }

    feature_sets = build_feature_sets(samples_by_split)

    assert "front_side_baseline" in feature_sets
    assert "back_only" in feature_sets
    assert "front_side_back_combined" in feature_sets
    assert feature_sets["front_side_baseline"]["features_by_split"]["train"].shape[0] > 0


def test_benchmark_output_schema_is_stable(tmp_path, monkeypatch) -> None:
    output = tmp_path / "artifacts" / "phase_5l"
    dataset = output / "dataset"
    monkeypatch.chdir(tmp_path)

    result = run_back_view_shoulder_benchmark(
        dataset_root=dataset,
        output_dir=output,
        sample_count=30,
        seed=15,
        model_types=["ridge"],
        regenerate_dataset=True,
    )

    expected_artifacts = [
        "dataset_validation.json",
        "dataset_validation.csv",
        "benchmark_results.json",
        "benchmark_results.csv",
        "per_target_results.csv",
        "back_view_feature_summary.md",
        "recommendation_summary.md",
    ]
    for filename in expected_artifacts:
        assert (output / filename).exists()
    with (output / "benchmark_results.csv").open("r", newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert {
        "run_name",
        "feature_set",
        "model_type",
        "feature_count",
        "test_shoulder_group_mae",
        "test_shoulder_cm_mae",
    } <= set(row)
    assert result["summary"]["back_view_improves_shoulder"] in {True, False}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))
