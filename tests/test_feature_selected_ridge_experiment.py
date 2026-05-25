import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.run_feature_selected_ridge_experiment import (
    MODEL_FAMILY,
    TARGET_COLUMNS,
    effective_feature_counts,
    normalize_feature_count_grid,
    rank_features_by_train_correlation,
    run_feature_selected_ridge_experiment,
)


def test_feature_ranking_uses_train_split_signal_only() -> None:
    train_features = np.asarray(
        [
            [1.0, 100.0, 0.0],
            [2.0, 10.0, 0.0],
            [3.0, 100.0, 0.0],
            [4.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    train_target = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    ranking = rank_features_by_train_correlation(train_features, train_target)

    assert int(ranking[0]) == 0
    assert int(ranking[-1]) == 2


def test_invalid_feature_count_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Feature counts must be positive"):
        normalize_feature_count_grid([10, 0, "all"])


def test_effective_feature_counts_clamp_and_include_all() -> None:
    assert effective_feature_counts([10, 25, "all"], total_feature_count=20) == [10, 20]


def test_feature_selected_experiment_runs_and_writes_outputs(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2aa"
    monkeypatch.chdir(tmp_path)

    result = run_feature_selected_ridge_experiment(
        dataset_root,
        output_dir,
        feature_count_grid=[5, 10, "all"],
    )

    expected_files = [
        "config.json",
        "metrics.json",
        "model.json",
        "feature_names.json",
        "per_target_errors.json",
        "predictions_train.csv",
        "predictions_val.csv",
        "predictions_test.csv",
        "selected_features.json",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    selected = json.loads((output_dir / "selected_features.json").read_text(encoding="utf-8"))
    assert result["metrics"]["model_family"] == MODEL_FAMILY
    assert metrics["sample_counts"] == {"train": 16, "val": 2, "test": 2}
    assert set(metrics["selected_feature_counts"]) == set(TARGET_COLUMNS)
    assert set(selected["selected_feature_counts"]) == set(TARGET_COLUMNS)
    assert set(selected["selected_features"]) == set(TARGET_COLUMNS)
    assert all(selected["selected_features"][target] for target in TARGET_COLUMNS)


def test_feature_selected_prediction_csv_has_rows(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2aa_predictions"
    monkeypatch.chdir(tmp_path)

    run_feature_selected_ridge_experiment(dataset_root, output_dir, feature_count_grid=[5, "all"])

    with (output_dir / "predictions_test.csv").open("r", newline="", encoding="utf-8") as predictions_file:
        rows = list(csv.DictReader(predictions_file))

    assert len(rows) == 2
    assert rows[0]["split"] == "test"
    for target in TARGET_COLUMNS:
        assert f"true_{target}" in rows[0]
        assert f"pred_{target}" in rows[0]
        assert f"abs_error_{target}" in rows[0]


def test_feature_selected_config_records_selection_settings(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2aa_config"
    monkeypatch.chdir(tmp_path)

    run_feature_selected_ridge_experiment(dataset_root, output_dir, feature_count_grid=[5, "all"], ridge_alpha=3.0)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["type"] == MODEL_FAMILY
    assert config["model"]["hyperparameters"]["ridge_alpha"] == 3.0
    assert config["model"]["hyperparameters"]["feature_count_grid"] == [5, "all"]
    assert config["model"]["hyperparameters"]["selection_split"] == "val"


def test_feature_selected_missing_dataset_raises_helpful_error(tmp_path) -> None:
    missing_dataset = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        run_feature_selected_ridge_experiment(missing_dataset, tmp_path / "artifacts" / "missing")


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_feature_selected_test"
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_dir = dataset_root / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    body_shapes = ["average", "athletic", "curvy", "broad"]
    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for index in range(1, count + 1):
            sample_id = f"sample_{index:06d}"
            front_width = 16 + (index % 8)
            side_width = 8 + (index % 5)
            _write_rect_image(front_dir / f"{sample_id}_front.png", rect=(20, 10, 20 + front_width, 54), size=(64, 64))
            _write_rect_image(side_dir / f"{sample_id}_side.png", rect=(24, 10, 24 + side_width, 54), size=(64, 64))
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(80 + front_width),
                    "waist_cm": str(70 + front_width),
                    "hip_cm": str(85 + front_width),
                    "shoulder_cm": str(38 + (index % 5)),
                    "inseam_cm": str(70 + (index % 8)),
                    "sleeve_cm": str(55 + (index % 7)),
                    "neck_cm": str(32 + (index % 4)),
                    "thigh_cm": str(45 + side_width),
                    "calf_cm": str(32 + (index % 6)),
                    "body_shape": body_shapes[index % len(body_shapes)],
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    result = build_dataset_manifest(dataset_root)
    assert result["valid"] is True
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (50, 50, 50))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)
