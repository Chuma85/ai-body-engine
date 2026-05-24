import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.run_image_feature_experiment import (
    MODEL_TYPE,
    TARGET_COLUMNS,
    build_prediction_rows,
    calculate_per_target_errors,
    main,
    prediction_fieldnames,
    run_image_feature_experiment,
)


def test_prediction_rows_contain_true_pred_and_error_columns() -> None:
    samples = [{"sample_id": "sample_000001"}]
    targets = np.asarray([[170.0, 70.0]], dtype=np.float64)
    predictions = np.asarray([[172.5, 66.0]], dtype=np.float64)
    target_columns = ["height_cm", "weight_kg"]

    rows = build_prediction_rows(samples, "test", targets, predictions, target_columns)

    assert rows == [
        {
            "sample_id": "sample_000001",
            "split": "test",
            "true_height_cm": 170.0,
            "pred_height_cm": 172.5,
            "abs_error_height_cm": 2.5,
            "true_weight_kg": 70.0,
            "pred_weight_kg": 66.0,
            "abs_error_weight_kg": 4.0,
        }
    ]
    assert prediction_fieldnames(target_columns) == [
        "sample_id",
        "split",
        "true_height_cm",
        "pred_height_cm",
        "abs_error_height_cm",
        "true_weight_kg",
        "pred_weight_kg",
        "abs_error_weight_kg",
    ]


def test_per_target_errors_are_calculated_correctly() -> None:
    rows = [
        {"abs_error_height_cm": 1.0, "abs_error_weight_kg": 4.0},
        {"abs_error_height_cm": 3.0, "abs_error_weight_kg": 6.0},
    ]

    errors = calculate_per_target_errors(rows, ["height_cm", "weight_kg"])

    assert errors["height_cm"]["count"] == 2
    assert errors["height_cm"]["mae"] == 2.0
    assert errors["height_cm"]["max_abs_error"] == 3.0
    assert errors["weight_kg"]["mae"] == 5.0


def test_tiny_fixture_experiment_creates_complete_outputs(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2s"
    monkeypatch.chdir(tmp_path)

    result = run_image_feature_experiment(dataset_root, output_dir)

    expected_files = [
        "config.json",
        "metrics.json",
        "model.json",
        "feature_names.json",
        "per_target_errors.json",
        "predictions_train.csv",
        "predictions_val.csv",
        "predictions_test.csv",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_type"] == MODEL_TYPE
    assert metrics["sample_counts"] == {"train": 16, "val": 2, "test": 2}
    assert metrics["feature_count"] > 0
    assert "overall_mae" in metrics["test"]
    assert result["metrics"]["test"]["overall_mae"] == metrics["test"]["overall_mae"]


def test_prediction_csv_has_expected_columns_and_rows(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2s_predictions"
    monkeypatch.chdir(tmp_path)

    run_image_feature_experiment(dataset_root, output_dir)

    with (output_dir / "predictions_test.csv").open("r", newline="", encoding="utf-8") as predictions_file:
        rows = list(csv.DictReader(predictions_file))

    assert len(rows) == 2
    assert rows[0]["sample_id"].startswith("sample_")
    assert rows[0]["split"] == "test"
    for target in TARGET_COLUMNS:
        assert f"true_{target}" in rows[0]
        assert f"pred_{target}" in rows[0]
        assert f"abs_error_{target}" in rows[0]


def test_feature_names_are_stable_and_non_empty(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2s_features"
    monkeypatch.chdir(tmp_path)

    run_image_feature_experiment(dataset_root, output_dir)

    feature_names = json.loads((output_dir / "feature_names.json").read_text(encoding="utf-8"))
    assert feature_names[:4] == [
        "front_image_width_px",
        "front_image_height_px",
        "front_foreground_area_ratio",
        "front_bbox_width_px",
    ]
    assert "front_to_side_bbox_height_ratio" in feature_names


def test_config_contains_dataset_targets_and_model_settings(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2s_config"
    monkeypatch.chdir(tmp_path)

    run_image_feature_experiment(dataset_root, output_dir, ridge_alpha=7.5)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["dataset"] == str(dataset_root)
    assert config["target_columns"] == TARGET_COLUMNS
    assert config["feature_extractor"]["name"] == "image_silhouette_features"
    assert config["feature_extractor"]["version"] == "phase_2p"
    assert config["model"]["regression_method"] == "ridge_regression"
    assert config["model"]["ridge_alpha"] == 7.5
    assert "created_at_utc" in config


def test_experiment_cli_creates_outputs(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2s_cli"
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dataset", str(dataset_root), "--output", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions_train.csv").exists()


def test_missing_dataset_raises_helpful_error(tmp_path) -> None:
    missing_dataset = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        run_image_feature_experiment(missing_dataset, tmp_path / "artifacts" / "missing")


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_experiment_test"
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
