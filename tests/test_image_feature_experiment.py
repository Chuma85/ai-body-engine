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
from training.experiments.compare_image_feature_models import compare_image_feature_models
from training.experiments.analyze_target_diagnostics import (
    analyze_target_diagnostics,
    calculate_target_diagnostics,
    percent_mae,
    signed_bias,
    worst_sample_rows,
)
from training.experiments.run_target_tuned_image_feature_experiment import (
    run_target_tuned_image_feature_experiment,
)
from training.experiments.analyze_feature_importance import (
    analyze_feature_importance,
    dominant_feature_warning,
    feature_group,
    near_constant_feature_names,
    rank_absolute_values,
    rank_negative_values,
    rank_positive_values,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION


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


def test_target_diagnostics_percent_mae_and_signed_bias() -> None:
    rows = [
        {"true_height_cm": "100.0", "pred_height_cm": "110.0", "abs_error_height_cm": "10.0"},
        {"true_height_cm": "200.0", "pred_height_cm": "180.0", "abs_error_height_cm": "20.0"},
    ]

    diagnostics = calculate_target_diagnostics(rows, "height_cm")

    assert percent_mae(15.0, 150.0) == 10.0
    assert signed_bias([10.0, -20.0]) == -5.0
    assert diagnostics["mae"] == 15.0
    assert diagnostics["mean_true"] == 150.0
    assert diagnostics["mae_percent_of_mean_true"] == 10.0
    assert diagnostics["signed_error_mean"] == -5.0
    assert diagnostics["underprediction_count"] == 1
    assert diagnostics["overprediction_count"] == 1


def test_worst_sample_extraction_orders_by_absolute_error() -> None:
    rows = [
        {"sample_id": "a", "split": "test", "true_height_cm": "100", "pred_height_cm": "104", "abs_error_height_cm": "4"},
        {"sample_id": "b", "split": "test", "true_height_cm": "100", "pred_height_cm": "112", "abs_error_height_cm": "12"},
        {"sample_id": "c", "split": "test", "true_height_cm": "100", "pred_height_cm": "108", "abs_error_height_cm": "8"},
    ]

    worst_rows = worst_sample_rows(rows, ["height_cm"], limit_per_target=2)

    assert [row["sample_id"] for row in worst_rows] == ["b", "c"]
    assert worst_rows[0]["signed_error"] == 12.0


def test_feature_importance_coefficient_rankings() -> None:
    values = np.asarray([0.2, -2.0, 1.5, -0.1])

    assert rank_positive_values(values, 2) == [2, 0]
    assert rank_negative_values(values, 2) == [1, 3]
    assert rank_absolute_values(values, 2) == [1, 2]


def test_feature_group_detection_from_feature_names() -> None:
    assert feature_group("front_bbox_height_ratio") == "bbox_scale_position"
    assert feature_group("side_waist_width_ratio") == "band_width_profile"
    assert feature_group("front_arm_span_to_torso_ratio") == "arm_span_extension"
    assert feature_group("front_thigh_to_height_ratio") == "height_normalized_ratio"
    assert feature_group("front_to_side_area_ratio") == "cross_view_ratio"


def test_near_constant_feature_detection() -> None:
    names = ["constant_feature", "variable_feature"]
    stds = np.asarray([0.0, 0.25])

    assert near_constant_feature_names(names, stds) == ["constant_feature"]


def test_low_signal_and_dominant_feature_warning_logic() -> None:
    coefficients = np.asarray([9.0, 1.0, 0.0])

    warning = dominant_feature_warning("weight_kg", ["front_bbox_height_ratio", "side_waist_width_ratio", "front_area"], coefficients)

    assert warning is not None
    assert "Dominant feature" in warning


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


def test_target_diagnostics_report_is_created(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    experiment_dir = tmp_path / "artifacts" / "experiments" / "phase_2w_source"
    output_dir = tmp_path / "artifacts" / "analysis" / "phase_2w_diagnostics"
    monkeypatch.chdir(tmp_path)
    run_image_feature_experiment(dataset_root, experiment_dir)

    result = analyze_target_diagnostics(experiment_dir, output_dir)

    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["worst_samples_path"]).exists()
    assert "height_cm" in result["summary"]["per_target"]
    assert result["summary"]["hardest_targets"]


def test_feature_importance_report_and_csvs_are_created(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    experiment_dir = tmp_path / "artifacts" / "experiments" / "phase_2x_source"
    output_dir = tmp_path / "artifacts" / "analysis" / "phase_2x_features"
    monkeypatch.chdir(tmp_path)
    run_image_feature_experiment(dataset_root, experiment_dir)

    result = analyze_feature_importance(experiment_dir, dataset_root, output_dir)

    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["per_target_top_features_path"]).exists()
    assert Path(result["feature_group_summary_path"]).exists()
    assert result["summary"]["feature_count"] > 0
    assert "height_cm" in result["summary"]["per_target"]
    assert "band_width_profile" in {row["feature_group"] for row in result["summary"]["feature_groups"]}


def test_feature_importance_missing_model_raises_helpful_error(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing model.json"):
        analyze_feature_importance(experiment_dir, tmp_path / "dataset", tmp_path / "analysis")


def test_target_diagnostics_missing_predictions_raise_helpful_error(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "metrics.json").write_text(json.dumps({"target_columns": ["height_cm"]}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing prediction CSV"):
        analyze_target_diagnostics(experiment_dir, tmp_path / "analysis")


def test_target_tuned_experiment_runs_and_writes_outputs(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2w_tuned"
    monkeypatch.chdir(tmp_path)

    result = run_target_tuned_image_feature_experiment(dataset_root, output_dir, alpha_grid=[0.1, 1.0])

    expected_files = [
        "config.json",
        "metrics.json",
        "model.json",
        "feature_names.json",
        "per_target_errors.json",
        "predictions_train.csv",
        "predictions_val.csv",
        "predictions_test.csv",
        "selected_hyperparameters.json",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()
    selected = json.loads((output_dir / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    assert set(selected) == set(TARGET_COLUMNS)
    assert set(selected.values()) <= {0.1, 1.0}
    assert result["metrics"]["model_family"] == "target_tuned_ridge"


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
    assert config["feature_count"] > 0
    assert config["feature_extractor"]["name"] == "image_silhouette_features"
    assert config["feature_extractor"]["version"] == FEATURE_EXTRACTOR_VERSION
    assert config["model"]["type"] == "ridge"
    assert config["model"]["artifact_type"] == MODEL_TYPE
    assert config["model"]["regression_method"] == "ridge_regression"
    assert config["model"]["hyperparameters"]["ridge_alpha"] == 7.5
    assert "created_at_utc" in config


def test_model_selection_supports_mean_and_ridge(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2t_mean"
    monkeypatch.chdir(tmp_path)

    result = run_image_feature_experiment(dataset_root, output_dir, model_type="mean")

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["type"] == "mean"
    assert config["model"]["hyperparameters"] == {}
    assert result["metrics"]["model_family"] == "mean"
    assert result["metrics"]["model_type"] == "image_feature_mean_regressor"


def test_model_selection_supports_knn(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2t_knn"
    monkeypatch.chdir(tmp_path)

    result = run_image_feature_experiment(dataset_root, output_dir, model_type="knn", knn_k=3)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
    assert config["model"]["type"] == "knn"
    assert config["model"]["hyperparameters"]["k"] == 3
    assert model["hyperparameters"]["k"] == 3
    assert result["metrics"]["model_family"] == "knn"


def test_invalid_model_type_raises_clear_error(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 20)

    with pytest.raises(ValueError, match="Unknown model type"):
        run_image_feature_experiment(dataset_root, tmp_path / "artifacts" / "bad_model", model_type="tree")


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


def test_comparison_runner_creates_model_subdirs_summary_and_report(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "experiments" / "phase_2t_compare"
    monkeypatch.chdir(tmp_path)

    result = compare_image_feature_models(dataset_root, output_dir, model_types=["mean", "ridge"])

    assert (output_dir / "mean" / "metrics.json").exists()
    assert (output_dir / "ridge" / "metrics.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["summary"]["model_types"] == ["mean", "ridge"]
    assert result["summary"]["best_model_overall"]["model_type"] in {"mean", "ridge"}
    assert "height_cm" in result["summary"]["best_model_per_target"]


def test_comparison_runner_invalid_model_raises_clear_error(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 20)

    with pytest.raises(ValueError, match="Unknown model type"):
        compare_image_feature_models(dataset_root, tmp_path / "artifacts" / "bad_compare", model_types=["ridge", "tree"])


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
