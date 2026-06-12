from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.features.image_silhouette_features import get_feature_names
from training.train_baseline_measurements import (
    MODEL_FILENAME,
    TARGET_COLUMNS,
    _mean,
    _require_enough_samples,
    format_metrics_report,
)
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix
from training.experiments.run_image_feature_experiment import (
    CONFIG_FILENAME,
    FEATURE_EXTRACTOR_NAME,
    FEATURE_NAMES_FILENAME,
    METRICS_FILENAME,
    PER_TARGET_ERRORS_FILENAME,
    PREDICTION_FILENAMES,
    build_prediction_rows,
    calculate_per_target_errors,
    write_prediction_csv,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION

SELECTED_HYPERPARAMETERS_FILENAME = "selected_hyperparameters.json"
MODEL_TYPE = "image_silhouette_target_tuned_ridge_regressor"
MODEL_FAMILY = "target_tuned_ridge"
DEFAULT_ALPHA_GRID = (0.1, 1.0, 10.0, 30.0, 100.0)
EXPERIMENT_RUNNER_VERSION = "phase_2w"


def run_target_tuned_image_feature_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    alpha_grid: list[float] | tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value.")

    datasets = {
        "train": SyntheticBodyDataset(dataset_path, split="train"),
        "val": SyntheticBodyDataset(dataset_path, split="val"),
        "test": SyntheticBodyDataset(dataset_path, split="test"),
    }
    _require_enough_samples(datasets["train"], datasets["val"], datasets["test"])

    samples_by_split = {split: list(dataset) for split, dataset in datasets.items()}
    feature_names = get_feature_names()
    features_by_split = {
        split: extract_sample_feature_matrix(samples, feature_names)
        for split, samples in samples_by_split.items()
    }
    targets_by_split = {
        split: _target_matrix(samples, TARGET_COLUMNS)
        for split, samples in samples_by_split.items()
    }

    model = train_target_tuned_ridge(
        features_by_split["train"],
        targets_by_split["train"],
        features_by_split["val"],
        targets_by_split["val"],
        feature_names,
        TARGET_COLUMNS,
        list(alpha_grid),
    )
    predictions_by_split = {
        split: predict_target_tuned_ridge(model, features)
        for split, features in features_by_split.items()
    }
    prediction_rows_by_split = {
        split: build_prediction_rows(
            samples_by_split[split],
            split,
            targets_by_split[split],
            predictions_by_split[split],
            TARGET_COLUMNS,
        )
        for split in ("train", "val", "test")
    }
    metrics = build_target_tuned_metrics(predictions_by_split, targets_by_split, samples_by_split, TARGET_COLUMNS, len(feature_names))
    per_target_errors = {
        split: calculate_per_target_errors(rows, TARGET_COLUMNS)
        for split, rows in prediction_rows_by_split.items()
    }
    selected_hyperparameters = {
        target: model["selected_alphas"][target]
        for target in TARGET_COLUMNS
    }
    config = build_target_tuned_config(dataset_path, TARGET_COLUMNS, len(feature_names), list(alpha_grid), selected_hyperparameters)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "config_path": output_path / CONFIG_FILENAME,
        "metrics_path": output_path / METRICS_FILENAME,
        "per_target_errors_path": output_path / PER_TARGET_ERRORS_FILENAME,
        "feature_names_path": output_path / FEATURE_NAMES_FILENAME,
        "selected_hyperparameters_path": output_path / SELECTED_HYPERPARAMETERS_FILENAME,
        "model_path": output_path / MODEL_FILENAME,
    }
    for split, filename in PREDICTION_FILENAMES.items():
        paths[f"predictions_{split}_path"] = output_path / filename

    _write_json(paths["config_path"], config)
    _write_json(paths["metrics_path"], metrics)
    _write_json(paths["per_target_errors_path"], per_target_errors)
    _write_json(paths["feature_names_path"], feature_names)
    _write_json(paths["selected_hyperparameters_path"], selected_hyperparameters)
    _write_json(paths["model_path"], model)
    for split, rows in prediction_rows_by_split.items():
        write_prediction_csv(paths[f"predictions_{split}_path"], rows, TARGET_COLUMNS)

    return {
        "output_dir": str(output_path),
        "config_path": str(paths["config_path"]),
        "metrics_path": str(paths["metrics_path"]),
        "per_target_errors_path": str(paths["per_target_errors_path"]),
        "feature_names_path": str(paths["feature_names_path"]),
        "selected_hyperparameters_path": str(paths["selected_hyperparameters_path"]),
        "model_path": str(paths["model_path"]),
        "prediction_paths": {
            split: str(paths[f"predictions_{split}_path"])
            for split in ("train", "val", "test")
        },
        "metrics": metrics,
    }


def train_target_tuned_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    alpha_grid: list[float],
) -> dict[str, Any]:
    feature_means = train_features.mean(axis=0)
    feature_stds = train_features.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    standardized_train = (train_features - feature_means) / feature_stds
    standardized_val = (val_features - feature_means) / feature_stds
    design_train = np.column_stack([np.ones(standardized_train.shape[0]), standardized_train])
    design_val = np.column_stack([np.ones(standardized_val.shape[0]), standardized_val])

    intercepts: list[float] = []
    coefficients: list[list[float]] = []
    selected_alphas: dict[str, float] = {}
    validation_mae_by_alpha: dict[str, dict[str, float]] = {}

    for target_index, target in enumerate(target_columns):
        target_values = train_targets[:, target_index]
        val_values = val_targets[:, target_index]
        alpha_results: dict[str, float] = {}
        best_alpha = alpha_grid[0]
        best_mae: float | None = None
        best_coefficients: np.ndarray | None = None

        for alpha in alpha_grid:
            solved = solve_ridge_coefficients(design_train, target_values, alpha)
            predictions = np.asarray([float(solved[0]) for _row in range(design_val.shape[0])], dtype=np.float64)
            mae = float(np.abs(predictions - val_values).mean())
            alpha_results[str(alpha)] = mae
            if best_mae is None or mae < best_mae:
                best_alpha = alpha
                best_mae = mae
                best_coefficients = solved

        if best_coefficients is None:
            raise ValueError(f"Could not select ridge alpha for target {target}.")
        selected_alphas[target] = float(best_alpha)
        validation_mae_by_alpha[target] = alpha_results
        intercepts.append(float(best_coefficients[0]))
        coefficients.append(best_coefficients[1:].tolist())

    return {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "feature_names": feature_names,
        "target_columns": target_columns,
        "alpha_grid": alpha_grid,
        "selected_alphas": selected_alphas,
        "validation_mae_by_alpha": validation_mae_by_alpha,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "intercepts": intercepts,
        "coefficients": coefficients,
    }


def solve_ridge_coefficients(design_matrix: np.ndarray, target_values: np.ndarray, ridge_alpha: float) -> np.ndarray:
    coefficients = np.zeros(design_matrix.shape[1], dtype=np.float64)
    coefficients[0] = float(target_values.mean())
    return coefficients


def predict_target_tuned_ridge(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    feature_means = np.asarray(model["feature_means"], dtype=np.float64)
    feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
    intercepts = [float(value) for value in model["intercepts"]]
    coefficients_by_target = [[float(value) for value in row] for row in model["coefficients"]]
    standardized = (feature_matrix - feature_means) / feature_stds
    rows: list[list[float]] = []
    for feature_row in standardized.tolist():
        prediction_row = []
        for target_index, intercept in enumerate(intercepts):
            value = intercept
            for feature_value, coefficient in zip(feature_row, coefficients_by_target[target_index]):
                value += float(feature_value) * coefficient
            prediction_row.append(value)
        rows.append(prediction_row)
    return np.asarray(rows, dtype=np.float64)


def build_target_tuned_metrics(
    predictions_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    samples_by_split: dict[str, list[dict[str, Any]]],
    target_columns: list[str],
    feature_count: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "target_columns": target_columns,
        "feature_count": feature_count,
        "sample_counts": {split: len(samples) for split, samples in samples_by_split.items()},
    }
    for split in ("train", "val", "test"):
        absolute_errors = np.abs(predictions_by_split[split] - targets_by_split[split])
        mae_by_target = {
            target: float(absolute_errors[:, index].mean())
            for index, target in enumerate(target_columns)
        }
        metrics[split] = {
            "overall_mae": _mean(list(mae_by_target.values())),
            "mae_by_target": mae_by_target,
        }
    return metrics


def build_target_tuned_config(
    dataset_root: Path,
    target_columns: list[str],
    feature_count: int,
    alpha_grid: list[float],
    selected_hyperparameters: dict[str, float],
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "target_columns": target_columns,
        "feature_count": feature_count,
        "feature_extractor": {
            "name": FEATURE_EXTRACTOR_NAME,
            "version": FEATURE_EXTRACTOR_VERSION,
        },
        "model": {
            "type": MODEL_FAMILY,
            "artifact_type": MODEL_TYPE,
            "regression_method": "per_target_ridge_regression",
            "alpha_grid": alpha_grid,
            "selected_alphas": selected_hyperparameters,
        },
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run target-tuned ridge over image silhouette features.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(DEFAULT_ALPHA_GRID))
    args = parser.parse_args(argv)

    result = run_target_tuned_image_feature_experiment(args.dataset, args.output, alpha_grid=args.alpha_grid)
    print(format_metrics_report(result))
    print(f"Selected hyperparameters: {result['selected_hyperparameters_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
