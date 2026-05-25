from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
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
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import (
    MODEL_FILENAME,
    TARGET_COLUMNS,
    _mean,
    _require_enough_samples,
    format_metrics_report,
)
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix

MODEL_TYPE = "image_silhouette_feature_selected_ridge_regressor"
MODEL_FAMILY = "feature_selected_ridge"
SELECTED_FEATURES_FILENAME = "selected_features.json"
DEFAULT_FEATURE_COUNT_GRID = (10, 25, 50, 100, "all")
DEFAULT_RIDGE_ALPHA = 10.0
EXPERIMENT_RUNNER_VERSION = "phase_2aa"


def run_feature_selected_ridge_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    feature_count_grid: list[int | str] | tuple[int | str, ...] = DEFAULT_FEATURE_COUNT_GRID,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")
    normalized_grid = normalize_feature_count_grid(feature_count_grid)

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

    model = train_feature_selected_ridge(
        features_by_split["train"],
        targets_by_split["train"],
        features_by_split["val"],
        targets_by_split["val"],
        feature_names,
        TARGET_COLUMNS,
        normalized_grid,
        ridge_alpha=ridge_alpha,
    )
    predictions_by_split = {
        split: predict_feature_selected_ridge(model, features)
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
    metrics = build_feature_selected_metrics(
        predictions_by_split,
        targets_by_split,
        samples_by_split,
        TARGET_COLUMNS,
        feature_count=len(feature_names),
        selected_feature_counts=model["selected_feature_counts"],
    )
    per_target_errors = {
        split: calculate_per_target_errors(rows, TARGET_COLUMNS)
        for split, rows in prediction_rows_by_split.items()
    }
    selected_features = build_selected_features_payload(model)
    config = build_config(dataset_path, TARGET_COLUMNS, len(feature_names), normalized_grid, ridge_alpha)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "config_path": output_path / CONFIG_FILENAME,
        "metrics_path": output_path / METRICS_FILENAME,
        "per_target_errors_path": output_path / PER_TARGET_ERRORS_FILENAME,
        "feature_names_path": output_path / FEATURE_NAMES_FILENAME,
        "selected_features_path": output_path / SELECTED_FEATURES_FILENAME,
        "model_path": output_path / MODEL_FILENAME,
    }
    for split, filename in PREDICTION_FILENAMES.items():
        paths[f"predictions_{split}_path"] = output_path / filename

    _write_json(paths["config_path"], config)
    _write_json(paths["metrics_path"], metrics)
    _write_json(paths["per_target_errors_path"], per_target_errors)
    _write_json(paths["feature_names_path"], feature_names)
    _write_json(paths["selected_features_path"], selected_features)
    _write_json(paths["model_path"], model)
    for split, rows in prediction_rows_by_split.items():
        write_prediction_csv(paths[f"predictions_{split}_path"], rows, TARGET_COLUMNS)

    return {
        "output_dir": str(output_path),
        "config_path": str(paths["config_path"]),
        "metrics_path": str(paths["metrics_path"]),
        "per_target_errors_path": str(paths["per_target_errors_path"]),
        "feature_names_path": str(paths["feature_names_path"]),
        "selected_features_path": str(paths["selected_features_path"]),
        "model_path": str(paths["model_path"]),
        "prediction_paths": {
            split: str(paths[f"predictions_{split}_path"])
            for split in ("train", "val", "test")
        },
        "metrics": metrics,
    }


def train_feature_selected_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    feature_count_grid: list[int | str],
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> dict[str, Any]:
    if train_features.shape[0] < 2:
        raise ValueError("Need at least two training rows for feature-selected ridge.")
    effective_counts = effective_feature_counts(feature_count_grid, len(feature_names))

    models_by_target: dict[str, dict[str, Any]] = {}
    selected_feature_counts: dict[str, int] = {}
    validation_mae_by_count: dict[str, dict[str, float]] = {}
    selected_features: dict[str, list[str]] = {}
    selected_feature_indices: dict[str, list[int]] = {}

    for target_index, target in enumerate(target_columns):
        ranking = rank_features_by_train_correlation(train_features, train_targets[:, target_index])
        target_results: dict[str, float] = {}
        best_count = effective_counts[0]
        best_mae: float | None = None
        best_model: dict[str, Any] | None = None

        for count in effective_counts:
            indices = ranking[:count]
            target_model = fit_single_target_ridge(
                train_features[:, indices],
                train_targets[:, target_index],
                ridge_alpha,
            )
            predictions = predict_single_target_ridge(target_model, val_features[:, indices])
            mae = float(np.abs(predictions - val_targets[:, target_index]).mean())
            target_results[str(count)] = mae
            if best_mae is None or mae < best_mae:
                best_count = count
                best_mae = mae
                best_model = target_model

        if best_model is None:
            raise ValueError(f"Could not select feature count for target {target}.")
        indices = ranking[:best_count]
        selected_feature_counts[target] = int(best_count)
        validation_mae_by_count[target] = target_results
        selected_feature_indices[target] = [int(index) for index in indices]
        selected_features[target] = [feature_names[index] for index in indices]
        models_by_target[target] = {
            **best_model,
            "target": target,
            "selected_feature_count": int(best_count),
            "selected_feature_indices": [int(index) for index in indices],
            "selected_feature_names": selected_features[target],
        }

    return {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "target_columns": target_columns,
        "feature_names": feature_names,
        "ridge_alpha": ridge_alpha,
        "feature_count_grid": [str(value) if value == "all" else int(value) for value in feature_count_grid],
        "selected_feature_counts": selected_feature_counts,
        "selected_feature_indices": selected_feature_indices,
        "selected_features": selected_features,
        "validation_mae_by_count": validation_mae_by_count,
        "target_models": models_by_target,
        "feature_ranking_method": "absolute_train_pearson_correlation",
    }


def normalize_feature_count_grid(feature_count_grid: list[int | str] | tuple[int | str, ...]) -> list[int | str]:
    if not feature_count_grid:
        raise ValueError("feature_count_grid must contain at least one entry.")
    normalized: list[int | str] = []
    for value in feature_count_grid:
        if isinstance(value, str):
            if value != "all":
                try:
                    parsed = int(value)
                except ValueError as error:
                    raise ValueError(f"Invalid feature count '{value}'. Use a positive integer or 'all'.") from error
                value = parsed
            else:
                normalized.append(value)
                continue
        if int(value) <= 0:
            raise ValueError("Feature counts must be positive integers or 'all'.")
        normalized.append(int(value))
    return normalized


def effective_feature_counts(feature_count_grid: list[int | str], total_feature_count: int) -> list[int]:
    counts = []
    for value in feature_count_grid:
        count = total_feature_count if value == "all" else int(value)
        if count <= 0:
            raise ValueError("Feature counts must be positive integers or 'all'.")
        counts.append(min(count, total_feature_count))
    return sorted(set(counts))


def rank_features_by_train_correlation(feature_matrix: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    correlations = np.asarray(
        [
            abs(pearson_correlation(feature_matrix[:, feature_index], target_values))
            for feature_index in range(feature_matrix.shape[1])
        ],
        dtype=np.float64,
    )
    return np.argsort(correlations)[::-1]


def pearson_correlation(first_values: np.ndarray, second_values: np.ndarray) -> float:
    if len(first_values) != len(second_values) or len(first_values) < 2:
        return 0.0
    first_centered = first_values - float(np.mean(first_values))
    second_centered = second_values - float(np.mean(second_values))
    first_std = float(np.sqrt(np.mean(first_centered**2)))
    second_std = float(np.sqrt(np.mean(second_centered**2)))
    if first_std <= 1e-9 or second_std <= 1e-9:
        return 0.0
    covariance = float(np.mean(first_centered * second_centered))
    return covariance / (first_std * second_std)


def fit_single_target_ridge(feature_matrix: np.ndarray, target_values: np.ndarray, ridge_alpha: float) -> dict[str, Any]:
    feature_means = feature_matrix.mean(axis=0)
    feature_stds = feature_matrix.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    standardized = (feature_matrix - feature_means) / feature_stds
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target_values)
    return {
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
    }


def predict_single_target_ridge(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    feature_means = np.asarray(model["feature_means"], dtype=np.float64)
    feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (feature_matrix - feature_means) / feature_stds
    return standardized @ coefficients + float(model["intercept"])


def predict_feature_selected_ridge(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    predictions = []
    for target in model["target_columns"]:
        target_model = model["target_models"][target]
        indices = np.asarray(target_model["selected_feature_indices"], dtype=np.int64)
        predictions.append(predict_single_target_ridge(target_model, feature_matrix[:, indices]))
    return np.column_stack(predictions)


def build_feature_selected_metrics(
    predictions_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    samples_by_split: dict[str, list[dict[str, Any]]],
    target_columns: list[str],
    feature_count: int,
    selected_feature_counts: dict[str, int],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "target_columns": target_columns,
        "feature_count": feature_count,
        "selected_feature_counts": selected_feature_counts,
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


def build_selected_features_payload(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_ranking_method": model["feature_ranking_method"],
        "feature_count_grid": model["feature_count_grid"],
        "selected_feature_counts": model["selected_feature_counts"],
        "selected_features": model["selected_features"],
        "validation_mae_by_count": model["validation_mae_by_count"],
    }


def build_config(
    dataset_root: Path,
    target_columns: list[str],
    feature_count: int,
    feature_count_grid: list[int | str],
    ridge_alpha: float,
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
            "regression_method": "per_target_feature_selected_ridge_regression",
            "hyperparameters": {
                "ridge_alpha": ridge_alpha,
                "feature_count_grid": [str(value) if value == "all" else int(value) for value in feature_count_grid],
                "feature_ranking_method": "absolute_train_pearson_correlation",
                "selection_split": "val",
            },
        },
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run target-specific feature-selected ridge over image silhouette features.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    parser.add_argument("--feature-count-grid", nargs="+", default=list(DEFAULT_FEATURE_COUNT_GRID))
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    args = parser.parse_args(argv)

    result = run_feature_selected_ridge_experiment(
        args.dataset,
        args.output,
        feature_count_grid=args.feature_count_grid,
        ridge_alpha=args.ridge_alpha,
    )
    print(format_metrics_report(result))
    print(f"Selected features artifact: {result['selected_features_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
