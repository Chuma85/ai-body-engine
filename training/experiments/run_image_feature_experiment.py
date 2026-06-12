from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import (
    MODEL_FILENAME,
    TARGET_COLUMNS,
    _mean,
    _require_enough_samples,
    format_metrics_report,
)
from training.train_image_feature_baseline import (
    _target_matrix,
    extract_sample_feature_matrix,
    train_ridge_regressor,
)

CONFIG_FILENAME = "config.json"
FEATURE_NAMES_FILENAME = "feature_names.json"
PER_TARGET_ERRORS_FILENAME = "per_target_errors.json"
PREDICTION_FILENAMES = {
    "train": "predictions_train.csv",
    "val": "predictions_val.csv",
    "test": "predictions_test.csv",
}
METRICS_FILENAME = "metrics.json"
MODEL_TYPE = "image_silhouette_ridge_regressor"
FEATURE_EXTRACTOR_NAME = "image_silhouette_features"
EXPERIMENT_RUNNER_VERSION = "phase_2t"
SUPPORTED_MODEL_TYPES = ("mean", "ridge", "knn")
DEFAULT_MODEL_TYPE = "ridge"
DEFAULT_KNN_K = 5


def run_image_feature_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    model_type: str = DEFAULT_MODEL_TYPE,
    ridge_alpha: float = 10.0,
    knn_k: int = DEFAULT_KNN_K,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")
    _validate_model_type(model_type)

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

    model = train_image_feature_model(
        model_type,
        features_by_split["train"],
        targets_by_split["train"],
        feature_names,
        TARGET_COLUMNS,
        ridge_alpha=ridge_alpha,
        knn_k=knn_k,
    )
    predictions_by_split = {
        split: predict_image_feature_model(model, feature_matrix)
        for split, feature_matrix in features_by_split.items()
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

    metrics = build_metrics(
        model,
        features_by_split,
        samples_by_split,
        TARGET_COLUMNS,
        feature_count=len(feature_names),
    )
    per_target_errors = {
        split: calculate_per_target_errors(rows, TARGET_COLUMNS)
        for split, rows in prediction_rows_by_split.items()
    }
    config = build_config(dataset_path, TARGET_COLUMNS, len(feature_names), model, model_type, ridge_alpha, knn_k)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "config_path": output_path / CONFIG_FILENAME,
        "metrics_path": output_path / METRICS_FILENAME,
        "per_target_errors_path": output_path / PER_TARGET_ERRORS_FILENAME,
        "feature_names_path": output_path / FEATURE_NAMES_FILENAME,
        "model_path": output_path / MODEL_FILENAME,
    }
    for split, filename in PREDICTION_FILENAMES.items():
        paths[f"predictions_{split}_path"] = output_path / filename

    _write_json(paths["config_path"], config)
    _write_json(paths["metrics_path"], metrics)
    _write_json(paths["per_target_errors_path"], per_target_errors)
    _write_json(paths["feature_names_path"], feature_names)
    _write_json(paths["model_path"], model)
    for split, rows in prediction_rows_by_split.items():
        write_prediction_csv(paths[f"predictions_{split}_path"], rows, TARGET_COLUMNS)

    return {
        "output_dir": str(output_path),
        "config_path": str(paths["config_path"]),
        "metrics_path": str(paths["metrics_path"]),
        "per_target_errors_path": str(paths["per_target_errors_path"]),
        "feature_names_path": str(paths["feature_names_path"]),
        "model_path": str(paths["model_path"]),
        "prediction_paths": {
            split: str(paths[f"predictions_{split}_path"])
            for split in ("train", "val", "test")
        },
        "metrics": metrics,
    }


def build_metrics(
    model: dict[str, Any],
    features_by_split: dict[str, np.ndarray],
    samples_by_split: dict[str, list[dict[str, Any]]],
    target_columns: list[str],
    feature_count: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_type": model["model_type"],
        "model_family": model["model_family"],
        "target_columns": target_columns,
        "feature_count": feature_count,
        "sample_counts": {
            split: len(samples)
            for split, samples in samples_by_split.items()
        },
    }
    for split in ("train", "val", "test"):
        metrics[split] = evaluate_image_feature_model(
            model,
            features_by_split[split],
            samples_by_split[split],
            target_columns,
        )
    return metrics


def train_image_feature_model(
    model_type: str,
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    ridge_alpha: float = 10.0,
    knn_k: int = DEFAULT_KNN_K,
) -> dict[str, Any]:
    _validate_model_type(model_type)
    if model_type == "mean":
        return train_mean_regressor(target_matrix, target_columns)
    if model_type == "ridge":
        model = train_ridge_regressor(feature_matrix, target_matrix, feature_names, target_columns, ridge_alpha)
        model["model_family"] = "ridge"
        model["hyperparameters"] = {"ridge_alpha": ridge_alpha}
        return model
    if model_type == "knn":
        return train_knn_regressor(feature_matrix, target_matrix, feature_names, target_columns, knn_k)
    raise AssertionError(f"Unhandled model type: {model_type}")


def train_mean_regressor(target_matrix: np.ndarray, target_columns: list[str]) -> dict[str, Any]:
    if target_matrix.shape[0] < 1:
        raise ValueError("Need at least one training row for mean image-feature baseline.")
    target_means = target_matrix.mean(axis=0)
    return {
        "model_type": "image_feature_mean_regressor",
        "model_family": "mean",
        "target_columns": target_columns,
        "hyperparameters": {},
        "target_means": target_means.tolist(),
    }


def train_knn_regressor(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    knn_k: int,
) -> dict[str, Any]:
    if feature_matrix.shape[0] < 1:
        raise ValueError("Need at least one training row for KNN image-feature baseline.")
    if knn_k < 1:
        raise ValueError("knn_k must be at least 1.")

    feature_means = feature_matrix.mean(axis=0)
    feature_stds = feature_matrix.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    standardized = (feature_matrix - feature_means) / feature_stds
    effective_k = min(knn_k, feature_matrix.shape[0])
    return {
        "model_type": "image_feature_knn_regressor",
        "model_family": "knn",
        "feature_names": feature_names,
        "target_columns": target_columns,
        "hyperparameters": {"k": effective_k},
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "train_features": standardized.tolist(),
        "train_targets": target_matrix.tolist(),
    }


def predict_image_feature_model(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    model_family = model["model_family"]
    if model_family == "mean":
        target_means = np.asarray(model["target_means"], dtype=np.float64)
        return np.tile(target_means, (feature_matrix.shape[0], 1))
    if model_family == "ridge":
        feature_means = np.asarray(model["feature_means"], dtype=np.float64)
        feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
        intercepts = [float(value) for value in model["intercepts"]]
        coefficients = [[float(value) for value in row] for row in model["coefficients"]]
        standardized = (feature_matrix - feature_means) / feature_stds
        rows: list[list[float]] = []
        for feature_row in standardized.tolist():
            prediction_row = []
            for target_index, intercept in enumerate(intercepts):
                value = intercept
                for feature_index, feature_value in enumerate(feature_row):
                    value += float(feature_value) * coefficients[feature_index][target_index]
                prediction_row.append(value)
            rows.append(prediction_row)
        return np.asarray(rows, dtype=np.float64)
    if model_family == "knn":
        return predict_knn_regressor(model, feature_matrix)
    raise ValueError(f"Unknown trained model family '{model_family}'.")


def predict_knn_regressor(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    feature_means = np.asarray(model["feature_means"], dtype=np.float64)
    feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
    train_features = np.asarray(model["train_features"], dtype=np.float64)
    train_targets = np.asarray(model["train_targets"], dtype=np.float64)
    k = int(model["hyperparameters"]["k"])
    standardized = (feature_matrix - feature_means) / feature_stds
    predictions = []
    train_feature_rows = train_features.tolist()
    for row in standardized.tolist():
        distances = np.asarray(
            [
                sum((float(a) - float(b)) ** 2 for a, b in zip(train_row, row)) ** 0.5
                for train_row in train_feature_rows
            ],
            dtype=np.float64,
        )
        nearest_indices = np.argsort(distances)[:k]
        predictions.append(train_targets[nearest_indices].mean(axis=0))
    return np.asarray(predictions, dtype=np.float64)


def evaluate_image_feature_model(
    model: dict[str, Any],
    feature_matrix: np.ndarray,
    samples: list[dict[str, Any]],
    target_columns: list[str],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot evaluate image feature experiment with zero samples.")

    predictions = predict_image_feature_model(model, feature_matrix)
    targets = _target_matrix(samples, target_columns)
    absolute_errors = np.abs(predictions - targets)
    mae_by_target = {
        target: float(absolute_errors[:, index].mean())
        for index, target in enumerate(target_columns)
    }
    return {
        "overall_mae": _mean(list(mae_by_target.values())),
        "mae_by_target": mae_by_target,
    }


def build_prediction_rows(
    samples: list[dict[str, Any]],
    split: str,
    target_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    target_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, sample in enumerate(samples):
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "split": split,
        }
        for target_index, target in enumerate(target_columns):
            true_value = float(target_matrix[row_index, target_index])
            prediction = float(prediction_matrix[row_index, target_index])
            row[f"true_{target}"] = true_value
            row[f"pred_{target}"] = prediction
            row[f"abs_error_{target}"] = abs(prediction - true_value)
        rows.append(row)
    return rows


def calculate_per_target_errors(
    prediction_rows: list[dict[str, Any]],
    target_columns: list[str],
) -> dict[str, dict[str, float | int]]:
    if not prediction_rows:
        raise ValueError("Cannot calculate per-target errors with zero prediction rows.")

    errors: dict[str, dict[str, float | int]] = {}
    for target in target_columns:
        values = [float(row[f"abs_error_{target}"]) for row in prediction_rows]
        errors[target] = {
            "count": len(values),
            "mae": _mean(values),
            "max_abs_error": max(values),
        }
    return errors


def build_config(
    dataset_root: Path,
    target_columns: list[str],
    feature_count: int,
    model: dict[str, Any],
    model_type: str,
    ridge_alpha: float,
    knn_k: int,
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
            "type": model_type,
            "artifact_type": model["model_type"],
            "regression_method": _regression_method(model_type),
            "hyperparameters": _model_hyperparameters(model_type, ridge_alpha, knn_k),
        },
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], target_columns: list[str]) -> None:
    fieldnames = prediction_fieldnames(target_columns)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fieldnames})


def prediction_fieldnames(target_columns: list[str]) -> list[str]:
    fieldnames = ["sample_id", "split"]
    for target in target_columns:
        fieldnames.extend([f"true_{target}", f"pred_{target}", f"abs_error_{target}"])
    return fieldnames


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _validate_model_type(model_type: str) -> None:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type '{model_type}'. Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}."
        )


def _regression_method(model_type: str) -> str:
    return {
        "mean": "target_mean",
        "ridge": "ridge_regression",
        "knn": "k_nearest_neighbors",
    }[model_type]


def _model_hyperparameters(model_type: str, ridge_alpha: float, knn_k: int) -> dict[str, Any]:
    if model_type == "ridge":
        return {"ridge_alpha": ridge_alpha}
    if model_type == "knn":
        return {"k": knn_k}
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a repeatable image-feature measurement experiment.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL_TYPE, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    args = parser.parse_args(argv)

    result = run_image_feature_experiment(
        args.dataset,
        args.output,
        model_type=args.model,
        ridge_alpha=args.ridge_alpha,
        knn_k=args.knn_k,
    )
    print(format_metrics_report(result))
    print(f"Config artifact: {result['config_path']}")
    print(f"Feature names artifact: {result['feature_names_path']}")
    print(f"Per-target errors artifact: {result['per_target_errors_path']}")
    for split, path in result["prediction_paths"].items():
        print(f"{split} predictions: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
