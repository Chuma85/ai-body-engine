from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.features.image_silhouette_features import (
    extract_front_side_features,
    feature_vector,
    get_feature_names,
)
from training.train_baseline_measurements import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    TARGET_COLUMNS,
    _mean,
    _require_enough_samples,
    format_metrics_report,
)

MODEL_TYPE = "image_silhouette_ridge_regressor"


def train_image_feature_baseline(
    dataset_root: str | Path,
    output_dir: str | Path,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    train_dataset = SyntheticBodyDataset(dataset_root, split="train")
    val_dataset = SyntheticBodyDataset(dataset_root, split="val")
    test_dataset = SyntheticBodyDataset(dataset_root, split="test")
    _require_enough_samples(train_dataset, val_dataset, test_dataset)

    train_samples = list(train_dataset)
    val_samples = list(val_dataset)
    test_samples = list(test_dataset)
    feature_names = get_feature_names()

    train_features = extract_sample_feature_matrix(train_samples, feature_names)
    val_features = extract_sample_feature_matrix(val_samples, feature_names)
    test_features = extract_sample_feature_matrix(test_samples, feature_names)
    train_targets = _target_matrix(train_samples, TARGET_COLUMNS)

    model = train_ridge_regressor(train_features, train_targets, feature_names, TARGET_COLUMNS, ridge_alpha)
    metrics = {
        "model_type": MODEL_TYPE,
        "target_columns": TARGET_COLUMNS,
        "feature_names": feature_names,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "train": evaluate_feature_regressor(model, train_features, train_samples, TARGET_COLUMNS),
        "val": evaluate_feature_regressor(model, val_features, val_samples, TARGET_COLUMNS),
        "test": evaluate_feature_regressor(model, test_features, test_samples, TARGET_COLUMNS),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / MODEL_FILENAME
    metrics_path = output_path / METRICS_FILENAME
    _write_json(model_path, model)
    _write_json(metrics_path, metrics)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def extract_sample_feature_matrix(samples: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        rows.append(feature_vector(features, feature_names))
    return np.asarray(rows, dtype=np.float64)


def train_ridge_regressor(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    if feature_matrix.shape[0] < 2:
        raise ValueError("Need at least two training rows for image feature baseline.")

    feature_means = feature_matrix.mean(axis=0)
    feature_stds = feature_matrix.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    standardized = (feature_matrix - feature_means) / feature_stds
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target_matrix)

    return {
        "model_type": MODEL_TYPE,
        "feature_names": feature_names,
        "target_columns": target_columns,
        "ridge_alpha": ridge_alpha,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "intercepts": coefficients[0, :].tolist(),
        "coefficients": coefficients[1:, :].tolist(),
    }


def predict_feature_regressor(model: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    feature_means = np.asarray(model["feature_means"], dtype=np.float64)
    feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
    intercepts = np.asarray(model["intercepts"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (feature_matrix - feature_means) / feature_stds
    return standardized @ coefficients + intercepts


def evaluate_feature_regressor(
    model: dict[str, Any],
    feature_matrix: np.ndarray,
    samples: list[dict[str, Any]],
    target_columns: list[str],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot evaluate image feature baseline with zero samples.")

    predictions = predict_feature_regressor(model, feature_matrix)
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


def _target_matrix(samples: list[dict[str, Any]], target_columns: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        try:
            rows.append([float(sample["measurements"][target]) for target in target_columns])
        except KeyError as error:
            raise ValueError(f"Sample {sample.get('sample_id', '')} is missing target {error}.") from error
    return np.asarray(rows, dtype=np.float64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight image-feature measurement baseline.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Directory for model.json and metrics.json artifacts.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)

    result = train_image_feature_baseline(args.dataset, args.output, ridge_alpha=args.ridge_alpha)
    print(format_metrics_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
