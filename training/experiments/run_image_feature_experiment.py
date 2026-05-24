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
from training.features.image_silhouette_features import get_feature_names
from training.train_baseline_measurements import (
    MODEL_FILENAME,
    TARGET_COLUMNS,
    _mean,
    _require_enough_samples,
    format_metrics_report,
)
from training.train_image_feature_baseline import (
    MODEL_TYPE,
    _target_matrix,
    evaluate_feature_regressor,
    extract_sample_feature_matrix,
    predict_feature_regressor,
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
FEATURE_EXTRACTOR_NAME = "image_silhouette_features"
FEATURE_EXTRACTOR_VERSION = "phase_2p"
EXPERIMENT_RUNNER_VERSION = "phase_2s"


def run_image_feature_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")

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

    model = train_ridge_regressor(
        features_by_split["train"],
        targets_by_split["train"],
        feature_names,
        TARGET_COLUMNS,
        ridge_alpha,
    )
    predictions_by_split = {
        split: predict_feature_regressor(model, feature_matrix)
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
    config = build_config(dataset_path, TARGET_COLUMNS, ridge_alpha)

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
        "model_type": MODEL_TYPE,
        "target_columns": target_columns,
        "feature_count": feature_count,
        "sample_counts": {
            split: len(samples)
            for split, samples in samples_by_split.items()
        },
    }
    for split in ("train", "val", "test"):
        metrics[split] = evaluate_feature_regressor(
            model,
            features_by_split[split],
            samples_by_split[split],
            target_columns,
        )
    return metrics


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


def build_config(dataset_root: Path, target_columns: list[str], ridge_alpha: float) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "target_columns": target_columns,
        "feature_extractor": {
            "name": FEATURE_EXTRACTOR_NAME,
            "version": FEATURE_EXTRACTOR_VERSION,
        },
        "model": {
            "type": MODEL_TYPE,
            "regression_method": "ridge_regression",
            "ridge_alpha": ridge_alpha,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a repeatable image-feature measurement experiment.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args(argv)

    result = run_image_feature_experiment(args.dataset, args.output, ridge_alpha=args.ridge_alpha)
    print(format_metrics_report(result))
    print(f"Config artifact: {result['config_path']}")
    print(f"Feature names artifact: {result['feature_names_path']}")
    print(f"Per-target errors artifact: {result['per_target_errors_path']}")
    for split, path in result["prediction_paths"].items():
        print(f"{split} predictions: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
