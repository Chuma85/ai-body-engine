from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from training.datasets.synthetic_body_dataset import MEASUREMENT_COLUMNS, SyntheticBodyDataset

TARGET_COLUMNS = [*MEASUREMENT_COLUMNS]
MODEL_FILENAME = "model.json"
METRICS_FILENAME = "metrics.json"


def train_baseline(dataset_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    train_dataset = SyntheticBodyDataset(dataset_root, split="train")
    val_dataset = SyntheticBodyDataset(dataset_root, split="val")
    test_dataset = SyntheticBodyDataset(dataset_root, split="test")

    _require_enough_samples(train_dataset, val_dataset, test_dataset)

    train_samples = list(train_dataset)
    val_samples = list(val_dataset)
    test_samples = list(test_dataset)

    model = train_body_shape_mean_regressor(train_samples, TARGET_COLUMNS)
    metrics = {
        "model_type": model["model_type"],
        "target_columns": TARGET_COLUMNS,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "train": evaluate_regressor(model, train_samples, TARGET_COLUMNS),
        "val": evaluate_regressor(model, val_samples, TARGET_COLUMNS),
        "test": evaluate_regressor(model, test_samples, TARGET_COLUMNS),
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


def train_body_shape_mean_regressor(samples: list[dict[str, Any]], target_columns: list[str]) -> dict[str, Any]:
    global_sums = {target: 0.0 for target in target_columns}
    global_counts = {target: 0 for target in target_columns}
    shape_sums: dict[str, dict[str, float]] = {}
    shape_counts: dict[str, dict[str, int]] = {}

    for sample in samples:
        body_shape = sample.get("body_shape") or "__missing__"
        shape_sums.setdefault(body_shape, {target: 0.0 for target in target_columns})
        shape_counts.setdefault(body_shape, {target: 0 for target in target_columns})

        for target in target_columns:
            value = sample["measurements"].get(target)
            if value is None:
                continue
            global_sums[target] += value
            global_counts[target] += 1
            shape_sums[body_shape][target] += value
            shape_counts[body_shape][target] += 1

    global_means = {
        target: _safe_mean(global_sums[target], global_counts[target], target)
        for target in target_columns
    }
    body_shape_means = {
        body_shape: {
            target: (
                shape_sums[body_shape][target] / shape_counts[body_shape][target]
                if shape_counts[body_shape][target] > 0
                else global_means[target]
            )
            for target in target_columns
        }
        for body_shape in sorted(shape_sums)
    }

    return {
        "model_type": "body_shape_mean_regressor",
        "target_columns": target_columns,
        "global_means": global_means,
        "body_shape_means": body_shape_means,
    }


def predict_measurements(model: dict[str, Any], sample: dict[str, Any], target_columns: list[str]) -> dict[str, float]:
    body_shape = sample.get("body_shape") or "__missing__"
    shape_means = model["body_shape_means"].get(body_shape, model["global_means"])
    return {target: float(shape_means.get(target, model["global_means"][target])) for target in target_columns}


def evaluate_regressor(model: dict[str, Any], samples: list[dict[str, Any]], target_columns: list[str]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot evaluate baseline with zero samples.")

    absolute_errors = {target: [] for target in target_columns}
    for sample in samples:
        prediction = predict_measurements(model, sample, target_columns)
        for target in target_columns:
            actual = sample["measurements"].get(target)
            if actual is None:
                continue
            absolute_errors[target].append(abs(prediction[target] - actual))

    mae_by_target = {
        target: _mean(errors) for target, errors in absolute_errors.items() if errors
    }
    if not mae_by_target:
        raise ValueError("Cannot evaluate baseline because no target values were present.")

    return {
        "overall_mae": _mean(list(mae_by_target.values())),
        "mae_by_target": mae_by_target,
    }


def format_metrics_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        f"Model: {metrics['model_type']}",
        f"Model artifact: {result['model_path']}",
        f"Metrics artifact: {result['metrics_path']}",
        (
            "Samples: "
            f"train={metrics['sample_counts']['train']} "
            f"val={metrics['sample_counts']['val']} "
            f"test={metrics['sample_counts']['test']}"
        ),
    ]
    for split in ("train", "val", "test"):
        lines.append(f"{split} overall MAE: {metrics[split]['overall_mae']:.4f}")
    return "\n".join(lines)


def _require_enough_samples(
    train_dataset: SyntheticBodyDataset,
    val_dataset: SyntheticBodyDataset,
    test_dataset: SyntheticBodyDataset,
) -> None:
    if len(train_dataset) < 2 or len(val_dataset) < 1 or len(test_dataset) < 1:
        raise ValueError(
            "Not enough samples for baseline training. "
            f"Need at least train=2, val=1, test=1; got "
            f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}."
        )


def _safe_mean(total: float, count: int, target: str) -> float:
    if count == 0:
        raise ValueError(f"Training samples are missing target values for {target}.")
    return total / count


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate mean of empty values.")
    return sum(values) / len(values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight synthetic measurement baseline.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Directory for model.json and metrics.json artifacts.")
    args = parser.parse_args(argv)

    result = train_baseline(args.dataset, args.output)
    print(format_metrics_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
