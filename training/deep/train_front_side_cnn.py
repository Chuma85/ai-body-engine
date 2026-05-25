from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.deep.dependencies import DeepLearningDependencyError, import_torch
from training.deep.models.simple_front_side_cnn import SimpleFrontSideCNN
from training.deep.synthetic_body_image_dataset import TARGET_COLUMNS, SyntheticBodyImageDataset

CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"
MODEL_FILENAME = "model.pt"
PER_TARGET_ERRORS_FILENAME = "per_target_errors.json"
TARGET_NAMES_FILENAME = "target_names.json"
MODEL_TYPE = "simple_front_side_cnn"
MODEL_FAMILY = "front_side_cnn"
EXPERIMENT_RUNNER_VERSION = "phase_3c"
PREDICTION_FILENAMES = {
    "train": "predictions_train.csv",
    "val": "predictions_val.csv",
    "test": "predictions_test.csv",
}


def train_front_side_cnn(
    dataset_root: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    limit_samples: int | None = None,
    image_size: int = 128,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    save_predictions: bool = True,
    patience: int = 3,
    weight_decay: float = 0.0,
    target_normalization_enabled: bool = True,
) -> dict[str, Any]:
    torch = import_torch()
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if patience < 0:
        raise ValueError("patience must be a non-negative integer.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    set_training_seed(seed)

    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")

    train_dataset = SyntheticBodyImageDataset(dataset_path, split="train", image_size=image_size, as_tensors=True, limit_samples=limit_samples)
    val_dataset = SyntheticBodyImageDataset(dataset_path, split="val", image_size=image_size, as_tensors=True, limit_samples=limit_samples)
    test_dataset = SyntheticBodyImageDataset(dataset_path, split="test", image_size=image_size, as_tensors=True, limit_samples=limit_samples)
    if len(train_dataset) < 2 or len(val_dataset) < 1 or len(test_dataset) < 1:
        raise ValueError("Need at least two train samples, one validation sample, and one test sample for deep training.")

    torch_device = resolve_device(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    target_mean, target_std = target_normalization(train_dataset, enabled=target_normalization_enabled)

    model = SimpleFrontSideCNN(target_count=len(TARGET_COLUMNS)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    epoch_metrics = []
    best_state_dict: dict[str, Any] | None = None
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, target_mean, target_std, torch_device)
        train_eval = evaluate_predictions(model, train_loader, target_mean, target_std, torch_device)
        val_eval = evaluate_predictions(model, val_loader, target_mean, target_std, torch_device)
        val_mae = float(val_eval["metrics"]["overall_mae"])
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_overall_mae": train_eval["metrics"]["overall_mae"],
                "val_overall_mae": val_mae,
            }
        )
        if is_improved(val_mae, best_val_mae):
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = clone_model_state_dict(model)
        else:
            epochs_without_improvement += 1
            if should_stop_early(epochs_without_improvement, patience):
                early_stopping_triggered = True
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    evaluations = {
        "train": evaluate_predictions(model, train_loader, target_mean, target_std, torch_device),
        "val": evaluate_predictions(model, val_loader, target_mean, target_std, torch_device),
        "test": evaluate_predictions(model, test_loader, target_mean, target_std, torch_device),
    }
    metrics = {
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "target_columns": list(TARGET_COLUMNS),
        "target_names": list(TARGET_COLUMNS),
        "sample_counts": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        "epochs": epochs,
        "epochs_completed": len(epoch_metrics),
        "best_epoch": best_epoch,
        "best_val_overall_mae": best_val_mae,
        "early_stopping_triggered": early_stopping_triggered,
        "target_normalization_enabled": target_normalization_enabled,
        "image_size": image_size,
        "limit_samples": limit_samples,
        "device": str(torch_device),
        "train": evaluations["train"]["metrics"],
        "val": evaluations["val"]["metrics"],
        "test": evaluations["test"]["metrics"],
        "epoch_metrics": epoch_metrics,
    }
    per_target_errors = {
        split: calculate_per_target_errors(evaluation["prediction_rows"])
        for split, evaluation in evaluations.items()
    }
    config = build_config(
        dataset_path,
        epochs,
        limit_samples,
        image_size,
        batch_size,
        learning_rate,
        seed,
        str(torch_device),
        save_predictions,
        patience,
        weight_decay,
        target_normalization_enabled,
        target_mean,
        target_std,
        best_epoch,
        len(epoch_metrics),
        early_stopping_triggered,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / CONFIG_FILENAME
    metrics_path = output_path / METRICS_FILENAME
    per_target_errors_path = output_path / PER_TARGET_ERRORS_FILENAME
    target_names_path = output_path / TARGET_NAMES_FILENAME
    model_path = output_path / MODEL_FILENAME
    _write_json(config_path, config)
    _write_json(metrics_path, metrics)
    _write_json(per_target_errors_path, per_target_errors)
    _write_json(target_names_path, list(TARGET_COLUMNS))
    prediction_paths = {}
    if save_predictions:
        for split, filename in PREDICTION_FILENAMES.items():
            prediction_path = output_path / filename
            write_prediction_csv(prediction_path, evaluations[split]["prediction_rows"])
            prediction_paths[split] = str(prediction_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_columns": list(TARGET_COLUMNS),
            "target_mean": target_mean.cpu().numpy().tolist(),
            "target_std": target_std.cpu().numpy().tolist(),
            "target_normalization_enabled": target_normalization_enabled,
            "best_epoch": best_epoch,
            "config": config,
        },
        model_path,
    )

    return {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "per_target_errors_path": str(per_target_errors_path),
        "target_names_path": str(target_names_path),
        "model_path": str(model_path),
        "prediction_paths": prediction_paths,
        "metrics": metrics,
    }


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = import_torch()
    torch.manual_seed(seed)


def is_improved(current_value: float, best_value: float, min_delta: float = 1e-6) -> bool:
    return current_value < best_value - min_delta


def should_stop_early(epochs_without_improvement: int, patience: int) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def best_epoch_from_history(epoch_metrics: list[dict[str, Any]]) -> int:
    if not epoch_metrics:
        return 0
    return int(min(epoch_metrics, key=lambda row: float(row["val_overall_mae"]))["epoch"])


def clone_model_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def resolve_device(device: str) -> Any:
    torch = import_torch()
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of: cpu, cuda, auto.")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def target_normalization(dataset: SyntheticBodyImageDataset, enabled: bool = True) -> tuple[Any, Any]:
    torch = import_torch()
    targets = torch.stack([dataset[index]["targets"] for index in range(len(dataset))])
    if not enabled:
        target_mean = torch.zeros(targets.shape[1], dtype=targets.dtype)
        target_std = torch.ones(targets.shape[1], dtype=targets.dtype)
        return target_mean, target_std
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0)
    target_std = torch.where(target_std < 1e-6, torch.ones_like(target_std), target_std)
    return target_mean, target_std


def train_one_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any, target_mean: Any, target_std: Any, device: Any) -> float:
    model.train()
    losses = []
    for batch in loader:
        front_images = batch["front_image"].to(device)
        side_images = batch["side_image"].to(device)
        targets = batch["targets"].to(device)
        normalized_targets = normalize_targets(targets, target_mean.to(device), target_std.to(device))

        optimizer.zero_grad()
        predictions = model(front_images, side_images)
        loss = loss_fn(predictions, normalized_targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_predictions(model: Any, loader: Any, target_mean: Any, target_std: Any, device: Any) -> dict[str, Any]:
    torch = import_torch()
    model.eval()
    prediction_rows = []
    target_rows = []
    sample_ids: list[str] = []
    splits: list[str] = []
    with torch.no_grad():
        for batch in loader:
            front_images = batch["front_image"].to(device)
            side_images = batch["side_image"].to(device)
            targets = batch["targets"].to(device)
            normalized_predictions = model(front_images, side_images)
            predictions = inverse_transform_targets(normalized_predictions, target_mean.to(device), target_std.to(device))
            prediction_rows.append(predictions.cpu())
            target_rows.append(targets.cpu())
            sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
            splits.extend(str(split) for split in batch["split"])

    predictions = torch.cat(prediction_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    absolute_errors = torch.abs(predictions - targets).numpy()
    prediction_csv_rows = build_prediction_rows(sample_ids, splits, targets.numpy(), predictions.numpy())
    metrics = calculate_metrics_from_errors(absolute_errors)
    return {
        "metrics": metrics,
        "prediction_rows": prediction_csv_rows,
    }


def calculate_metrics_from_errors(absolute_errors: np.ndarray) -> dict[str, Any]:
    mae_by_target = {target: float(absolute_errors[:, index].mean()) for index, target in enumerate(TARGET_COLUMNS)}
    return {
        "overall_mae": float(np.mean(list(mae_by_target.values()))),
        "mae_by_target": mae_by_target,
    }


def normalize_targets(targets: Any, target_mean: Any, target_std: Any) -> Any:
    return (targets - target_mean) / target_std


def inverse_transform_targets(normalized_targets: Any, target_mean: Any, target_std: Any) -> Any:
    return normalized_targets * target_std + target_mean


def build_prediction_rows(
    sample_ids: list[str],
    splits: list[str],
    target_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": splits[row_index],
        }
        for target_index, target in enumerate(TARGET_COLUMNS):
            true_value = float(target_matrix[row_index, target_index])
            prediction = float(prediction_matrix[row_index, target_index])
            row[f"true_{target}"] = true_value
            row[f"pred_{target}"] = prediction
            row[f"abs_error_{target}"] = abs(prediction - true_value)
        rows.append(row)
    return rows


def calculate_per_target_errors(prediction_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    if not prediction_rows:
        raise ValueError("Cannot calculate per-target errors with zero prediction rows.")
    errors: dict[str, dict[str, float | int]] = {}
    for target in TARGET_COLUMNS:
        values = [float(row[f"abs_error_{target}"]) for row in prediction_rows]
        errors[target] = {
            "count": len(values),
            "mae": float(np.mean(values)),
            "max_abs_error": max(values),
        }
    return errors


def write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = prediction_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fieldnames})


def prediction_fieldnames() -> list[str]:
    fieldnames = ["sample_id", "split"]
    for target in TARGET_COLUMNS:
        fieldnames.extend([f"true_{target}", f"pred_{target}", f"abs_error_{target}"])
    return fieldnames


def build_config(
    dataset_root: Path,
    epochs: int,
    limit_samples: int | None,
    image_size: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    save_predictions: bool,
    patience: int,
    weight_decay: float,
    target_normalization_enabled: bool,
    target_mean: Any,
    target_std: Any,
    best_epoch: int,
    epochs_completed: int,
    early_stopping_triggered: bool,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "target_columns": list(TARGET_COLUMNS),
        "target_names": list(TARGET_COLUMNS),
        "image_size": image_size,
        "epochs": epochs,
        "limit_samples": limit_samples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "seed": seed,
        "device": device,
        "save_predictions": save_predictions,
        "target_normalization": {
            "enabled": target_normalization_enabled,
            "target_mean": target_mean.cpu().numpy().tolist(),
            "target_std": target_std.cpu().numpy().tolist(),
        },
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "early_stopping_triggered": early_stopping_triggered,
        "model": {
            "type": MODEL_FAMILY,
            "artifact_type": MODEL_TYPE,
            "regression_method": "front_side_cnn_regression",
            "hyperparameters": {
                "image_size": image_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "patience": patience,
            },
        },
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_training_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join(
        [
            f"Model: {metrics['model_type']}",
            f"Samples: train={metrics['sample_counts']['train']} val={metrics['sample_counts']['val']}",
            f"test samples: {metrics['sample_counts']['test']}",
            f"best epoch: {metrics['best_epoch']}",
            f"train overall MAE: {metrics['train']['overall_mae']:.4f}",
            f"val overall MAE: {metrics['val']['overall_mae']:.4f}",
            f"test overall MAE: {metrics['test']['overall_mae']:.4f}",
            f"Config artifact: {result['config_path']}",
            f"Metrics artifact: {result['metrics_path']}",
            f"Per-target errors artifact: {result['per_target_errors_path']}",
            f"Model artifact: {result['model_path']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny CPU smoke training pass for a front/side CNN.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Output directory for deep smoke artifacts.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--disable-target-normalization", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = train_front_side_cnn(
            args.dataset,
            args.output,
            epochs=args.epochs,
            limit_samples=args.limit_samples,
            image_size=args.image_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            save_predictions=args.save_predictions,
            patience=args.patience,
            weight_decay=args.weight_decay,
            target_normalization_enabled=not args.disable_target_normalization,
        )
    except DeepLearningDependencyError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(format_training_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
