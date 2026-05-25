from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
MODEL_TYPE = "simple_front_side_cnn"
EXPERIMENT_RUNNER_VERSION = "phase_3a"


def train_front_side_cnn(
    dataset_root: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    limit_samples: int | None = None,
    image_size: int = 128,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    torch = import_torch()
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")

    train_dataset = SyntheticBodyImageDataset(dataset_path, split="train", image_size=image_size, as_tensors=True, limit_samples=limit_samples)
    val_dataset = SyntheticBodyImageDataset(dataset_path, split="val", image_size=image_size, as_tensors=True, limit_samples=limit_samples)
    if len(train_dataset) < 2 or len(val_dataset) < 1:
        raise ValueError("Need at least two train samples and one validation sample for deep smoke training.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_mean, target_std = target_normalization(train_dataset)

    device = torch.device("cpu")
    model = SimpleFrontSideCNN(target_count=len(TARGET_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, target_mean, target_std, device)
        train_mae = evaluate_mae(model, train_loader, target_mean, target_std, device)
        val_mae = evaluate_mae(model, val_loader, target_mean, target_std, device)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_overall_mae": train_mae["overall_mae"],
                "val_overall_mae": val_mae["overall_mae"],
            }
        )

    final_train = evaluate_mae(model, train_loader, target_mean, target_std, device)
    final_val = evaluate_mae(model, val_loader, target_mean, target_std, device)
    metrics = {
        "model_type": MODEL_TYPE,
        "target_columns": list(TARGET_COLUMNS),
        "sample_counts": {"train": len(train_dataset), "val": len(val_dataset)},
        "epochs": epochs,
        "train": final_train,
        "val": final_val,
        "epoch_metrics": epoch_metrics,
    }
    config = build_config(dataset_path, epochs, limit_samples, image_size, batch_size, learning_rate)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / CONFIG_FILENAME
    metrics_path = output_path / METRICS_FILENAME
    model_path = output_path / MODEL_FILENAME
    _write_json(config_path, config)
    _write_json(metrics_path, metrics)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_columns": list(TARGET_COLUMNS),
            "target_mean": target_mean.cpu().numpy().tolist(),
            "target_std": target_std.cpu().numpy().tolist(),
            "config": config,
        },
        model_path,
    )

    return {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "metrics": metrics,
    }


def target_normalization(dataset: SyntheticBodyImageDataset) -> tuple[Any, Any]:
    torch = import_torch()
    targets = torch.stack([dataset[index]["targets"] for index in range(len(dataset))])
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
        normalized_targets = (targets - target_mean.to(device)) / target_std.to(device)

        optimizer.zero_grad()
        predictions = model(front_images, side_images)
        loss = loss_fn(predictions, normalized_targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_mae(model: Any, loader: Any, target_mean: Any, target_std: Any, device: Any) -> dict[str, Any]:
    torch = import_torch()
    model.eval()
    prediction_rows = []
    target_rows = []
    with torch.no_grad():
        for batch in loader:
            front_images = batch["front_image"].to(device)
            side_images = batch["side_image"].to(device)
            targets = batch["targets"].to(device)
            normalized_predictions = model(front_images, side_images)
            predictions = normalized_predictions * target_std.to(device) + target_mean.to(device)
            prediction_rows.append(predictions.cpu())
            target_rows.append(targets.cpu())

    predictions = torch.cat(prediction_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    absolute_errors = torch.abs(predictions - targets).numpy()
    mae_by_target = {
        target: float(absolute_errors[:, index].mean())
        for index, target in enumerate(TARGET_COLUMNS)
    }
    return {
        "overall_mae": float(np.mean(list(mae_by_target.values()))),
        "mae_by_target": mae_by_target,
    }


def build_config(
    dataset_root: Path,
    epochs: int,
    limit_samples: int | None,
    image_size: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "model_type": MODEL_TYPE,
        "target_columns": list(TARGET_COLUMNS),
        "image_size": image_size,
        "epochs": epochs,
        "limit_samples": limit_samples,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": "cpu",
        "experiment_runner_version": EXPERIMENT_RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def format_training_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join(
        [
            f"Model: {metrics['model_type']}",
            f"Samples: train={metrics['sample_counts']['train']} val={metrics['sample_counts']['val']}",
            f"train overall MAE: {metrics['train']['overall_mae']:.4f}",
            f"val overall MAE: {metrics['val']['overall_mae']:.4f}",
            f"Config artifact: {result['config_path']}",
            f"Metrics artifact: {result['metrics_path']}",
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
        )
    except DeepLearningDependencyError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(format_training_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
