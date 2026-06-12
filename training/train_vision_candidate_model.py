from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

from training.datasets.multimodal_verified_dataset import (
    READINESS_MULTIMODAL_READY,
    MultimodalVerifiedDataset,
)
from training.train_baseline_measurements import _mean
from training.train_candidate_model import DEFAULT_SEED, DEFAULT_TARGET_COLUMNS, measurement_value, split_indices_for_count
from training.measurements.measurement_targets import GENDER_MEASUREMENT_SCHEMA_VERSION, target_available_for_profile


VISION_MODEL_JSON = "vision_model.json"
VISION_MODEL_WEIGHTS = "vision_model.pt"
VISION_CONFIG_JSON = "vision_training_config.json"
VISION_METRICS_JSON = "vision_training_metrics.json"
VISION_REGISTRY_JSON = "vision_candidate_model_registry.json"
VISION_REPORT_MD = "vision_candidate_training_report.md"
MODEL_FAMILY = "vision_multimodal_three_view_regressor"
ARTIFACT_TYPE = "vision_multimodal_body_ai_candidate"
CANDIDATE_TYPE = "vision_multimodal"


class VisionCandidateTrainingError(ValueError):
    pass


class ViewImageEncoder(nn.Module):
    def __init__(self, branch_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, branch_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dim: int, branch_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, branch_dim),
            nn.ReLU(),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.encoder(metadata)


class VisionMultimodalRegressor(nn.Module):
    def __init__(self, metadata_dim: int, target_count: int, branch_dim: int = 8, fusion_dim: int = 32) -> None:
        super().__init__()
        self.front_image_encoder = ViewImageEncoder(branch_dim)
        self.side_image_encoder = ViewImageEncoder(branch_dim)
        self.back_image_encoder = ViewImageEncoder(branch_dim)
        self.metadata_feature_encoder = MetadataEncoder(metadata_dim, branch_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(branch_dim * 4, fusion_dim),
            nn.ReLU(),
        )
        self.measurement_prediction_head = nn.Linear(fusion_dim, target_count)

    def forward(
        self,
        front_image: torch.Tensor,
        side_image: torch.Tensor,
        back_image: torch.Tensor,
        metadata_features: torch.Tensor,
    ) -> torch.Tensor:
        front = self.front_image_encoder(front_image)
        side = self.side_image_encoder(side_image)
        back = self.back_image_encoder(back_image)
        metadata = self.metadata_feature_encoder(metadata_features)
        fused = self.fusion_layer(torch.cat([front, side, back, metadata], dim=1))
        return self.measurement_prediction_head(fused)


def train_vision_candidate_model(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    records_file: str | Path | None = None,
    storage_root: str | Path | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    random_seed: int = DEFAULT_SEED,
    image_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 0.001,
    branch_dim: int = 8,
    fusion_dim: int = 32,
    val_size: float = 0.2,
    test_size: float = 0.2,
    target_columns: list[str] | None = None,
    device: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    set_deterministic_seed(random_seed)
    targets = target_columns or list(DEFAULT_TARGET_COLUMNS)
    dataset = MultimodalVerifiedDataset(
        dataset_root,
        records_file,
        storage_root=storage_root,
        image_size=(image_size, image_size),
        include_tensors=True,
    )
    samples = list(dataset)
    selected_version = resolve_dataset_version(samples, dataset_version)
    samples = [sample for sample in samples if sample["datasetVersion"] == selected_version]
    require_multimodal_ready(samples)
    require_image_tensors(samples)
    require_target_coverage(samples, targets)
    metadata_feature_names = build_metadata_feature_names(samples)
    if not metadata_feature_names:
        raise VisionCandidateTrainingError("Vision candidate training requires numeric pose, validation, or verification metadata features.")

    split_indices = split_indices_for_count(len(samples), random_seed, val_size, test_size)
    split_samples = {split: [samples[index] for index in indices] for split, indices in split_indices.items()}
    tensors = build_tensor_batches(split_samples, metadata_feature_names, targets)
    train_targets = tensors["train"]["targets"]
    train_targets_np = train_targets.numpy()
    target_mean = torch.as_tensor(np.nanmean(train_targets_np, axis=0), dtype=torch.float32)
    target_std = torch.as_tensor(np.nanstd(train_targets_np, axis=0), dtype=torch.float32)
    target_std = torch.where(target_std < 1e-6, torch.ones_like(target_std), target_std)

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VisionMultimodalRegressor(
        metadata_dim=len(metadata_feature_names),
        target_count=len(targets),
        branch_dim=branch_dim,
        fusion_dim=fusion_dim,
    ).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for _epoch in range(epochs):
        model.train()
        batch = move_batch(tensors["train"], selected_device)
        optimizer.zero_grad()
        predictions = model(batch["front"], batch["side"], batch["back"], batch["metadata"])
        normalized_targets = (batch["targets"] - target_mean.to(selected_device)) / target_std.to(selected_device)
        mask = ~torch.isnan(normalized_targets)
        loss = loss_fn(predictions[mask], normalized_targets[mask])
        loss.backward()
        optimizer.step()

    metrics = build_metrics(model, tensors, targets, target_mean, target_std, selected_device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_model_version = resolve_model_version(output_path, model_version)
    timestamp = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    training_config = build_training_config(
        dataset_root=dataset_root,
        records_file=dataset.loader.records_path,
        dataset_version=selected_version,
        model_version=resolved_model_version,
        random_seed=random_seed,
        image_size=image_size,
        epochs=epochs,
        learning_rate=learning_rate,
        branch_dim=branch_dim,
        fusion_dim=fusion_dim,
        val_size=val_size,
        test_size=test_size,
        target_columns=targets,
        metadata_feature_names=metadata_feature_names,
        device=str(selected_device),
    )
    model_metadata = {
        "artifactType": ARTIFACT_TYPE,
        "candidateType": CANDIDATE_TYPE,
        "modelFamily": MODEL_FAMILY,
        "modelVersion": resolved_model_version,
        "datasetVersion": selected_version,
        "trainingTimestamp": timestamp,
        "recordCount": len(samples),
        "targetColumns": targets,
        "measurementSchemaVersion": GENDER_MEASUREMENT_SCHEMA_VERSION,
        "architecture": architecture_summary(model, len(metadata_feature_names), len(targets), branch_dim, fusion_dim),
        "imageUsage": {
            "pixelsConsumed": True,
            "front": "front_image_encoder",
            "side": "side_image_encoder",
            "back": "back_image_encoder",
            "separateViewBranches": True,
        },
        "metadataUsage": {
            "branch": "metadata_feature_encoder",
            "featureNames": metadata_feature_names,
            "excludesLineageMeasurements": True,
            "excludesCorrectionDeltas": True,
            "excludesFinalApprovedInputs": True,
        },
        "targetNormalization": {
            "mean": target_mean.detach().cpu().tolist(),
            "std": target_std.detach().cpu().tolist(),
        },
        "candidateOnly": True,
        "isProduction": False,
        "trainingConfig": training_config,
        "trainingMetrics": metrics,
    }
    model_json_path = output_path / VISION_MODEL_JSON
    weights_path = output_path / VISION_MODEL_WEIGHTS
    config_path = output_path / VISION_CONFIG_JSON
    metrics_path = output_path / VISION_METRICS_JSON
    registry_path = output_path / VISION_REGISTRY_JSON
    report_path = output_path / VISION_REPORT_MD
    write_json(model_json_path, model_metadata)
    torch.save(model.state_dict(), weights_path)
    write_json(config_path, training_config)
    write_json(metrics_path, metrics)
    registry = update_registry(registry_path, model_metadata, model_json_path, weights_path, config_path, metrics_path)
    report_path.write_text(format_training_report(model_metadata, metrics, registry), encoding="utf-8")
    return {
        "model_json_path": str(model_json_path),
        "model_weights_path": str(weights_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "registry_path": str(registry_path),
        "report_path": str(report_path),
        "model_metadata": model_metadata,
        "metrics": metrics,
        "registry": registry,
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def resolve_dataset_version(samples: list[dict[str, Any]], requested: str | None) -> str:
    versions = sorted({str(sample["datasetVersion"]) for sample in samples})
    if requested:
        if requested not in versions:
            raise VisionCandidateTrainingError(f"Requested dataset version '{requested}' was not found; available versions: {', '.join(versions)}")
        return requested
    if len(versions) != 1:
        raise VisionCandidateTrainingError(f"Multiple dataset versions found; pass --dataset-version. Available versions: {', '.join(versions)}")
    return versions[0]


def require_multimodal_ready(samples: list[dict[str, Any]]) -> None:
    if len(samples) < 4:
        raise VisionCandidateTrainingError(f"Need at least four multimodal records for vision candidate training; got {len(samples)}.")
    not_ready = [sample["sampleId"] for sample in samples if sample["readinessState"] != READINESS_MULTIMODAL_READY]
    if not_ready:
        raise VisionCandidateTrainingError(f"Vision candidate training requires multimodal_ready records; not ready: {', '.join(not_ready)}")


def require_image_tensors(samples: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for sample in samples:
        for view in ("front", "side", "back"):
            image = sample[f"{view}Image"]
            tensor = image.get("tensor")
            if tensor is None:
                missing.append(f"{sample['sampleId']}:{view}")
    if missing:
        raise VisionCandidateTrainingError(f"Vision candidate training requires front/side/back image tensors; missing: {', '.join(missing)}")


def require_target_coverage(samples: list[dict[str, Any]], target_columns: list[str]) -> None:
    missing: dict[str, list[str]] = {}
    compatible_counts: dict[str, int] = {target: 0 for target in target_columns}
    for sample in samples:
        for target in target_columns:
            if not target_available_for_profile(target, sample.get("profileType")):
                continue
            compatible_counts[target] += 1
            try:
                target_value(sample, target)
            except VisionCandidateTrainingError:
                missing.setdefault(sample["sampleId"], []).append(target)
    if missing:
        details = "; ".join(f"{sample_id}: {', '.join(targets)}" for sample_id, targets in sorted(missing.items()))
        raise VisionCandidateTrainingError(f"Multimodal records are missing final approved target values: {details}")
    insufficient = [target for target, count in compatible_counts.items() if count < 2]
    if insufficient:
        raise VisionCandidateTrainingError(f"Need at least two compatible records for target values: {', '.join(insufficient)}")


def target_value(sample: dict[str, Any], target: str) -> float:
    if not target_available_for_profile(target, sample.get("profileType")):
        return float("nan")
    try:
        return measurement_value(sample["finalApprovedMeasurements"].get(target), sample["sampleId"], target)
    except Exception as error:
        raise VisionCandidateTrainingError(f"Sample {sample['sampleId']} is missing numeric final approved value for {target}.") from error


def build_metadata_feature_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        names.update(raw_metadata_feature_values(sample).keys())
    blocked = [name for name in names if forbidden_feature_name(name)]
    if blocked:
        raise VisionCandidateTrainingError(f"Forbidden leakage-prone metadata features were found: {', '.join(sorted(blocked))}")
    return sorted(name for name in names if not forbidden_feature_name(name))


def metadata_feature_values(sample: dict[str, Any]) -> dict[str, float]:
    values = raw_metadata_feature_values(sample)
    return {name: value for name, value in values.items() if not forbidden_feature_name(name)}


def raw_metadata_feature_values(sample: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    values.update(flatten_numeric(sample["poseMetadata"], "pose"))
    values.update(flatten_numeric(sample["validationMetadata"], "validation"))
    values.update(flatten_numeric(sample["verificationMetadata"], "verification"))
    return values


def flatten_numeric(value: Any, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            features.update(flatten_numeric(nested, f"{prefix}.{normalize_key(str(key))}"))
        return features
    if isinstance(value, list):
        features[f"{prefix}.__count"] = float(len(value))
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list)):
                features.update(flatten_numeric(nested, f"{prefix}.{index}"))
            elif isinstance(nested, (bool, int, float)):
                features[f"{prefix}.{index}"] = float(nested)
        return features
    if isinstance(value, bool):
        return {prefix: 1.0 if value else 0.0}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    return features


def forbidden_feature_name(name: str) -> bool:
    normalized = normalize_key(name)
    forbidden_tokens = (
        "final_approved",
        "finalapproved",
        "customer_edit",
        "customer_measurement",
        "maker_adjustment",
        "maker_measurement",
        "correction_delta",
        "correctiondeltas",
        "lineage",
        "ai_estimate",
    )
    return any(token in normalized for token in forbidden_tokens)


def build_tensor_batches(
    split_samples: dict[str, list[dict[str, Any]]],
    metadata_feature_names: list[str],
    target_columns: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        split: {
            "front": image_tensor(samples, "front"),
            "side": image_tensor(samples, "side"),
            "back": image_tensor(samples, "back"),
            "metadata": metadata_tensor(samples, metadata_feature_names),
            "targets": targets_tensor(samples, target_columns),
        }
        for split, samples in split_samples.items()
    }


def image_tensor(samples: list[dict[str, Any]], view: str) -> torch.Tensor:
    arrays = []
    for sample in samples:
        tensor = sample[f"{view}Image"]["tensor"]
        if tensor is None:
            raise VisionCandidateTrainingError(f"Missing {view} image tensor for sample {sample['sampleId']}.")
        arrays.append(np.asarray(tensor, dtype=np.float32).transpose(2, 0, 1))
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32)


def metadata_tensor(samples: list[dict[str, Any]], feature_names: list[str]) -> torch.Tensor:
    rows = []
    for sample in samples:
        values = metadata_feature_values(sample)
        rows.append([values.get(feature, 0.0) for feature in feature_names])
    return torch.as_tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32)


def targets_tensor(samples: list[dict[str, Any]], target_columns: list[str]) -> torch.Tensor:
    rows = []
    for sample in samples:
        rows.append([target_value(sample, target) for target in target_columns])
    return torch.as_tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_metrics(
    model: VisionMultimodalRegressor,
    tensors: dict[str, dict[str, torch.Tensor]],
    target_columns: list[str],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "modelFamily": MODEL_FAMILY,
        "candidateType": CANDIDATE_TYPE,
        "targetColumns": list(target_columns),
        "metric": "mean_absolute_error_cm",
        "pixelsConsumed": True,
        "sampleCounts": {split: int(batch["targets"].shape[0]) for split, batch in tensors.items()},
    }
    for split, batch in tensors.items():
        metrics[split] = evaluate_split(model, batch, target_columns, target_mean, target_std, device)
    return metrics


def evaluate_split(
    model: VisionMultimodalRegressor,
    batch: dict[str, torch.Tensor],
    target_columns: list[str],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        moved = move_batch(batch, device)
        normalized_predictions = model(moved["front"], moved["side"], moved["back"], moved["metadata"])
        predictions = normalized_predictions.cpu() * target_std + target_mean
        targets = batch["targets"].cpu()
        absolute_errors = torch.abs(predictions - targets).numpy()
    mae_by_target = {
        target: round(float(np.nanmean(absolute_errors[:, index])), 6)
        for index, target in enumerate(target_columns)
        if not np.isnan(absolute_errors[:, index]).all()
    }
    return {
        "overallMae": round(_mean(list(mae_by_target.values())), 6),
        "maeByTarget": mae_by_target,
    }


def build_training_config(
    *,
    dataset_root: str | Path,
    records_file: str | Path,
    dataset_version: str,
    model_version: str,
    random_seed: int,
    image_size: int,
    epochs: int,
    learning_rate: float,
    branch_dim: int,
    fusion_dim: int,
    val_size: float,
    test_size: float,
    target_columns: list[str],
    metadata_feature_names: list[str],
    device: str,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "recordsFile": str(records_file),
        "datasetVersion": dataset_version,
        "modelVersion": model_version,
        "randomSeed": random_seed,
        "device": device,
        "splitPolicy": {
            "method": "deterministic_shuffle",
            "valSize": val_size,
            "testSize": test_size,
        },
        "model": {
            "family": MODEL_FAMILY,
            "artifactType": ARTIFACT_TYPE,
            "candidateType": CANDIDATE_TYPE,
            "architecture": "front_side_back_image_branches_plus_metadata_fusion",
            "hyperparameters": {
                "epochs": epochs,
                "learningRate": learning_rate,
                "branchDim": branch_dim,
                "fusionDim": fusion_dim,
                "imageSize": image_size,
            },
        },
        "inputs": {
            "frontImageTensor": True,
            "sideImageTensor": True,
            "backImageTensor": True,
            "poseMetadata": True,
            "validationMetadata": True,
            "verificationMetadata": True,
            "correctionDeltas": False,
            "lineageMeasurements": False,
            "finalApprovedMeasurementsAsInputs": False,
            "finalApprovedMeasurementsAsTargets": True,
        },
        "targetColumns": target_columns,
        "measurementSchemaVersion": GENDER_MEASUREMENT_SCHEMA_VERSION,
        "metadataFeatureNames": metadata_feature_names,
        "pixelsConsumed": True,
        "candidateOnly": True,
        "isProduction": False,
    }


def architecture_summary(
    model: VisionMultimodalRegressor,
    metadata_dim: int,
    target_count: int,
    branch_dim: int,
    fusion_dim: int,
) -> dict[str, Any]:
    return {
        "frontImageEncoder": model.front_image_encoder.__class__.__name__,
        "sideImageEncoder": model.side_image_encoder.__class__.__name__,
        "backImageEncoder": model.back_image_encoder.__class__.__name__,
        "metadataFeatureEncoder": model.metadata_feature_encoder.__class__.__name__,
        "fusionLayer": "concat(front, side, back, metadata) -> fusion MLP",
        "measurementPredictionHead": model.measurement_prediction_head.__class__.__name__,
        "metadataFeatureCount": metadata_dim,
        "targetCount": target_count,
        "branchDim": branch_dim,
        "fusionDim": fusion_dim,
    }


def update_registry(
    registry_path: str | Path,
    model_metadata: dict[str, Any],
    model_json_path: str | Path,
    weights_path: str | Path,
    config_path: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    path = Path(registry_path)
    registry = read_registry(path)
    entry = {
        "modelVersion": model_metadata["modelVersion"],
        "datasetVersion": model_metadata["datasetVersion"],
        "candidateType": CANDIDATE_TYPE,
        "pixelsConsumed": True,
        "productionModelUpdated": False,
        "readyForEvaluation": True,
        "isProduction": False,
        "promoted": False,
        "artifactType": ARTIFACT_TYPE,
        "modelFamily": MODEL_FAMILY,
        "recordCount": model_metadata["recordCount"],
        "trainingTimestamp": model_metadata["trainingTimestamp"],
        "modelJsonPath": str(model_json_path),
        "modelWeightsPath": str(weights_path),
        "trainingConfigPath": str(config_path),
        "trainingMetricsPath": str(metrics_path),
        "trainingMetrics": model_metadata["trainingMetrics"],
    }
    entries = [existing for existing in registry.get("candidates", []) if existing.get("modelVersion") != model_metadata["modelVersion"]]
    entries.append(entry)
    registry = {
        "schemaVersion": "vision_candidate_model_registry_v1",
        "productionModelUpdated": False,
        "productionModelVersion": registry.get("productionModelVersion"),
        "candidates": sorted(entries, key=lambda row: row["modelVersion"]),
    }
    write_json(path, registry)
    return registry


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schemaVersion": "vision_candidate_model_registry_v1", "productionModelVersion": None, "candidates": []}
    with path.open("r", encoding="utf-8") as registry_file:
        payload = json.load(registry_file)
    if not isinstance(payload, dict):
        raise VisionCandidateTrainingError(f"Vision candidate registry must be a JSON object: {path}")
    payload.setdefault("candidates", [])
    return payload


def resolve_model_version(output_dir: Path, requested: str | None) -> str:
    if requested:
        if not re.match(r"^vision_candidate_model_v[1-9][0-9]*$", requested):
            raise VisionCandidateTrainingError("model_version must look like vision_candidate_model_v1, vision_candidate_model_v2, ...")
        return requested
    registry = read_registry(output_dir / VISION_REGISTRY_JSON)
    highest = 0
    for entry in registry.get("candidates", []):
        match = re.match(r"^vision_candidate_model_v([1-9][0-9]*)$", str(entry.get("modelVersion", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"vision_candidate_model_v{highest + 1}"


def normalize_key(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip())
    return spaced.replace("-", "_").replace(" ", "_").lower()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def format_training_report(model_metadata: dict[str, Any], metrics: dict[str, Any], registry: dict[str, Any]) -> str:
    lines = [
        "# Phase H.5 Vision Multimodal Candidate",
        "",
        f"Model version: `{model_metadata['modelVersion']}`",
        f"Dataset version: `{model_metadata['datasetVersion']}`",
        f"Candidate type: `{CANDIDATE_TYPE}`",
        f"Pixels consumed: `{model_metadata['imageUsage']['pixelsConsumed']}`",
        f"Production model updated: `{registry['productionModelUpdated']}`",
        "",
        "## Architecture",
        "",
        "- Front image encoder",
        "- Side image encoder",
        "- Back image encoder",
        "- Metadata feature encoder",
        "- Fusion layer",
        "- Measurement prediction head",
        "",
        "## Metrics",
        "",
        "| Split | Records | Overall MAE |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {metrics['sampleCounts'][split]} | {metrics[split]['overallMae']:.4f} |")
    lines.extend(["", "No model was promoted and live inference was not changed.", ""])
    return "\n".join(lines)


def format_training_summary(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Vision candidate: {result['model_metadata']['modelVersion']}",
            f"Model metadata: {result['model_json_path']}",
            f"Model weights: {result['model_weights_path']}",
            f"Metrics: {result['metrics_path']}",
            f"Registry: {result['registry_path']}",
            f"Pixels consumed: {result['model_metadata']['imageUsage']['pixelsConsumed']}",
            "Production model updated: false",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a candidate-only multimodal vision Body AI model.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--output", required=True, help="Output directory for vision candidate artifacts.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--storage-root", help="Optional local root for resolving storage keys.")
    parser.add_argument("--dataset-version")
    parser.add_argument("--model-version")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--branch-dim", type=int, default=8)
    parser.add_argument("--fusion-dim", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-columns", nargs="+", default=DEFAULT_TARGET_COLUMNS)
    parser.add_argument("--device", help="Optional torch device override, such as cpu.")
    args = parser.parse_args(argv)
    result = train_vision_candidate_model(
        args.dataset,
        args.output,
        records_file=args.records_file,
        storage_root=args.storage_root,
        dataset_version=args.dataset_version,
        model_version=args.model_version,
        random_seed=args.random_seed,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        branch_dim=args.branch_dim,
        fusion_dim=args.fusion_dim,
        val_size=args.val_size,
        test_size=args.test_size,
        target_columns=list(args.target_columns),
        device=args.device,
    )
    print(format_training_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
