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

from training.datasets.verified_measurement_dataset import VerifiedMeasurementDatasetLoader
from training.train_baseline_measurements import _mean


MODEL_FILENAME = "model.json"
CONFIG_FILENAME = "training_config.json"
METRICS_FILENAME = "training_metrics.json"
REGISTRY_FILENAME = "candidate_model_registry.json"
REPORT_FILENAME = "candidate_training_report.md"
MODEL_FAMILY = "verified_measurement_metadata_ridge_regressor"
ARTIFACT_TYPE = "candidate_body_ai_measurement_model"
DEFAULT_TARGET_COLUMNS = [
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "sleeve_cm",
    "inseam_cm",
    "neck_cm",
]
DEFAULT_SEED = 42


class CandidateTrainingError(ValueError):
    pass


def train_candidate_model(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    records_file: str | Path | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    random_seed: int = DEFAULT_SEED,
    ridge_alpha: float = 1.0,
    val_size: float = 0.2,
    test_size: float = 0.2,
    target_columns: list[str] | None = None,
    compatibility_mode: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not compatibility_mode:
        raise CandidateTrainingError(
            "Only compatibility mode is available in Phase H.2; image pixels are validated but not consumed."
        )

    targets = target_columns or list(DEFAULT_TARGET_COLUMNS)
    loader = VerifiedMeasurementDatasetLoader(dataset_root, records_file)
    samples = list(loader)
    selected_version = _resolve_dataset_version(samples, dataset_version)
    samples = [sample for sample in samples if sample["dataset_version"] == selected_version]
    _require_enough_candidate_samples(samples)
    _require_target_coverage(samples, targets)

    feature_names = build_feature_names(samples)
    if not feature_names:
        raise CandidateTrainingError("Candidate training requires numeric pose, validation, or correction-delta features.")

    split_indices = split_indices_for_count(len(samples), random_seed, val_size, test_size)
    split_samples = {
        split: [samples[index] for index in indices]
        for split, indices in split_indices.items()
    }

    train_features = feature_matrix(split_samples["train"], feature_names)
    train_targets = target_matrix(split_samples["train"], targets)
    model_core = train_ridge_regressor(train_features, train_targets, feature_names, targets, ridge_alpha)
    metrics = build_training_metrics(model_core, split_samples, feature_names, targets)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_model_version = resolve_model_version(output_path, model_version)
    timestamp = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    training_config = build_training_config(
        dataset_root=dataset_root,
        records_file=loader.records_path,
        dataset_version=selected_version,
        model_version=resolved_model_version,
        random_seed=random_seed,
        ridge_alpha=ridge_alpha,
        val_size=val_size,
        test_size=test_size,
        target_columns=targets,
        feature_names=feature_names,
        compatibility_mode=compatibility_mode,
    )
    model = {
        **model_core,
        "artifactType": ARTIFACT_TYPE,
        "modelVersion": resolved_model_version,
        "datasetVersion": selected_version,
        "trainingTimestamp": timestamp,
        "recordCount": len(samples),
        "trainingConfig": training_config,
        "trainingMetrics": metrics,
        "candidateOnly": True,
        "isProduction": False,
        "imageUsage": {
            "front": "reference_validated_only",
            "side": "reference_validated_only",
            "back": "reference_validated_only",
            "pixelsConsumed": False,
            "compatibilityMode": True,
        },
    }

    model_path = output_path / MODEL_FILENAME
    config_path = output_path / CONFIG_FILENAME
    metrics_path = output_path / METRICS_FILENAME
    registry_path = output_path / REGISTRY_FILENAME
    report_path = output_path / REPORT_FILENAME
    _write_json(model_path, model)
    _write_json(config_path, training_config)
    _write_json(metrics_path, metrics)
    registry = update_candidate_registry(
        registry_path,
        model=model,
        model_path=model_path,
        config_path=config_path,
        metrics_path=metrics_path,
    )
    report_path.write_text(format_candidate_training_report(model, metrics, registry), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "registry_path": str(registry_path),
        "report_path": str(report_path),
        "model": model,
        "training_config": training_config,
        "metrics": metrics,
        "registry": registry,
    }


def build_training_config(
    *,
    dataset_root: str | Path,
    records_file: str | Path,
    dataset_version: str,
    model_version: str,
    random_seed: int,
    ridge_alpha: float,
    val_size: float,
    test_size: float,
    target_columns: list[str],
    feature_names: list[str],
    compatibility_mode: bool,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_root),
        "recordsFile": str(records_file),
        "datasetVersion": dataset_version,
        "modelVersion": model_version,
        "randomSeed": random_seed,
        "splitPolicy": {
            "method": "deterministic_shuffle",
            "valSize": val_size,
            "testSize": test_size,
        },
        "model": {
            "family": MODEL_FAMILY,
            "artifactType": ARTIFACT_TYPE,
            "regressionMethod": "ridge_regression",
            "hyperparameters": {"ridgeAlpha": ridge_alpha},
        },
        "targetColumns": list(target_columns),
        "featurePipeline": {
            "mode": "compatibility_metadata_features",
            "featureCount": len(feature_names),
            "featureNames": list(feature_names),
            "usesPoseMetadata": True,
            "usesValidationMetadata": True,
            "usesCorrectionDeltas": True,
            "usesFinalApprovedMeasurementsAsTargets": True,
            "usesImagePixels": False,
            "validatedImageReferences": ["front", "side", "back"],
            "limitation": "Phase H.2 validates image references but does not consume image pixels in compatibility mode.",
        },
        "candidateOnly": True,
        "isProduction": False,
    }


def build_feature_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        names.update(sample_feature_values(sample).keys())
    return sorted(names)


def sample_feature_values(sample: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {
        "image_reference.front_exists": 1.0 if sample["front_image_path"].exists() else 0.0,
        "image_reference.side_exists": 1.0 if sample["side_image_path"].exists() else 0.0,
        "image_reference.back_exists": 1.0 if sample["back_image_path"].exists() else 0.0,
    }
    values.update(flatten_numeric_features(sample["pose_metadata_summary"], "pose"))
    values.update(flatten_numeric_features(sample["validation_metadata_summary"], "validation"))
    values.update(flatten_numeric_features(sample["correction_deltas"], "correction_delta"))
    return values


def flatten_numeric_features(value: Any, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = normalize_feature_key(str(key))
            features.update(flatten_numeric_features(nested, f"{prefix}.{normalized_key}"))
        return features
    if isinstance(value, list):
        features[f"{prefix}.__count"] = float(len(value))
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list)):
                features.update(flatten_numeric_features(nested, f"{prefix}.{index}"))
            elif isinstance(nested, (int, float, bool)):
                features[f"{prefix}.{index}"] = float(nested)
        return features
    if isinstance(value, bool):
        return {prefix: 1.0 if value else 0.0}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    return features


def feature_matrix(samples: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        values = sample_feature_values(sample)
        rows.append([values.get(feature, 0.0) for feature in feature_names])
    return np.asarray(rows, dtype=np.float64)


def target_matrix(samples: list[dict[str, Any]], target_columns: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        row = []
        for target in target_columns:
            row.append(measurement_value(sample["final_approved_measurements"].get(target), sample["sample_id"], target))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def train_ridge_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    if features.shape[0] < 2:
        raise CandidateTrainingError("Need at least two training rows for candidate model training.")
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    standardized = (features - feature_means) / feature_stds
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {
        "modelFamily": MODEL_FAMILY,
        "featureNames": list(feature_names),
        "targetColumns": list(target_columns),
        "ridgeAlpha": ridge_alpha,
        "featureMeans": feature_means.tolist(),
        "featureStds": feature_stds.tolist(),
        "intercepts": coefficients[0, :].tolist(),
        "coefficients": coefficients[1:, :].tolist(),
    }


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    feature_means = np.asarray(model["featureMeans"], dtype=np.float64)
    feature_stds = np.asarray(model["featureStds"], dtype=np.float64)
    intercepts = np.asarray(model["intercepts"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (features - feature_means) / feature_stds
    return standardized @ coefficients + intercepts


def build_training_metrics(
    model: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    metrics = {
        "modelFamily": MODEL_FAMILY,
        "artifactType": ARTIFACT_TYPE,
        "targetColumns": list(target_columns),
        "sampleCounts": {split: len(samples) for split, samples in split_samples.items()},
        "metric": "mean_absolute_error_cm",
        "imagePixelsConsumed": False,
        "compatibilityMode": True,
    }
    for split, samples in split_samples.items():
        metrics[split] = evaluate_model(model, samples, feature_names, target_columns)
    return metrics


def evaluate_model(
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    if not samples:
        raise CandidateTrainingError("Cannot evaluate candidate model with zero samples.")
    predictions = predict(model, feature_matrix(samples, feature_names))
    targets = target_matrix(samples, target_columns)
    absolute_errors = np.abs(predictions - targets)
    mae_by_target = {
        target: round(float(absolute_errors[:, index].mean()), 6)
        for index, target in enumerate(target_columns)
    }
    return {
        "overallMae": round(_mean(list(mae_by_target.values())), 6),
        "maeByTarget": mae_by_target,
    }


def split_indices_for_count(count: int, seed: int, val_size: float, test_size: float) -> dict[str, list[int]]:
    if count < 4:
        raise CandidateTrainingError("Need at least four records for train/val/test candidate training.")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise CandidateTrainingError("val_size and test_size must be positive and leave training rows.")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    test_count = max(1, int(round(count * test_size)))
    val_count = max(1, int(round(count * val_size)))
    train_count = count - val_count - test_count
    if train_count < 2:
        train_count = 2
        remaining = count - train_count
        val_count = max(1, remaining // 2)
        test_count = remaining - val_count
    if test_count < 1 or val_count < 1:
        raise CandidateTrainingError("Need at least one validation and one test record.")
    return {
        "train": indices[:train_count],
        "val": indices[train_count: train_count + val_count],
        "test": indices[train_count + val_count:],
    }


def update_candidate_registry(
    registry_path: str | Path,
    *,
    model: dict[str, Any],
    model_path: str | Path,
    config_path: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    path = Path(registry_path)
    registry = read_registry(path)
    entry = {
        "modelVersion": model["modelVersion"],
        "datasetVersion": model["datasetVersion"],
        "trainingTimestamp": model["trainingTimestamp"],
        "recordCount": model["recordCount"],
        "artifactType": model["artifactType"],
        "modelFamily": model["modelFamily"],
        "candidateStatus": "ready_for_evaluation",
        "productionStatus": "not_production",
        "isProduction": False,
        "promoted": False,
        "productionModelUpdated": False,
        "modelPath": str(model_path),
        "trainingConfigPath": str(config_path),
        "trainingMetricsPath": str(metrics_path),
        "trainingMetrics": model["trainingMetrics"],
    }
    entries = [existing for existing in registry.get("candidates", []) if existing.get("modelVersion") != model["modelVersion"]]
    entries.append(entry)
    registry = {
        "schemaVersion": "candidate_model_registry_v1",
        "productionModelVersion": registry.get("productionModelVersion"),
        "productionModelUpdated": False,
        "candidates": sorted(entries, key=lambda row: row["modelVersion"]),
    }
    _write_json(path, registry)
    return registry


def read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schemaVersion": "candidate_model_registry_v1", "productionModelVersion": None, "candidates": []}
    with path.open("r", encoding="utf-8") as registry_file:
        payload = json.load(registry_file)
    if not isinstance(payload, dict):
        raise CandidateTrainingError(f"Candidate registry must be a JSON object: {path}")
    payload.setdefault("candidates", [])
    return payload


def resolve_model_version(output_dir: Path, requested: str | None) -> str:
    if requested:
        _validate_model_version(requested)
        return requested
    registry = read_registry(output_dir / REGISTRY_FILENAME)
    highest = 0
    for entry in registry.get("candidates", []):
        match = re.match(r"^candidate_model_v([1-9][0-9]*)$", str(entry.get("modelVersion", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"candidate_model_v{highest + 1}"


def _resolve_dataset_version(samples: list[dict[str, Any]], requested: str | None) -> str:
    versions = sorted({str(sample["dataset_version"]) for sample in samples})
    if requested:
        if requested not in versions:
            raise CandidateTrainingError(f"Requested dataset version '{requested}' was not found; available versions: {', '.join(versions)}")
        return requested
    if len(versions) != 1:
        raise CandidateTrainingError(f"Multiple dataset versions found; pass --dataset-version. Available versions: {', '.join(versions)}")
    return versions[0]


def _require_enough_candidate_samples(samples: list[dict[str, Any]]) -> None:
    if len(samples) < 4:
        raise CandidateTrainingError(f"Need at least four verified records for candidate training; got {len(samples)}.")


def _require_target_coverage(samples: list[dict[str, Any]], target_columns: list[str]) -> None:
    missing: dict[str, list[str]] = {}
    for sample in samples:
        for target in target_columns:
            try:
                measurement_value(sample["final_approved_measurements"].get(target), sample["sample_id"], target)
            except CandidateTrainingError:
                missing.setdefault(sample["sample_id"], []).append(target)
    if missing:
        details = "; ".join(f"{sample_id}: {', '.join(targets)}" for sample_id, targets in sorted(missing.items()))
        raise CandidateTrainingError(f"Verified records are missing required final approved target values: {details}")


def measurement_value(value: Any, sample_id: str, target: str) -> float:
    if isinstance(value, dict):
        for key in ("value_cm", "valueCm", "final_cm", "finalCm", "approved_cm", "approvedCm", "estimate_cm", "estimateCm"):
            if key in value:
                return measurement_value(value[key], sample_id, target)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CandidateTrainingError(f"Sample {sample_id} is missing numeric final approved value for {target}.") from error


def _validate_model_version(model_version: str) -> None:
    if not re.match(r"^candidate_model_v[1-9][0-9]*$", model_version):
        raise CandidateTrainingError("model_version must look like candidate_model_v1, candidate_model_v2, ...")


def normalize_feature_key(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip())
    return spaced.replace("-", "_").replace(" ", "_").lower()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def format_candidate_training_report(model: dict[str, Any], metrics: dict[str, Any], registry: dict[str, Any]) -> str:
    lines = [
        "# Phase H.2 Candidate Model Training",
        "",
        f"Model version: `{model['modelVersion']}`",
        f"Dataset version: `{model['datasetVersion']}`",
        f"Candidate status: `ready_for_evaluation`",
        f"Production model updated: `{registry['productionModelUpdated']}`",
        f"Image pixels consumed: `{metrics['imagePixelsConsumed']}`",
        "",
        "## Metrics",
        "",
        "| Split | Records | Overall MAE |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {metrics['sampleCounts'][split]} | {metrics[split]['overallMae']:.4f} |")
    lines.extend(["", "## Per-Measurement Test MAE", "", "| Measurement | MAE |", "| --- | ---: |"])
    for target, mae in metrics["test"]["maeByTarget"].items():
        lines.append(f"| `{target}` | {mae:.4f} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Compatibility mode validates front, side, and back image references but does not consume image pixels.",
            "- Pose metadata, validation metadata, and correction deltas are numeric compatibility features.",
            "- This artifact is a candidate only and is not wired into production inference or API responses.",
            "",
        ]
    )
    return "\n".join(lines)


def format_training_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return "\n".join(
        [
            f"Candidate model: {result['model']['modelVersion']}",
            f"Dataset version: {result['model']['datasetVersion']}",
            f"Model artifact: {result['model_path']}",
            f"Metrics artifact: {result['metrics_path']}",
            f"Registry: {result['registry_path']}",
            f"Test overall MAE: {metrics['test']['overallMae']:.4f}",
            "Production model updated: false",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a candidate-only Body AI measurement model from verified datasets.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--output", required=True, help="Output directory for candidate artifacts.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--dataset-version", help="Verified dataset version to train, such as v1.")
    parser.add_argument("--model-version", help="Candidate model version, such as candidate_model_v1.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-columns", nargs="+", default=DEFAULT_TARGET_COLUMNS)
    args = parser.parse_args(argv)

    result = train_candidate_model(
        args.dataset,
        args.output,
        records_file=args.records_file,
        dataset_version=args.dataset_version,
        model_version=args.model_version,
        random_seed=args.random_seed,
        ridge_alpha=args.ridge_alpha,
        val_size=args.val_size,
        test_size=args.test_size,
        target_columns=list(args.target_columns),
    )
    print(format_training_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
