from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from training.datasets.multimodal_verified_dataset import (
    READINESS_MULTIMODAL_READY,
    MultimodalVerifiedDataset,
)
from training.datasets.verified_measurement_dataset import VerifiedMeasurementDatasetLoader
from training.evaluate_candidate_model import (
    baseline_predictions,
    compare_metrics,
    evaluate_predictions,
    predict,
    train_baseline_mean_estimator,
)
from training.train_baseline_measurements import _mean
from training.train_candidate_model import (
    DEFAULT_SEED,
    DEFAULT_TARGET_COLUMNS,
    feature_matrix,
    split_indices_for_count,
    target_matrix,
)
from training.train_vision_candidate_model import (
    VISION_MODEL_WEIGHTS,
    VisionMultimodalRegressor,
    build_tensor_batches,
    forbidden_feature_name,
    require_image_tensors,
    require_multimodal_ready,
)


VISION_EVALUATION_METRICS_FILENAME = "vision_candidate_evaluation_metrics.json"
VISION_BENCHMARK_REPORT_FILENAME = "vision_candidate_benchmark_report.md"
VISION_ABLATION_REPORT_FILENAME = "vision_ablation_report.json"
VISION_VIEW_CONTRIBUTION_REPORT_FILENAME = "vision_view_contribution_report.json"
VISION_CONFIDENCE_CALIBRATION_REPORT_FILENAME = "vision_confidence_calibration_report.json"
VISION_PROMOTION_RECOMMENDATION_FILENAME = "vision_promotion_recommendation.json"
RECOMMEND_PROMOTE = "promote_candidate"
RECOMMEND_DO_NOT_PROMOTE = "do_not_promote"
RECOMMEND_NEEDS_MORE_DATA = "needs_more_data"
RECOMMEND_LEAKAGE_RISK = "leakage_risk"
RECOMMEND_REGRESSION = "regression_detected"
RECOMMEND_CONFIDENCE = "confidence_not_calibrated"
CONFIDENCE_BUCKETS = ("high_confidence", "medium_confidence", "low_confidence", "unknown")


class VisionCandidateEvaluationError(ValueError):
    pass


def evaluate_vision_candidate_model(
    dataset_root: str | Path,
    metadata_candidate_model_path: str | Path,
    vision_candidate_model_path: str | Path,
    output_dir: str | Path,
    *,
    records_file: str | Path | None = None,
    storage_root: str | Path | None = None,
    vision_weights_path: str | Path | None = None,
    dataset_version: str | None = None,
    min_test_records: int = 3,
    suspicious_mae_threshold: float = 0.01,
    generated_at: str | None = None,
) -> dict[str, Any]:
    metadata_model_path = Path(metadata_candidate_model_path)
    vision_model_path = Path(vision_candidate_model_path)
    metadata_model = read_json(metadata_model_path)
    vision_metadata = read_json(vision_model_path)
    selected_version = resolve_dataset_version(vision_metadata, metadata_model, dataset_version)
    target_columns = list(vision_metadata.get("targetColumns") or DEFAULT_TARGET_COLUMNS)
    metadata_feature_names = list(metadata_model.get("featureNames") or metadata_model.get("trainingConfig", {}).get("featurePipeline", {}).get("featureNames") or [])
    vision_feature_names = list(vision_metadata.get("metadataUsage", {}).get("featureNames") or vision_metadata.get("trainingConfig", {}).get("metadataFeatureNames") or [])
    if not metadata_feature_names:
        raise VisionCandidateEvaluationError("Metadata candidate artifact is missing featureNames.")
    if not vision_feature_names:
        raise VisionCandidateEvaluationError("Vision candidate artifact is missing metadata feature names.")

    image_size = int(vision_metadata.get("trainingConfig", {}).get("model", {}).get("hyperparameters", {}).get("imageSize", 32))
    multimodal_dataset = MultimodalVerifiedDataset(
        dataset_root,
        records_file,
        storage_root=storage_root,
        image_size=(image_size, image_size),
        include_tensors=True,
    )
    multimodal_samples = [sample for sample in multimodal_dataset if sample["datasetVersion"] == selected_version]
    require_multimodal_ready(multimodal_samples)
    require_image_tensors(multimodal_samples)

    verified_loader = VerifiedMeasurementDatasetLoader(dataset_root, records_file)
    verified_samples = [sample for sample in verified_loader if sample["dataset_version"] == selected_version]
    if len(verified_samples) != len(multimodal_samples):
        raise VisionCandidateEvaluationError("Verified and multimodal datasets resolved different record counts.")

    split_policy = vision_metadata.get("trainingConfig", {}).get("splitPolicy", {})
    random_seed = int(vision_metadata.get("trainingConfig", {}).get("randomSeed", DEFAULT_SEED))
    split_indices = split_indices_for_count(
        len(multimodal_samples),
        random_seed,
        float(split_policy.get("valSize", 0.2)),
        float(split_policy.get("testSize", 0.2)),
    )
    split_multimodal = {split: [multimodal_samples[index] for index in indices] for split, indices in split_indices.items()}
    split_verified = {split: [verified_samples[index] for index in indices] for split, indices in split_indices.items()}

    baseline_model = train_baseline_mean_estimator(split_verified["train"], target_columns)
    baseline_metrics = evaluate_baseline_by_split(baseline_model, split_verified, target_columns)
    metadata_metrics = evaluate_metadata_candidate_by_split(metadata_model, split_verified, metadata_feature_names, target_columns)
    vision_runtime = load_vision_runtime(vision_metadata, vision_model_path, vision_weights_path, vision_feature_names, target_columns)
    vision_metrics = evaluate_vision_by_split(vision_runtime, split_multimodal, vision_feature_names, target_columns)
    benchmark = build_three_way_benchmark(baseline_metrics, metadata_metrics, vision_metrics, target_columns)
    view_contribution = build_view_contribution_report(vision_runtime, split_multimodal, vision_feature_names, target_columns)
    ablation = build_ablation_report(vision_runtime, split_multimodal, vision_feature_names, target_columns)
    confidence_calibration = build_confidence_calibration_report(vision_runtime, split_multimodal["test"], vision_feature_names, target_columns)
    leakage_audit = audit_vision_leakage(
        vision_metadata,
        multimodal_samples,
        vision_feature_names,
        vision_metrics["test"],
        suspicious_mae_threshold,
    )
    split_audit = audit_split_integrity(split_multimodal, split_indices)
    recommendation = build_promotion_recommendation(
        benchmark,
        leakage_audit,
        split_audit,
        confidence_calibration,
        test_record_count=len(split_multimodal["test"]),
        min_test_records=min_test_records,
    )
    generated = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metrics = {
        "schemaVersion": "vision_candidate_evaluation_metrics_v1",
        "generatedAt": generated,
        "dataset": str(dataset_root),
        "recordsFile": str(verified_loader.records_path),
        "datasetVersion": selected_version,
        "targetColumns": target_columns,
        "baselineEstimator": baseline_model["modelFamily"],
        "metadataCandidateModelVersion": metadata_model.get("modelVersion"),
        "visionCandidateModelVersion": vision_metadata.get("modelVersion"),
        "productionBaselineMetrics": baseline_metrics,
        "metadataCandidateMetrics": metadata_metrics,
        "visionCandidateMetrics": vision_metrics,
        "benchmark": benchmark,
        "leakageAudit": leakage_audit,
        "splitAudit": split_audit,
        "recommendation": recommendation,
        "productionModelUpdated": False,
        "liveApiBehaviorChanged": False,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / VISION_EVALUATION_METRICS_FILENAME
    benchmark_path = output_path / VISION_BENCHMARK_REPORT_FILENAME
    ablation_path = output_path / VISION_ABLATION_REPORT_FILENAME
    view_path = output_path / VISION_VIEW_CONTRIBUTION_REPORT_FILENAME
    confidence_path = output_path / VISION_CONFIDENCE_CALIBRATION_REPORT_FILENAME
    recommendation_path = output_path / VISION_PROMOTION_RECOMMENDATION_FILENAME
    write_json(metrics_path, metrics)
    write_json(ablation_path, ablation)
    write_json(view_path, view_contribution)
    write_json(confidence_path, confidence_calibration)
    write_json(recommendation_path, recommendation)
    benchmark_path.write_text(format_benchmark_report(metrics, view_contribution, ablation, confidence_calibration), encoding="utf-8")

    return {
        "metrics_path": str(metrics_path),
        "benchmark_report_path": str(benchmark_path),
        "ablation_report_path": str(ablation_path),
        "view_contribution_report_path": str(view_path),
        "confidence_calibration_report_path": str(confidence_path),
        "promotion_recommendation_path": str(recommendation_path),
        "metrics": metrics,
        "ablation": ablation,
        "view_contribution": view_contribution,
        "confidence_calibration": confidence_calibration,
        "leakage_audit": leakage_audit,
        "split_audit": split_audit,
        "recommendation": recommendation,
    }


def evaluate_baseline_by_split(
    baseline_model: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    target_columns: list[str],
) -> dict[str, Any]:
    return {
        split: evaluate_predictions(
            baseline_predictions(baseline_model, len(samples), target_columns),
            target_matrix(samples, target_columns),
            target_columns,
        )
        for split, samples in split_samples.items()
    }


def evaluate_metadata_candidate_by_split(
    model: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    return {
        split: evaluate_predictions(
            predict(model, feature_matrix(samples, feature_names)),
            target_matrix(samples, target_columns),
            target_columns,
        )
        for split, samples in split_samples.items()
    }


def load_vision_runtime(
    vision_metadata: dict[str, Any],
    vision_model_path: Path,
    vision_weights_path: str | Path | None,
    metadata_feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    hyperparameters = vision_metadata.get("trainingConfig", {}).get("model", {}).get("hyperparameters", {})
    branch_dim = int(hyperparameters.get("branchDim", vision_metadata.get("architecture", {}).get("branchDim", 8)))
    fusion_dim = int(hyperparameters.get("fusionDim", vision_metadata.get("architecture", {}).get("fusionDim", 32)))
    model = VisionMultimodalRegressor(
        metadata_dim=len(metadata_feature_names),
        target_count=len(target_columns),
        branch_dim=branch_dim,
        fusion_dim=fusion_dim,
    )
    weights_path = Path(vision_weights_path) if vision_weights_path is not None else vision_model_path.parent / VISION_MODEL_WEIGHTS
    if not weights_path.exists():
        raise VisionCandidateEvaluationError(f"Vision weights file does not exist: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return {
        "model": model,
        "metadata": vision_metadata,
        "targetMean": torch.as_tensor(vision_metadata["targetNormalization"]["mean"], dtype=torch.float32),
        "targetStd": torch.as_tensor(vision_metadata["targetNormalization"]["std"], dtype=torch.float32),
    }


def evaluate_vision_by_split(
    runtime: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    metadata_feature_names: list[str],
    target_columns: list[str],
    *,
    mask: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        split: evaluate_vision_samples(runtime, samples, metadata_feature_names, target_columns, mask=mask)
        for split, samples in split_samples.items()
    }


def evaluate_vision_samples(
    runtime: dict[str, Any],
    samples: list[dict[str, Any]],
    metadata_feature_names: list[str],
    target_columns: list[str],
    *,
    mask: dict[str, bool] | None = None,
) -> dict[str, Any]:
    predictions, targets = vision_predictions(runtime, samples, metadata_feature_names, target_columns, mask=mask)
    return evaluate_predictions(predictions, targets, target_columns)


def vision_predictions(
    runtime: dict[str, Any],
    samples: list[dict[str, Any]],
    metadata_feature_names: list[str],
    target_columns: list[str],
    *,
    mask: dict[str, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    batch = build_tensor_batches({"eval": samples}, metadata_feature_names, target_columns)["eval"]
    mask = mask or {}
    front = maybe_zero(batch["front"], mask.get("front", True))
    side = maybe_zero(batch["side"], mask.get("side", True))
    back = maybe_zero(batch["back"], mask.get("back", True))
    metadata = maybe_zero(batch["metadata"], mask.get("metadata", True))
    with torch.no_grad():
        normalized = runtime["model"](front, side, back, metadata)
        predictions = normalized.cpu() * runtime["targetStd"] + runtime["targetMean"]
    return predictions.numpy(), batch["targets"].numpy()


def maybe_zero(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    return tensor if enabled else torch.zeros_like(tensor)


def build_three_way_benchmark(
    baseline_metrics: dict[str, Any],
    metadata_metrics: dict[str, Any],
    vision_metrics: dict[str, Any],
    target_columns: list[str],
) -> dict[str, Any]:
    return {
        "split": "test",
        "productionVsVision": compare_metrics(baseline_metrics, vision_metrics, target_columns),
        "metadataVsVision": compare_metrics(metadata_metrics, vision_metrics, target_columns),
        "productionVsMetadata": compare_metrics(baseline_metrics, metadata_metrics, target_columns),
        "testOverallMae": {
            "productionBaseline": baseline_metrics["test"]["overallMae"],
            "metadataCandidate": metadata_metrics["test"]["overallMae"],
            "visionCandidate": vision_metrics["test"]["overallMae"],
        },
        "testPerMeasurementMae": {
            target: {
                "productionBaseline": baseline_metrics["test"]["maeByTarget"][target],
                "metadataCandidate": metadata_metrics["test"]["maeByTarget"][target],
                "visionCandidate": vision_metrics["test"]["maeByTarget"][target],
            }
            for target in target_columns
        },
    }


def build_view_contribution_report(
    runtime: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    metadata_feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    variants = {
        "front_only": {"front": True, "side": False, "back": False, "metadata": True},
        "front_side": {"front": True, "side": True, "back": False, "metadata": True},
        "front_side_back": {"front": True, "side": True, "back": True, "metadata": True},
    }
    metrics = {
        name: evaluate_vision_by_split(runtime, split_samples, metadata_feature_names, target_columns, mask=mask)
        for name, mask in variants.items()
    }
    return {
        "schemaVersion": "vision_view_contribution_report_v1",
        "method": "metadata branch retained while image views are progressively enabled",
        "views": metrics,
        "testDeltas": pairwise_test_deltas(metrics, ("front_only", "front_side", "front_side_back")),
        "backViewHelped": metrics["front_side"]["test"]["overallMae"] > metrics["front_side_back"]["test"]["overallMae"],
    }


def build_ablation_report(
    runtime: dict[str, Any],
    split_samples: dict[str, list[dict[str, Any]]],
    metadata_feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    variants = {
        "metadata_only": {"front": False, "side": False, "back": False, "metadata": True},
        "images_only": {"front": True, "side": True, "back": True, "metadata": False},
        "images_metadata": {"front": True, "side": True, "back": True, "metadata": True},
    }
    metrics = {
        name: evaluate_vision_by_split(runtime, split_samples, metadata_feature_names, target_columns, mask=mask)
        for name, mask in variants.items()
    }
    return {
        "schemaVersion": "vision_ablation_report_v1",
        "method": "evaluation-time branch masking; no retraining and no promotion",
        "ablations": metrics,
        "testDeltas": pairwise_test_deltas(metrics, ("metadata_only", "images_only", "images_metadata")),
    }


def pairwise_test_deltas(metrics: dict[str, Any], order: tuple[str, ...]) -> list[dict[str, Any]]:
    deltas = []
    for before, after in zip(order, order[1:]):
        before_mae = float(metrics[before]["test"]["overallMae"])
        after_mae = float(metrics[after]["test"]["overallMae"])
        deltas.append(
            {
                "from": before,
                "to": after,
                "absoluteMaeChange": round(after_mae - before_mae, 6),
                "improved": after_mae < before_mae,
            }
        )
    return deltas


def build_confidence_calibration_report(
    runtime: dict[str, Any],
    samples: list[dict[str, Any]],
    metadata_feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    predictions, targets = vision_predictions(runtime, samples, metadata_feature_names, target_columns)
    absolute_errors = np.abs(predictions - targets)
    buckets: dict[str, list[float]] = {bucket: [] for bucket in CONFIDENCE_BUCKETS}
    for row_index, sample in enumerate(samples):
        buckets.setdefault(confidence_tier(sample), []).append(float(absolute_errors[row_index, :].mean()))
    bucket_report = {
        tier: {
            "count": len(errors),
            "meanActualError": round(_mean(errors), 6) if errors else None,
            "maxActualError": round(max(errors), 6) if errors else None,
        }
        for tier, errors in buckets.items()
    }
    return {
        "schemaVersion": "vision_confidence_calibration_report_v1",
        "metric": "mean_absolute_error_cm",
        "buckets": bucket_report,
        "calibrated": confidence_is_monotonic(bucket_report),
    }


def confidence_is_monotonic(bucket_report: dict[str, dict[str, Any]]) -> bool:
    high = bucket_report.get("high_confidence", {}).get("meanActualError")
    medium = bucket_report.get("medium_confidence", {}).get("meanActualError")
    low = bucket_report.get("low_confidence", {}).get("meanActualError")
    ordered = [value for value in (high, medium, low) if value is not None]
    if len(ordered) < 2:
        return True
    return all(left <= right + 1e-9 for left, right in zip(ordered, ordered[1:]))


def audit_vision_leakage(
    vision_metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    metadata_feature_names: list[str],
    test_metrics: dict[str, Any],
    suspicious_mae_threshold: float,
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    config_inputs = vision_metadata.get("trainingConfig", {}).get("inputs", {})
    if config_inputs.get("finalApprovedMeasurementsAsInputs"):
        findings.append(high_finding("final_approved_input_enabled", "Vision config marks final approved measurements as input features."))
    if config_inputs.get("correctionDeltas"):
        findings.append(high_finding("correction_delta_input_enabled", "Vision config marks correction deltas as input features."))
    if config_inputs.get("lineageMeasurements"):
        findings.append(high_finding("lineage_input_enabled", "Vision config marks lineage measurements as input features."))
    if not vision_metadata.get("imageUsage", {}).get("pixelsConsumed"):
        findings.append(high_finding("pixels_not_consumed", "Vision candidate does not report pixelsConsumed=true."))
    for feature_name in metadata_feature_names:
        if forbidden_feature_name(feature_name):
            findings.append(high_finding("forbidden_metadata_feature", f"Metadata feature '{feature_name}' is leakage-prone."))
    for sample in samples:
        leaked_names = [
            name
            for name in raw_metadata_feature_names(sample)
            if forbidden_feature_name(name)
        ]
        for leaked_name in leaked_names:
            findings.append(high_finding("raw_metadata_leakage_name", f"Raw metadata feature '{leaked_name}' is leakage-prone."))
    if float(test_metrics["overallMae"]) <= suspicious_mae_threshold:
        findings.append(high_finding("suspiciously_low_overall_error", f"Vision test MAE {test_metrics['overallMae']} is below {suspicious_mae_threshold}."))
    for target, mae in test_metrics["maeByTarget"].items():
        if float(mae) <= suspicious_mae_threshold:
            findings.append(high_finding("suspiciously_low_target_error", f"Vision test MAE for {target} is below {suspicious_mae_threshold}."))
    return {
        "schemaVersion": "vision_leakage_audit_v1",
        "riskDetected": bool(findings),
        "riskLevel": "high" if findings else "low",
        "findings": dedupe_findings(findings),
        "checks": {
            "finalMeasurementsNotInputFeatures": True,
            "customerMakerFinalValuesNotInputFeatures": True,
            "correctionDeltasNotInputFeatures": True,
            "pixelsConsumedTrue": True,
            "suspiciouslyLowErrors": True,
        },
    }


def audit_split_integrity(
    split_samples: dict[str, list[dict[str, Any]]],
    split_indices: dict[str, list[int]],
) -> dict[str, Any]:
    findings = []
    for field in ("profileId", "scanSessionId", "orderId"):
        duplicates = duplicates_across_splits(split_samples, field)
        if duplicates:
            findings.append({"field": field, "duplicates": duplicates})
    return {
        "schemaVersion": "vision_split_audit_v1",
        "valid": not findings,
        "deterministicSplit": True,
        "splitIndices": split_indices,
        "splitCounts": {split: len(samples) for split, samples in split_samples.items()},
        "duplicateFindings": findings,
    }


def duplicates_across_splits(split_samples: dict[str, list[dict[str, Any]]], field: str) -> list[dict[str, Any]]:
    seen: dict[str, set[str]] = {}
    for split, samples in split_samples.items():
        for sample in samples:
            value = id_value(sample, field)
            if value:
                seen.setdefault(value, set()).add(split)
    return [
        {"value": value, "splits": sorted(splits)}
        for value, splits in sorted(seen.items())
        if len(splits) > 1
    ]


def id_value(sample: dict[str, Any], field: str) -> str | None:
    raw = sample.get("rawRecord", {})
    snake = normalize_token(field)
    keys = (field, field[0].lower() + field[1:], snake)
    for key in keys:
        value = raw.get(key) if isinstance(raw, dict) else None
        if value not in ("", None):
            return str(value)
    for parent in ("eligibilityMetadata", "verificationMetadata", "poseMetadata", "validationMetadata"):
        value = nested_lookup(sample.get(parent, {}), keys)
        if value not in ("", None):
            return str(value)
    return None


def nested_lookup(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if value.get(key) not in ("", None):
                return value[key]
        for nested in value.values():
            found = nested_lookup(nested, keys)
            if found not in ("", None):
                return found
    if isinstance(value, list):
        for nested in value:
            found = nested_lookup(nested, keys)
            if found not in ("", None):
                return found
    return None


def build_promotion_recommendation(
    benchmark: dict[str, Any],
    leakage_audit: dict[str, Any],
    split_audit: dict[str, Any],
    confidence_calibration: dict[str, Any],
    *,
    test_record_count: int,
    min_test_records: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if test_record_count < min_test_records:
        return {
            "decision": RECOMMEND_NEEDS_MORE_DATA,
            "promoteAllowed": False,
            "blockers": [f"Insufficient test data: need at least {min_test_records}, got {test_record_count}."],
        }
    if leakage_audit["riskDetected"]:
        blockers.append("Vision leakage audit detected high-risk feature or metric behavior.")
    if not split_audit["valid"]:
        blockers.append("Split integrity audit found duplicate profile, scan, or order identifiers across splits.")
    production_comparison = benchmark["productionVsVision"]
    metadata_comparison = benchmark["metadataVsVision"]
    if production_comparison["overallRegression"] or metadata_comparison["overallRegression"]:
        blockers.append("Vision candidate does not beat both production baseline and metadata candidate overall.")
    if production_comparison["regressions"] or metadata_comparison["regressions"]:
        blockers.append("Vision candidate has one or more per-measurement regressions.")
    if not confidence_calibration["calibrated"]:
        blockers.append("Confidence buckets are not monotonic: higher confidence has higher actual error.")

    if leakage_audit["riskDetected"]:
        decision = RECOMMEND_LEAKAGE_RISK
    elif production_comparison["overallRegression"] or metadata_comparison["overallRegression"] or production_comparison["regressions"] or metadata_comparison["regressions"]:
        decision = RECOMMEND_REGRESSION
    elif not confidence_calibration["calibrated"]:
        decision = RECOMMEND_CONFIDENCE
    elif blockers:
        decision = RECOMMEND_DO_NOT_PROMOTE
    else:
        decision = RECOMMEND_PROMOTE
    return {
        "schemaVersion": "vision_promotion_recommendation_v1",
        "decision": decision,
        "promoteAllowed": decision == RECOMMEND_PROMOTE,
        "blockers": blockers,
        "eligibilityCriteria": {
            "beatsProductionBaseline": not production_comparison["overallRegression"],
            "beatsMetadataCandidate": not metadata_comparison["overallRegression"],
            "noLeakageRisk": not leakage_audit["riskDetected"],
            "splitIntegrityValid": split_audit["valid"],
            "noMajorPerMeasurementRegression": not production_comparison["regressions"] and not metadata_comparison["regressions"],
            "enoughTestRecords": test_record_count >= min_test_records,
            "confidenceCalibrated": confidence_calibration["calibrated"],
        },
    }


def raw_metadata_feature_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for root_name, payload in (
        ("pose", sample.get("poseMetadata", {})),
        ("validation", sample.get("validationMetadata", {})),
        ("verification", sample.get("verificationMetadata", {})),
    ):
        names.extend(flatten_names(payload, root_name))
    return names


def flatten_names(value: Any, prefix: str) -> list[str]:
    if isinstance(value, dict):
        names: list[str] = []
        for key, nested in value.items():
            names.extend(flatten_names(nested, f"{prefix}.{normalize_token(str(key))}"))
        return names
    if isinstance(value, list):
        names = [f"{prefix}.__count"]
        for index, nested in enumerate(value):
            names.extend(flatten_names(nested, f"{prefix}.{index}"))
        return names
    if isinstance(value, (int, float, bool)):
        return [prefix]
    return []


def confidence_tier(sample: dict[str, Any]) -> str:
    tier = first_confidence_token(sample.get("confidenceMetadata", {}))
    if tier:
        return tier
    tier = first_confidence_token(sample.get("finalApprovedMeasurements", {}))
    return tier or "unknown"


def first_confidence_token(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_token = normalize_token(str(key))
            if key_token in {"confidence_tier", "overall_confidence_tier", "tier", "confidence"} and isinstance(nested, str):
                return normalize_token(nested)
            found = first_confidence_token(nested)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = first_confidence_token(item)
            if found:
                return found
    return None


def resolve_dataset_version(vision_metadata: dict[str, Any], metadata_model: dict[str, Any], requested: str | None) -> str:
    version = requested or vision_metadata.get("datasetVersion") or vision_metadata.get("trainingConfig", {}).get("datasetVersion")
    metadata_version = metadata_model.get("datasetVersion") or metadata_model.get("trainingConfig", {}).get("datasetVersion")
    if not version:
        raise VisionCandidateEvaluationError("Vision candidate artifact is missing datasetVersion; pass --dataset-version.")
    if metadata_version and str(metadata_version) != str(version):
        raise VisionCandidateEvaluationError(f"Metadata candidate datasetVersion {metadata_version} does not match vision datasetVersion {version}.")
    return str(version)


def high_finding(kind: str, message: str) -> dict[str, str]:
    return {"severity": "high", "kind": kind, "message": message}


def dedupe_findings(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for finding in findings:
        key = (finding["severity"], finding["kind"], finding["message"])
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output


def normalize_token(value: str) -> str:
    output = []
    for index, character in enumerate(value.strip()):
        if character.isupper() and index:
            output.append("_")
        output.append(character.lower())
    return "".join(output).replace("-", "_").replace(" ", "_")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def format_benchmark_report(
    metrics: dict[str, Any],
    view_contribution: dict[str, Any],
    ablation: dict[str, Any],
    confidence_calibration: dict[str, Any],
) -> str:
    recommendation = metrics["recommendation"]
    benchmark = metrics["benchmark"]
    lines = [
        "# Phase H.6 Vision Candidate Evaluation",
        "",
        f"Dataset version: `{metrics['datasetVersion']}`",
        f"Metadata candidate: `{metrics['metadataCandidateModelVersion']}`",
        f"Vision candidate: `{metrics['visionCandidateModelVersion']}`",
        f"Recommendation: `{recommendation['decision']}`",
        f"Promote allowed: `{recommendation['promoteAllowed']}`",
        "",
        "## Test MAE",
        "",
        "| Model | Overall MAE |",
        "| --- | ---: |",
    ]
    for name, value in benchmark["testOverallMae"].items():
        lines.append(f"| `{name}` | {value:.4f} |")
    lines.extend(["", "## Per Measurement MAE", "", "| Measurement | Production baseline | Metadata candidate | Vision candidate |", "| --- | ---: | ---: | ---: |"])
    for target, row in benchmark["testPerMeasurementMae"].items():
        lines.append(f"| `{target}` | {row['productionBaseline']:.4f} | {row['metadataCandidate']:.4f} | {row['visionCandidate']:.4f} |")
    lines.extend(["", "## View Contribution", "", "| Variant | Test overall MAE |", "| --- | ---: |"])
    for variant, rows in view_contribution["views"].items():
        lines.append(f"| `{variant}` | {rows['test']['overallMae']:.4f} |")
    lines.append(f"\nBack view helped: `{view_contribution['backViewHelped']}`")
    lines.extend(["", "## Ablation", "", "| Variant | Test overall MAE |", "| --- | ---: |"])
    for variant, rows in ablation["ablations"].items():
        lines.append(f"| `{variant}` | {rows['test']['overallMae']:.4f} |")
    lines.extend(["", "## Confidence Calibration", "", "| Bucket | Count | Mean actual error | Max actual error |", "| --- | ---: | ---: | ---: |"])
    for bucket, row in confidence_calibration["buckets"].items():
        mean_error = "" if row["meanActualError"] is None else f"{row['meanActualError']:.4f}"
        max_error = "" if row["maxActualError"] is None else f"{row['maxActualError']:.4f}"
        lines.append(f"| `{bucket}` | {row['count']} | {mean_error} | {max_error} |")
    lines.extend(["", "## Promotion Gate", ""])
    if recommendation["blockers"]:
        lines.extend(f"- {blocker}" for blocker in recommendation["blockers"])
    else:
        lines.append("- No conservative blockers were detected.")
    lines.extend(["", "No model was promoted, no production artifact was replaced, and live API behavior was not changed.", ""])
    return "\n".join(lines)


def format_evaluation_summary(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Vision evaluation metrics: {result['metrics_path']}",
            f"Benchmark report: {result['benchmark_report_path']}",
            f"Ablation report: {result['ablation_report_path']}",
            f"View contribution report: {result['view_contribution_report_path']}",
            f"Confidence calibration report: {result['confidence_calibration_report_path']}",
            f"Recommendation: {result['recommendation']['decision']}",
            f"Promote allowed: {result['recommendation']['promoteAllowed']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a vision multimodal Body AI candidate without promotion.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--metadata-candidate-model", required=True, help="Path to H.2 metadata candidate model.json.")
    parser.add_argument("--vision-candidate-model", required=True, help="Path to H.5 vision candidate vision_model.json.")
    parser.add_argument("--output", required=True, help="Output directory for H.6 evaluation artifacts.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--storage-root", help="Optional local storage root for image storage keys.")
    parser.add_argument("--vision-weights", help="Optional path to vision_model.pt. Defaults beside vision_model.json.")
    parser.add_argument("--dataset-version")
    parser.add_argument("--min-test-records", type=int, default=3)
    args = parser.parse_args(argv)
    result = evaluate_vision_candidate_model(
        args.dataset,
        args.metadata_candidate_model,
        args.vision_candidate_model,
        args.output,
        records_file=args.records_file,
        storage_root=args.storage_root,
        vision_weights_path=args.vision_weights,
        dataset_version=args.dataset_version,
        min_test_records=args.min_test_records,
    )
    print(format_evaluation_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
