from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.datasets.verified_measurement_dataset import VerifiedMeasurementDatasetLoader
from training.train_baseline_measurements import _mean
from training.train_candidate_model import (
    DEFAULT_TARGET_COLUMNS,
    REGISTRY_FILENAME,
    feature_matrix,
    measurement_value,
    predict,
    split_indices_for_count,
    target_matrix,
)


EVALUATION_METRICS_FILENAME = "candidate_evaluation_metrics.json"
LEAKAGE_AUDIT_FILENAME = "leakage_audit.json"
SPLIT_AUDIT_FILENAME = "split_audit.json"
REPORT_FILENAME = "candidate_evaluation_report.md"
CONFIDENCE_BUCKETS = ("high_confidence", "medium_confidence", "low_confidence", "unknown")
RECOMMEND_PROMOTE = "promote"
RECOMMEND_DO_NOT_PROMOTE = "do_not_promote"
RECOMMEND_NEEDS_MORE_DATA = "needs_more_data"
RECOMMEND_LEAKAGE_RISK = "leakage_risk"
RECOMMEND_REGRESSION = "regression_detected"


class CandidateEvaluationError(ValueError):
    pass


def evaluate_candidate_model(
    dataset_root: str | Path,
    candidate_model_path: str | Path,
    output_dir: str | Path,
    *,
    records_file: str | Path | None = None,
    dataset_version: str | None = None,
    registry_path: str | Path | None = None,
    min_test_records: int = 3,
    suspicious_mae_threshold: float = 0.01,
    generated_at: str | None = None,
) -> dict[str, Any]:
    model_path = Path(candidate_model_path)
    model = _read_json(model_path)
    config = model.get("trainingConfig", {})
    selected_version = dataset_version or str(model.get("datasetVersion") or config.get("datasetVersion") or "")
    if not selected_version:
        raise CandidateEvaluationError("Candidate model is missing datasetVersion; pass --dataset-version.")

    loader = VerifiedMeasurementDatasetLoader(dataset_root, records_file)
    samples = [sample for sample in loader if sample["dataset_version"] == selected_version]
    if not samples:
        raise CandidateEvaluationError(f"No verified records found for dataset version '{selected_version}'.")

    target_columns = list(model.get("targetColumns") or config.get("targetColumns") or DEFAULT_TARGET_COLUMNS)
    feature_names = list(model.get("featureNames") or config.get("featurePipeline", {}).get("featureNames") or [])
    if not feature_names:
        raise CandidateEvaluationError("Candidate model is missing featureNames.")

    split_policy = config.get("splitPolicy", {})
    split_indices = split_indices_for_count(
        len(samples),
        int(config.get("randomSeed", 42)),
        float(split_policy.get("valSize", 0.2)),
        float(split_policy.get("testSize", 0.2)),
    )
    split_samples = {split: [samples[index] for index in indices] for split, indices in split_indices.items()}

    candidate_metrics = evaluate_candidate_by_split(model, split_samples, feature_names, target_columns)
    baseline_model = train_baseline_mean_estimator(split_samples["train"], target_columns)
    baseline_metrics = evaluate_baseline_by_split(baseline_model, split_samples, target_columns)
    comparison = compare_metrics(baseline_metrics, candidate_metrics, target_columns)
    confidence_calibration = confidence_calibration_analysis(model, split_samples["test"], feature_names, target_columns)
    leakage_audit = audit_leakage(model, samples, split_samples, feature_names, target_columns, suspicious_mae_threshold)
    split_audit = audit_split_integrity(split_samples, split_indices)
    compatibility = compatibility_disclaimer(model)
    registry = load_candidate_registry(registry_path or model_path.parent / REGISTRY_FILENAME)
    recommendation = build_recommendation(
        comparison,
        leakage_audit,
        split_audit,
        test_record_count=len(split_samples["test"]),
        min_test_records=min_test_records,
    )

    metrics = {
        "schemaVersion": "candidate_evaluation_metrics_v1",
        "generatedAt": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": str(dataset_root),
        "recordsFile": str(loader.records_path),
        "datasetVersion": selected_version,
        "candidateModelPath": str(model_path),
        "candidateModelVersion": model.get("modelVersion"),
        "targetColumns": target_columns,
        "baselineEstimator": baseline_model["modelFamily"],
        "baselineMetrics": baseline_metrics,
        "candidateMetrics": candidate_metrics,
        "comparison": comparison,
        "confidenceCalibration": confidence_calibration,
        "compatibilityMode": compatibility,
        "recommendation": recommendation,
        "candidateRegistry": registry,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / EVALUATION_METRICS_FILENAME
    leakage_path = output_path / LEAKAGE_AUDIT_FILENAME
    split_path = output_path / SPLIT_AUDIT_FILENAME
    report_path = output_path / REPORT_FILENAME
    _write_json(metrics_path, metrics)
    _write_json(leakage_path, leakage_audit)
    _write_json(split_path, split_audit)
    report_path.write_text(format_evaluation_report(metrics, leakage_audit, split_audit), encoding="utf-8")

    return {
        "metrics_path": str(metrics_path),
        "leakage_audit_path": str(leakage_path),
        "split_audit_path": str(split_path),
        "report_path": str(report_path),
        "metrics": metrics,
        "leakage_audit": leakage_audit,
        "split_audit": split_audit,
        "recommendation": recommendation,
    }


def evaluate_candidate_by_split(
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


def train_baseline_mean_estimator(samples: list[dict[str, Any]], target_columns: list[str]) -> dict[str, Any]:
    if not samples:
        raise CandidateEvaluationError("Cannot build baseline estimator without training samples.")
    means: dict[str, float] = {}
    for target in target_columns:
        values = [measurement_value(sample["final_approved_measurements"].get(target), sample["sample_id"], target) for sample in samples]
        means[target] = _mean(values)
    return {
        "modelFamily": "verified_train_split_mean_baseline",
        "targetColumns": target_columns,
        "targetMeans": means,
        "productionStatus": "baseline_comparison_only",
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


def baseline_predictions(baseline_model: dict[str, Any], count: int, target_columns: list[str]) -> np.ndarray:
    row = [float(baseline_model["targetMeans"][target]) for target in target_columns]
    return np.asarray([row for _index in range(count)], dtype=np.float64)


def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray, target_columns: list[str]) -> dict[str, Any]:
    absolute_errors = np.abs(predictions - targets)
    mae_by_target = {
        target: round(float(absolute_errors[:, index].mean()), 6)
        for index, target in enumerate(target_columns)
    }
    return {
        "overallMae": round(_mean(list(mae_by_target.values())), 6),
        "maeByTarget": mae_by_target,
    }


def compare_metrics(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    target_columns: list[str],
    split: str = "test",
) -> dict[str, Any]:
    baseline = baseline_metrics[split]
    candidate = candidate_metrics[split]
    baseline_overall = float(baseline["overallMae"])
    candidate_overall = float(candidate["overallMae"])
    overall_improvement = baseline_overall - candidate_overall
    per_target: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []
    for target in target_columns:
        baseline_mae = float(baseline["maeByTarget"][target])
        candidate_mae = float(candidate["maeByTarget"][target])
        improvement = baseline_mae - candidate_mae
        row = {
            "baselineMae": baseline_mae,
            "candidateMae": candidate_mae,
            "absoluteImprovement": round(improvement, 6),
            "percentageImprovement": round((improvement / baseline_mae) * 100, 6) if baseline_mae else 0.0,
            "regression": improvement < 0,
        }
        per_target[target] = row
        if row["regression"]:
            regressions.append({"target": target, **row})
    return {
        "split": split,
        "baselineOverallMae": baseline_overall,
        "candidateOverallMae": candidate_overall,
        "absoluteImprovement": round(overall_improvement, 6),
        "percentageImprovement": round((overall_improvement / baseline_overall) * 100, 6) if baseline_overall else 0.0,
        "overallRegression": overall_improvement < 0,
        "perTarget": per_target,
        "regressions": regressions,
    }


def confidence_calibration_analysis(
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    feature_names: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    predictions = predict(model, feature_matrix(samples, feature_names))
    targets = target_matrix(samples, target_columns)
    absolute_errors = np.abs(predictions - targets)
    buckets: dict[str, list[float]] = {bucket: [] for bucket in CONFIDENCE_BUCKETS}
    for row_index, sample in enumerate(samples):
        tier = confidence_tier(sample)
        buckets.setdefault(tier, [])
        buckets[tier].append(float(absolute_errors[row_index, :].mean()))
    return {
        tier: {
            "count": len(errors),
            "meanAbsoluteError": round(_mean(errors), 6) if errors else None,
            "maxAbsoluteError": round(max(errors), 6) if errors else None,
        }
        for tier, errors in buckets.items()
    }


def audit_leakage(
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    split_samples: dict[str, list[dict[str, Any]]],
    feature_names: list[str],
    target_columns: list[str],
    suspicious_mae_threshold: float,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    feature_tokens = {feature: normalize_token(feature) for feature in feature_names}
    for feature, normalized in feature_tokens.items():
        if any(token in normalized for token in ("final_approved", "finalapproved", "final_measurement", "approved_measurement")):
            findings.append(high_finding("final_approved_feature", f"Feature '{feature}' appears to include final approved measurements."))
        if any(token in normalized for token in ("customer_edit", "customer_measurement", "maker_adjustment", "maker_measurement")):
            findings.append(high_finding("lineage_measurement_feature", f"Feature '{feature}' appears to include customer or maker measurement lineage."))
        if normalized.startswith("correction_delta") or ".correction_delta." in normalized:
            findings.append(medium_finding("correction_delta_feature", f"Feature '{feature}' uses correction deltas; evaluate availability and leakage risk."))

    feature_values = feature_matrix(samples, feature_names)
    targets = target_matrix(samples, target_columns)
    for feature_index, feature in enumerate(feature_names):
        values = feature_values[:, feature_index]
        for target_index, target in enumerate(target_columns):
            target_values = targets[:, target_index]
            if values.shape[0] and np.allclose(values, target_values, atol=1e-9, rtol=0.0):
                findings.append(high_finding("feature_equals_target", f"Feature '{feature}' exactly matches target '{target}'."))
            if values.shape[0] > 1 and float(np.std(values)) > 0 and float(np.std(target_values)) > 0:
                correlation = float(np.corrcoef(values, target_values)[0, 1])
                if abs(correlation) >= 0.999:
                    findings.append(high_finding("near_perfect_feature_target_correlation", f"Feature '{feature}' has near-perfect correlation with target '{target}' ({correlation:.6f})."))

    test_metrics = evaluate_candidate_by_split(model, {"test": split_samples["test"]}, feature_names, target_columns)["test"]
    if float(test_metrics["overallMae"]) <= suspicious_mae_threshold:
        findings.append(high_finding("suspiciously_low_overall_error", f"Test overall MAE {test_metrics['overallMae']} is below suspicious threshold {suspicious_mae_threshold}."))
    for target, mae in test_metrics["maeByTarget"].items():
        if float(mae) <= suspicious_mae_threshold:
            findings.append(high_finding("suspiciously_low_target_error", f"Test MAE for {target} is {mae}, below suspicious threshold {suspicious_mae_threshold}."))

    return {
        "schemaVersion": "candidate_leakage_audit_v1",
        "riskDetected": any(finding["severity"] in {"high", "critical"} for finding in findings),
        "riskLevel": risk_level(findings),
        "findings": dedupe_findings(findings),
        "checks": {
            "finalApprovedMeasurementsAsFeatures": True,
            "correctionDeltasEncodeTargets": True,
            "customerMakerFinalLineageFeatureNames": True,
            "suspiciouslyLowErrors": True,
        },
    }


def audit_split_integrity(
    split_samples: dict[str, list[dict[str, Any]]],
    split_indices: dict[str, list[int]],
) -> dict[str, Any]:
    duplicate_findings: list[dict[str, Any]] = []
    for field in ("profileId", "scanSessionId", "orderId"):
        duplicates = duplicates_across_splits(split_samples, field)
        if duplicates:
            duplicate_findings.append({"field": field, "duplicates": duplicates})
    return {
        "schemaVersion": "candidate_split_audit_v1",
        "valid": not duplicate_findings,
        "deterministicSplit": True,
        "splitIndices": split_indices,
        "splitCounts": {split: len(samples) for split, samples in split_samples.items()},
        "duplicateFindings": duplicate_findings,
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
    raw = sample.get("raw_record", {})
    snake = normalize_token(field)
    candidates = (
        field,
        field[0].lower() + field[1:],
        snake,
        snake.replace("_id", "_id"),
    )
    for key in candidates:
        value = raw.get(key)
        if value not in ("", None):
            return str(value)
    for parent in ("eligibility_metadata", "verification_metadata_summary", "pose_metadata_summary", "validation_metadata_summary"):
        payload = sample.get(parent, {})
        for key in candidates:
            value = payload.get(key) if isinstance(payload, dict) else None
            if value not in ("", None):
                return str(value)
    return None


def build_recommendation(
    comparison: dict[str, Any],
    leakage_audit: dict[str, Any],
    split_audit: dict[str, Any],
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
        blockers.append(f"Leakage risk detected: {leakage_audit['riskLevel']}.")
    if not split_audit["valid"]:
        blockers.append("Split integrity audit found duplicate profile, scan, or order identifiers across splits.")
    if comparison["overallRegression"] or comparison["regressions"]:
        blockers.append("Candidate regressed overall or per-measurement MAE versus baseline.")
    if leakage_audit["riskDetected"]:
        decision = RECOMMEND_LEAKAGE_RISK
    elif comparison["overallRegression"] or comparison["regressions"]:
        decision = RECOMMEND_REGRESSION
    elif blockers:
        decision = RECOMMEND_DO_NOT_PROMOTE
    else:
        decision = RECOMMEND_PROMOTE
    return {
        "decision": decision,
        "promoteAllowed": decision == RECOMMEND_PROMOTE,
        "blockers": blockers,
    }


def compatibility_disclaimer(model: dict[str, Any]) -> dict[str, Any]:
    image_usage = model.get("imageUsage", {})
    return {
        "pixelsConsumed": bool(image_usage.get("pixelsConsumed", False)),
        "backImageAccepted": True,
        "backImagePixelWeighted": False,
        "imageLearningImplemented": False,
        "disclaimer": (
            "Phase H.3 evaluates a compatibility-mode candidate. Front, side, and back image references are "
            "validated, but image pixels are not consumed and the back image is not yet pixel-weighted."
        ),
    }


def load_candidate_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"available": False, "path": str(registry_path), "candidates": []}
    payload = _read_json(registry_path)
    if not isinstance(payload, dict):
        raise CandidateEvaluationError(f"Candidate registry must be a JSON object: {registry_path}")
    return {"available": True, "path": str(registry_path), **payload}


def confidence_tier(sample: dict[str, Any]) -> str:
    tier = first_confidence_token(sample.get("confidence_metadata", {}))
    if tier:
        return tier
    tier = first_confidence_token(sample.get("final_approved_measurements", {}))
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


def high_finding(kind: str, message: str) -> dict[str, str]:
    return {"severity": "high", "kind": kind, "message": message}


def medium_finding(kind: str, message: str) -> dict[str, str]:
    return {"severity": "medium", "kind": kind, "message": message}


def risk_level(findings: list[dict[str, Any]]) -> str:
    severities = {finding["severity"] for finding in findings}
    if "critical" in severities or "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for finding in findings:
        key = (str(finding["severity"]), str(finding["kind"]), str(finding["message"]))
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output


def normalize_token(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def format_evaluation_report(metrics: dict[str, Any], leakage_audit: dict[str, Any], split_audit: dict[str, Any]) -> str:
    comparison = metrics["comparison"]
    recommendation = metrics["recommendation"]
    lines = [
        "# Phase H.3 Candidate Model Evaluation",
        "",
        f"Candidate model: `{metrics['candidateModelVersion']}`",
        f"Dataset version: `{metrics['datasetVersion']}`",
        f"Recommendation: `{recommendation['decision']}`",
        f"Promote allowed: `{recommendation['promoteAllowed']}`",
        "",
        "## Baseline Comparison",
        "",
        f"Baseline overall MAE: `{comparison['baselineOverallMae']:.4f}`",
        f"Candidate overall MAE: `{comparison['candidateOverallMae']:.4f}`",
        f"Absolute improvement: `{comparison['absoluteImprovement']:.4f}`",
        f"Percentage improvement: `{comparison['percentageImprovement']:.2f}%`",
        "",
        "| Measurement | Baseline MAE | Candidate MAE | Improvement | Regression |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for target, row in comparison["perTarget"].items():
        lines.append(
            f"| `{target}` | {row['baselineMae']:.4f} | {row['candidateMae']:.4f} | "
            f"{row['absoluteImprovement']:.4f} | `{row['regression']}` |"
        )
    lines.extend(["", "## Confidence Calibration", "", "| Bucket | Count | Mean actual error | Max actual error |", "| --- | ---: | ---: | ---: |"])
    for tier, row in metrics["confidenceCalibration"].items():
        mean_error = "" if row["meanAbsoluteError"] is None else f"{row['meanAbsoluteError']:.4f}"
        max_error = "" if row["maxAbsoluteError"] is None else f"{row['maxAbsoluteError']:.4f}"
        lines.append(f"| `{tier}` | {row['count']} | {mean_error} | {max_error} |")
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            f"Risk level: `{leakage_audit['riskLevel']}`",
            f"Risk detected: `{leakage_audit['riskDetected']}`",
        ]
    )
    lines.extend(f"- {finding['severity']}: {finding['message']}" for finding in leakage_audit["findings"])
    lines.extend(
        [
            "",
            "## Split Audit",
            "",
            f"Valid: `{split_audit['valid']}`",
            f"Split counts: `{split_audit['splitCounts']}`",
            "",
            "## Compatibility Disclaimer",
            "",
            metrics["compatibilityMode"]["disclaimer"],
            "",
            "## Recommendation Logic",
            "",
        ]
    )
    if recommendation["blockers"]:
        lines.extend(f"- {blocker}" for blocker in recommendation["blockers"])
    else:
        lines.append("- No conservative blockers were detected by this evaluation harness.")
    lines.extend(
        [
            "",
            "No model was promoted, no production artifact was replaced, and no live API behavior was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def format_evaluation_summary(result: dict[str, Any]) -> str:
    recommendation = result["recommendation"]
    return "\n".join(
        [
            f"Evaluation metrics: {result['metrics_path']}",
            f"Leakage audit: {result['leakage_audit_path']}",
            f"Split audit: {result['split_audit_path']}",
            f"Report: {result['report_path']}",
            f"Recommendation: {recommendation['decision']}",
            f"Promote allowed: {recommendation['promoteAllowed']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate Body AI model artifacts without promotion.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--candidate-model", required=True, help="Path to candidate model.json.")
    parser.add_argument("--output", required=True, help="Output directory for evaluation artifacts.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--dataset-version", help="Verified dataset version to evaluate.")
    parser.add_argument("--registry", help="Candidate registry path. Defaults to model directory candidate_model_registry.json.")
    parser.add_argument("--min-test-records", type=int, default=3)
    args = parser.parse_args(argv)
    result = evaluate_candidate_model(
        args.dataset,
        args.candidate_model,
        args.output,
        records_file=args.records_file,
        dataset_version=args.dataset_version,
        registry_path=args.registry,
        min_test_records=args.min_test_records,
    )
    print(format_evaluation_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
