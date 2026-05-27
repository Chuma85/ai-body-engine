from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.measurements.measurement_confidence import (
    DEFAULT_PHASE4D_PREDICTIONS,
    DEFAULT_RUN_NAME,
    apply_confidence_policy,
    load_prediction_rows,
    product_action_for_tier,
)

UNCERTAINTY_POLICY_JSON = "uncertainty_policy.json"
UNCERTAINTY_EVAL_JSON = "uncertainty_eval_results.json"
UNCERTAINTY_EVAL_CSV = "uncertainty_eval_results.csv"
PER_TARGET_SUMMARY_CSV = "per_target_uncertainty_summary.csv"
COVERAGE_SUMMARY_MD = "coverage_summary.md"
PRODUCT_ACTION_POLICY_MD = "product_action_policy.md"

CALIBRATION_SPLITS = {"train", "val"}
EVALUATION_SPLIT = "test"
TIERS = ["high_confidence", "medium_confidence", "low_confidence"]
TARGETS = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm"]
MIN_TIER_TARGET_ROWS = 20
INTERVAL_SAFETY_MULTIPLIER = 1.35


def calibrate_measurement_uncertainty(
    predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
    output_dir: str | Path = "artifacts/phase_4f_measurement_uncertainty_calibration",
    run_name: str = DEFAULT_RUN_NAME,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [apply_confidence_policy(row) for row in load_prediction_rows(predictions_csv) if row["run_name"] == run_name]
    if not rows:
        raise ValueError(f"No prediction rows found for run '{run_name}' in {predictions_csv}")
    calibration_rows = [row for row in rows if row["dataset_split"] in CALIBRATION_SPLITS]
    test_rows = [row for row in rows if row["dataset_split"] == EVALUATION_SPLIT]
    if not calibration_rows or not test_rows:
        raise ValueError("Need train/val rows for calibration and test rows for evaluation.")

    policy = fit_uncertainty_policy(calibration_rows)
    evaluated_rows = [apply_uncertainty_policy(row, policy) for row in rows]
    test_evaluated = [row for row in evaluated_rows if row["dataset_split"] == EVALUATION_SPLIT]
    overall_rows = build_coverage_rows(test_evaluated)
    per_target_rows = build_per_target_rows(test_evaluated)
    summary = {
        "predictions_csv": str(predictions_csv),
        "run_name": run_name,
        "calibration_splits": sorted(CALIBRATION_SPLITS),
        "evaluation_split": EVALUATION_SPLIT,
        "policy": policy,
        "overall": overall_rows,
        "per_target": per_target_rows,
        "interpretation": interpretation(overall_rows),
    }

    paths = {
        "uncertainty_policy_json": output_path / UNCERTAINTY_POLICY_JSON,
        "uncertainty_eval_json": output_path / UNCERTAINTY_EVAL_JSON,
        "uncertainty_eval_csv": output_path / UNCERTAINTY_EVAL_CSV,
        "per_target_summary_csv": output_path / PER_TARGET_SUMMARY_CSV,
        "coverage_summary_md": output_path / COVERAGE_SUMMARY_MD,
        "product_action_policy_md": output_path / PRODUCT_ACTION_POLICY_MD,
    }
    write_json(paths["uncertainty_policy_json"], policy)
    write_json(paths["uncertainty_eval_json"], summary)
    write_csv(paths["uncertainty_eval_csv"], overall_rows, coverage_fieldnames())
    write_csv(paths["per_target_summary_csv"], per_target_rows, coverage_fieldnames())
    paths["coverage_summary_md"].write_text(format_coverage_summary(summary), encoding="utf-8")
    paths["product_action_policy_md"].write_text(format_product_action_policy(policy), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def fit_uncertainty_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_error_stats = {
        target: error_stats([row for row in rows if row["target"] == target])
        for target in sorted({row["target"] for row in rows})
    }
    tier_error_stats = {
        tier: error_stats([row for row in rows if row["confidence_tier"] == tier])
        for tier in TIERS
    }
    target_tier_error_stats: dict[str, dict[str, Any]] = {}
    for target in target_error_stats:
        target_tier_error_stats[target] = {}
        for tier in TIERS:
            tier_rows = [row for row in rows if row["target"] == target and row["confidence_tier"] == tier]
            target_tier_error_stats[target][tier] = error_stats(tier_rows)
    return {
        "method": "train_val_empirical_abs_error_quantiles",
        "interval_safety_multiplier": INTERVAL_SAFETY_MULTIPLIER,
        "calibration_splits": sorted(CALIBRATION_SPLITS),
        "evaluation_split": EVALUATION_SPLIT,
        "min_tier_target_rows": MIN_TIER_TARGET_ROWS,
        "target_error_stats": target_error_stats,
        "tier_error_stats": tier_error_stats,
        "target_tier_error_stats": target_tier_error_stats,
        "fallback_order": ["target_tier_p90", "target_p90", "tier_p90", "global_p90"],
        "global_error_stats": error_stats(rows),
    }


def error_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["abs_error_cm"]) for row in rows], dtype=np.float64)
    if errors.size == 0:
        return {"count": 0, "mae": 0.0, "p50": 0.0, "p80": 0.0, "p90": 0.0}
    return {
        "count": int(errors.size),
        "mae": float(errors.mean()),
        "p50": float(np.percentile(errors, 50.0)),
        "p80": float(np.percentile(errors, 80.0)),
        "p90": float(np.percentile(errors, 90.0)),
    }


def apply_uncertainty_policy(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    estimated_error, source = estimated_error_for_row(row, policy)
    final_estimate = float(row["final_estimate_cm"])
    low = final_estimate - estimated_error
    high = final_estimate + estimated_error
    true_value = float(row["calibrated_label_cm"])
    return {
        **row,
        "estimated_error_cm": estimated_error,
        "prediction_interval_low_cm": low,
        "prediction_interval_high_cm": high,
        "interval_source": source,
        "covered_by_interval": low <= true_value <= high,
        "uncertainty_product_action": uncertainty_product_action(row["confidence_tier"], estimated_error),
    }


def estimated_error_for_row(row: dict[str, Any], policy: dict[str, Any]) -> tuple[float, str]:
    target = row["target"]
    tier = row["confidence_tier"]
    target_tier = policy["target_tier_error_stats"].get(target, {}).get(tier, {})
    if int(target_tier.get("count", 0)) >= MIN_TIER_TARGET_ROWS:
        return apply_interval_safety(float(target_tier["p90"]), row, policy), "target_tier_p90"
    target_stats = policy["target_error_stats"].get(target, {})
    if int(target_stats.get("count", 0)) > 0:
        return apply_interval_safety(float(target_stats["p90"]), row, policy), "target_p90"
    tier_stats = policy["tier_error_stats"].get(tier, {})
    if int(tier_stats.get("count", 0)) > 0:
        return apply_interval_safety(float(tier_stats["p90"]), row, policy), "tier_p90"
    return apply_interval_safety(float(policy["global_error_stats"]["p90"]), row, policy), "global_p90"


def apply_interval_safety(base_error: float, row: dict[str, Any], policy: dict[str, Any]) -> float:
    multiplier = float(policy.get("interval_safety_multiplier", INTERVAL_SAFETY_MULTIPLIER))
    tier_multiplier = {
        "high_confidence": 1.0,
        "medium_confidence": 1.1,
        "low_confidence": 1.25,
    }.get(str(row.get("confidence_tier")), 1.0)
    return float(base_error) * multiplier * tier_multiplier


def uncertainty_product_action(confidence_tier: str, estimated_error_cm: float) -> str:
    if confidence_tier == "low_confidence" or estimated_error_cm > 5.0:
        return "request_retake_or_tape_measurement"
    if confidence_tier == "medium_confidence" or estimated_error_cm > 3.0:
        return "require_manual_confirmation"
    return product_action_for_tier(confidence_tier)


def build_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [coverage_summary_row("all_targets", "all_confidence", rows)]
    for tier in TIERS:
        output.append(coverage_summary_row("all_targets", tier, [row for row in rows if row["confidence_tier"] == tier]))
    return output


def build_per_target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        output.append(coverage_summary_row(target, "all_confidence", target_rows))
        for tier in TIERS:
            output.append(coverage_summary_row(target, tier, [row for row in target_rows if row["confidence_tier"] == tier]))
    return output


def coverage_summary_row(target: str, confidence_tier: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["abs_error_cm"]) for row in rows], dtype=np.float64)
    intervals = np.asarray([float(row["estimated_error_cm"]) for row in rows], dtype=np.float64)
    coverage = np.asarray([bool(row["covered_by_interval"]) for row in rows], dtype=bool)
    return {
        "target": target,
        "confidence_tier": confidence_tier,
        "prediction_count": len(rows),
        "coverage": 0.0 if coverage.size == 0 else float(coverage.mean()),
        "mae_cm": safe_mean(errors),
        "p50_abs_error_cm": safe_percentile(errors, 50.0),
        "p80_abs_error_cm": safe_percentile(errors, 80.0),
        "p90_abs_error_cm": safe_percentile(errors, 90.0),
        "mean_estimated_error_cm": safe_mean(intervals),
        "median_estimated_error_cm": safe_percentile(intervals, 50.0),
    }


def interpretation(overall_rows: list[dict[str, Any]]) -> str:
    by_tier = {row["confidence_tier"]: row for row in overall_rows}
    low = by_tier.get("low_confidence", {})
    high = by_tier.get("high_confidence", {})
    if low and high and float(low["mean_estimated_error_cm"]) > float(high["mean_estimated_error_cm"]):
        return "Uncertainty intervals are wider for low-confidence predictions, and test coverage can now be reported explicitly."
    return "Uncertainty intervals were produced, but interval widths do not yet clearly track confidence tiers."


def safe_mean(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(values.mean())


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    return 0.0 if values.size == 0 else float(np.percentile(values, percentile))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coverage_fieldnames() -> list[str]:
    return [
        "target",
        "confidence_tier",
        "prediction_count",
        "coverage",
        "mae_cm",
        "p50_abs_error_cm",
        "p80_abs_error_cm",
        "p90_abs_error_cm",
        "mean_estimated_error_cm",
        "median_estimated_error_cm",
    ]


def format_coverage_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 4F Measurement Uncertainty Calibration",
        "",
        f"Run: `{summary['run_name']}`",
        f"Calibration splits: `{', '.join(summary['calibration_splits'])}`",
        f"Evaluation split: `{summary['evaluation_split']}`",
        "",
        "## Test Coverage",
        "",
        "| Target | Confidence | Count | Coverage | MAE | P90 Error | Mean Estimated Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["overall"]:
        lines.append(
            f"| {row['target']} | {row['confidence_tier']} | {row['prediction_count']} | {float(row['coverage']):.3f} | "
            f"{float(row['mae_cm']):.4f} | {float(row['p90_abs_error_cm']):.4f} | {float(row['mean_estimated_error_cm']):.4f} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], "", "This is synthetic-calibrated interval validation, not real-world production readiness."])
    return "\n".join(lines) + "\n"


def format_product_action_policy(policy: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 4F Product Action Policy",
            "",
            "Prediction intervals are calibrated from train/val absolute errors and evaluated on test.",
            "",
            "- high confidence with interval <= 3 cm: accept as AI estimate.",
            "- medium confidence or interval > 3 cm: require manual confirmation.",
            "- low confidence or interval > 5 cm: request retake or tape measurement.",
            "",
            "Fallback order for interval width:",
            "",
            *[f"- `{item}`" for item in policy["fallback_order"]],
            "",
            "These actions remain conservative because the calibration is synthetic-only.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate prediction intervals for geometry residual measurements.")
    parser.add_argument("--predictions", default=DEFAULT_PHASE4D_PREDICTIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    result = calibrate_measurement_uncertainty(args.predictions, args.output, run_name=args.run_name)
    print(f"Summary: {result['coverage_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
