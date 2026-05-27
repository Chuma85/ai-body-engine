from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

CONFIDENCE_POLICY_JSON = "confidence_policy.json"
CONFIDENCE_EVAL_JSON = "confidence_eval_results.json"
CONFIDENCE_EVAL_CSV = "confidence_eval_results.csv"
PER_TARGET_CONFIDENCE_CSV = "per_target_confidence_summary.csv"
SUMMARY_MD = "confidence_gate_summary.md"

DEFAULT_PHASE4D_PREDICTIONS = "artifacts/phase_4d_residual_correction/residual_training_summary.csv"
DEFAULT_RUN_NAME = "geometry_plus_residual__gradient_boosting"

HIGH_RESIDUAL_ABS_MAX_CM = 4.0
HIGH_RESIDUAL_REL_MAX = 0.045
MEDIUM_RESIDUAL_ABS_MAX_CM = 8.0
MEDIUM_RESIDUAL_REL_MAX = 0.09


def evaluate_measurement_confidence(
    predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
    output_dir: str | Path = "artifacts/phase_4e_measurement_confidence_gating",
    run_name: str = DEFAULT_RUN_NAME,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_prediction_rows(predictions_csv) if row["run_name"] == run_name]
    if not rows:
        raise ValueError(f"No prediction rows found for run '{run_name}' in {predictions_csv}")

    evaluated_rows = [apply_confidence_policy(row) for row in rows]
    overall_rows = build_overall_confidence_rows(evaluated_rows)
    per_target_rows = build_per_target_confidence_rows(evaluated_rows)
    policy = confidence_policy_payload()
    summary = {
        "predictions_csv": str(predictions_csv),
        "run_name": run_name,
        "policy": policy,
        "overall": overall_rows,
        "per_target": per_target_rows,
        "interpretation": interpretation(overall_rows),
    }

    paths = {
        "confidence_policy_json": output_path / CONFIDENCE_POLICY_JSON,
        "confidence_eval_json": output_path / CONFIDENCE_EVAL_JSON,
        "confidence_eval_csv": output_path / CONFIDENCE_EVAL_CSV,
        "per_target_confidence_csv": output_path / PER_TARGET_CONFIDENCE_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_json(paths["confidence_policy_json"], policy)
    write_json(paths["confidence_eval_json"], summary)
    write_csv(paths["confidence_eval_csv"], overall_rows, confidence_eval_fieldnames())
    write_csv(paths["per_target_confidence_csv"], per_target_rows, per_target_fieldnames())
    paths["summary_md"].write_text(format_summary(summary), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Residual prediction CSV does not exist: {csv_path}")
    rows: list[dict[str, Any]] = []
    numeric_fields = {
        "geometry_estimate_cm",
        "calibrated_label_cm",
        "residual_cm",
        "predicted_residual_cm",
        "final_estimate_cm",
        "abs_error_cm",
    }
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            converted: dict[str, Any] = dict(row)
            for field in numeric_fields:
                converted[field] = float(row[field])
            rows.append(converted)
    return rows


def apply_confidence_policy(row: dict[str, Any]) -> dict[str, Any]:
    residual_abs = abs(float(row["predicted_residual_cm"]))
    final_estimate = max(abs(float(row["final_estimate_cm"])), 1e-9)
    residual_relative = residual_abs / final_estimate
    geometry_flags = str(row.get("geometry_quality_flags", ""))
    existing_flags = str(row.get("confidence_flags", ""))
    reasons = confidence_reasons(residual_abs, residual_relative, geometry_flags, existing_flags)
    tier = confidence_tier(reasons, residual_abs, residual_relative)
    return {
        **row,
        "absolute_residual_correction_cm": residual_abs,
        "relative_residual_correction": residual_relative,
        "confidence_tier": tier,
        "product_action": product_action_for_tier(tier),
        "confidence_reasons": ";".join(reasons) if reasons else "small_residual_and_clean_geometry",
    }


def confidence_reasons(
    residual_abs: float,
    residual_relative: float,
    geometry_flags: str,
    existing_flags: str,
) -> list[str]:
    reasons = []
    if geometry_flags and geometry_flags != "ok":
        reasons.append("geometry_quality_flag")
    if existing_flags and existing_flags != "ok":
        reasons.extend(flag for flag in existing_flags.split(";") if flag and flag != "ok")
    if residual_abs > MEDIUM_RESIDUAL_ABS_MAX_CM:
        reasons.append("large_absolute_residual")
    elif residual_abs > HIGH_RESIDUAL_ABS_MAX_CM:
        reasons.append("moderate_absolute_residual")
    if residual_relative > MEDIUM_RESIDUAL_REL_MAX:
        reasons.append("large_relative_residual")
    elif residual_relative > HIGH_RESIDUAL_REL_MAX:
        reasons.append("moderate_relative_residual")
    return sorted(set(reasons))


def confidence_tier(reasons: list[str], residual_abs: float, residual_relative: float) -> str:
    severe_tokens = {"geometry_quality_flag", "large_residual_correction", "final_estimate_out_of_range", "large_absolute_residual", "large_relative_residual"}
    if any(reason in severe_tokens or reason.startswith("geometry_") for reason in reasons):
        return "low_confidence"
    if residual_abs <= HIGH_RESIDUAL_ABS_MAX_CM and residual_relative <= HIGH_RESIDUAL_REL_MAX and not reasons:
        return "high_confidence"
    return "medium_confidence"


def product_action_for_tier(tier: str) -> str:
    if tier == "high_confidence":
        return "accept_as_ai_estimate"
    if tier == "medium_confidence":
        return "require_manual_confirmation"
    if tier == "low_confidence":
        return "request_retake_or_tape_measurement"
    raise ValueError(f"Unknown confidence tier: {tier}")


def build_overall_confidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split_name, split_rows in split_groups(rows).items():
        for tier in ("high_confidence", "medium_confidence", "low_confidence"):
            tier_rows = [row for row in split_rows if row["confidence_tier"] == tier]
            output.append(confidence_summary_row(tier, "all_targets", tier_rows, split_rows, split_name))
    return output


def build_per_target_confidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split_name, split_rows in split_groups(rows, include_train_val=False).items():
        for target in sorted({row["target"] for row in split_rows}):
            target_rows = [row for row in split_rows if row["target"] == target]
            for tier in ("high_confidence", "medium_confidence", "low_confidence"):
                tier_rows = [row for row in target_rows if row["confidence_tier"] == tier]
                output.append(confidence_summary_row(tier, target, tier_rows, target_rows, split_name))
    return output


def split_groups(rows: list[dict[str, Any]], include_train_val: bool = True) -> dict[str, list[dict[str, Any]]]:
    groups = {"all": rows}
    split_names = ("train", "val", "test") if include_train_val else ("test",)
    for split_name in split_names:
        groups[split_name] = [row for row in rows if row.get("dataset_split") == split_name]
    return groups


def confidence_summary_row(
    tier: str,
    target: str,
    rows: list[dict[str, Any]],
    denominator_rows: list[dict[str, Any]],
    dataset_split: str,
) -> dict[str, Any]:
    errors = np.asarray([row["abs_error_cm"] for row in rows], dtype=np.float64)
    residuals = np.asarray([row["absolute_residual_correction_cm"] for row in rows], dtype=np.float64)
    count = len(rows)
    return {
        "dataset_split": dataset_split,
        "target": target,
        "confidence_tier": tier,
        "product_action": product_action_for_tier(tier),
        "prediction_count": count,
        "prediction_percent": 0.0 if not denominator_rows else count / len(denominator_rows) * 100.0,
        "mae_cm": safe_mean(errors),
        "median_abs_error_cm": safe_median(errors),
        "p90_abs_error_cm": safe_percentile(errors, 90.0),
        "mean_abs_residual_correction_cm": safe_mean(residuals),
    }


def confidence_policy_payload() -> dict[str, Any]:
    return {
        "tiers": ["high_confidence", "medium_confidence", "low_confidence"],
        "actions": {
            "high_confidence": "accept_as_ai_estimate",
            "medium_confidence": "require_manual_confirmation",
            "low_confidence": "request_retake_or_tape_measurement",
        },
        "thresholds": {
            "high_residual_abs_max_cm": HIGH_RESIDUAL_ABS_MAX_CM,
            "high_residual_relative_max": HIGH_RESIDUAL_REL_MAX,
            "medium_residual_abs_max_cm": MEDIUM_RESIDUAL_ABS_MAX_CM,
            "medium_residual_relative_max": MEDIUM_RESIDUAL_REL_MAX,
        },
        "low_confidence_reasons": [
            "large residual correction",
            "missing/invalid geometry",
            "unstable mask or scale",
            "out-of-range final estimate",
        ],
    }


def interpretation(overall_rows: list[dict[str, Any]]) -> str:
    test_rows = [row for row in overall_rows if row["dataset_split"] == "test"]
    by_tier = {row["confidence_tier"]: row for row in test_rows}
    high = by_tier.get("high_confidence", {})
    low = by_tier.get("low_confidence", {})
    if high and low and float(low["mae_cm"]) > float(high["mae_cm"]) + 0.25:
        return "On the test split, low-confidence predictions have meaningfully higher MAE than high-confidence predictions, so the policy is directionally useful."
    return "Confidence tiers mostly capture intervention size and product risk; error separation is weak and thresholds need real-world calibration."


def safe_mean(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(values.mean())


def safe_median(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.median(values))


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


def confidence_eval_fieldnames() -> list[str]:
    return ["dataset_split", "target", "confidence_tier", "product_action", "prediction_count", "prediction_percent", "mae_cm", "median_abs_error_cm", "p90_abs_error_cm", "mean_abs_residual_correction_cm"]


def per_target_fieldnames() -> list[str]:
    return confidence_eval_fieldnames()


def format_summary(summary: dict[str, Any]) -> str:
    all_rows = [row for row in summary["overall"] if row["dataset_split"] == "all"]
    test_rows = [row for row in summary["overall"] if row["dataset_split"] == "test"]
    lines = [
        "# Phase 4E Measurement Confidence Gating",
        "",
        f"Run: `{summary['run_name']}`",
        "",
        "## Overall Confidence Results (All Splits)",
        "",
        "| Tier | Action | Count | Percent | MAE | P90 Error | Mean Residual Correction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['confidence_tier']} | {row['product_action']} | {row['prediction_count']} | "
            f"{float(row['prediction_percent']):.1f} | {float(row['mae_cm']):.4f} | "
            f"{float(row['p90_abs_error_cm']):.4f} | {float(row['mean_abs_residual_correction_cm']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Test Split Confidence Results",
            "",
            "| Tier | Action | Count | Percent | MAE | P90 Error | Mean Residual Correction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| {row['confidence_tier']} | {row['product_action']} | {row['prediction_count']} | "
            f"{float(row['prediction_percent']):.1f} | {float(row['mae_cm']):.4f} | "
            f"{float(row['p90_abs_error_cm']):.4f} | {float(row['mean_abs_residual_correction_cm']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Product Actions",
            "",
            "- high confidence: accept as AI estimate",
            "- medium confidence: require manual confirmation",
            "- low confidence: request retake or tape measurement",
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "This remains synthetic-calibrated validation, not real-world production readiness.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate confidence gates for geometry residual measurements.")
    parser.add_argument("--predictions", default=DEFAULT_PHASE4D_PREDICTIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    result = evaluate_measurement_confidence(args.predictions, args.output, run_name=args.run_name)
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
