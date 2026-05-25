from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
PER_TARGET_TOP_FEATURES_FILENAME = "per_target_top_features.csv"
FEATURE_GROUP_SUMMARY_FILENAME = "feature_group_summary.csv"
LOW_CORRELATION_THRESHOLD = 0.20
NEAR_CONSTANT_STD_THRESHOLD = 1e-9
DOMINANT_FEATURE_SHARE_THRESHOLD = 0.35
TOP_FEATURE_LIMIT = 10


def analyze_feature_importance(
    experiment_dir: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir)
    dataset_path = Path(dataset_root)
    model_path = experiment_path / "model.json"
    feature_names_path = experiment_path / "feature_names.json"
    metrics_path = experiment_path / "metrics.json"
    predictions_test_path = experiment_path / "predictions_test.csv"

    for required_path, label in (
        (model_path, "model.json"),
        (feature_names_path, "feature_names.json"),
        (metrics_path, "metrics.json"),
        (predictions_test_path, "predictions_test.csv"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    model = _read_json(model_path)
    feature_names = list(_read_json(feature_names_path))
    metrics = _read_json(metrics_path)
    target_columns = list(metrics.get("target_columns", model.get("target_columns", [])))
    if not target_columns:
        raise ValueError(f"No target columns found in experiment metrics: {metrics_path}")

    coefficients = coefficient_matrix(model, feature_names, target_columns)
    dataset = SyntheticBodyDataset(dataset_path, split="all")
    samples = list(dataset)
    feature_matrix = extract_sample_feature_matrix(samples, feature_names)
    target_matrix = _target_matrix(samples, target_columns)
    correlations = feature_target_correlations(feature_matrix, target_matrix)
    feature_stds = feature_matrix.std(axis=0)

    summary = build_feature_importance_summary(
        experiment_path,
        dataset_path,
        feature_names,
        target_columns,
        coefficients,
        correlations,
        feature_stds,
        sample_count=len(samples),
    )
    top_feature_rows = build_top_feature_rows(summary, feature_names, target_columns, coefficients, correlations)
    group_rows = build_feature_group_rows(summary, feature_names, target_columns, coefficients, correlations)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    top_features_path = output_path / PER_TARGET_TOP_FEATURES_FILENAME
    group_summary_path = output_path / FEATURE_GROUP_SUMMARY_FILENAME
    _write_json(summary_path, summary)
    report_path.write_text(format_feature_importance_report(summary), encoding="utf-8")
    write_csv(top_features_path, top_feature_rows)
    write_csv(group_summary_path, group_rows)

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "per_target_top_features_path": str(top_features_path),
        "feature_group_summary_path": str(group_summary_path),
        "summary": summary,
    }


def coefficient_matrix(model: dict[str, Any], feature_names: list[str], target_columns: list[str]) -> np.ndarray:
    if "coefficients" not in model:
        raise ValueError("Model artifact has no coefficients. Re-run a ridge image-feature experiment.")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    expected_shape = (len(feature_names), len(target_columns))
    if coefficients.shape != expected_shape:
        raise ValueError(f"Model coefficients shape {coefficients.shape} does not match expected {expected_shape}.")
    return coefficients


def build_feature_importance_summary(
    experiment_path: Path,
    dataset_path: Path,
    feature_names: list[str],
    target_columns: list[str],
    coefficients: np.ndarray,
    correlations: np.ndarray,
    feature_stds: np.ndarray,
    sample_count: int,
) -> dict[str, Any]:
    per_target = {}
    warnings = []
    hard_targets = {"weight_kg", "waist_cm", "neck_cm", "shoulder_cm", "calf_cm"}

    for target_index, target in enumerate(target_columns):
        target_coefficients = coefficients[:, target_index]
        target_correlations = correlations[:, target_index]
        abs_coefficients = np.abs(target_coefficients)
        abs_correlations = np.abs(target_correlations)
        top_abs_indices = rank_absolute_values(target_coefficients, TOP_FEATURE_LIMIT)
        top_positive_indices = rank_positive_values(target_coefficients, TOP_FEATURE_LIMIT)
        top_negative_indices = rank_negative_values(target_coefficients, TOP_FEATURE_LIMIT)
        max_abs_correlation = float(abs_correlations.max()) if len(abs_correlations) else 0.0
        dominant_warning = dominant_feature_warning(target, feature_names, abs_coefficients)
        if dominant_warning:
            warnings.append(dominant_warning)
        if target in hard_targets and max_abs_correlation < LOW_CORRELATION_THRESHOLD:
            warnings.append(
                f"Low correlation signal: hard target {target} has max absolute feature correlation {max_abs_correlation:.3f}."
            )

        per_target[target] = {
            "top_positive_coefficients": feature_rank_records(top_positive_indices, feature_names, target_coefficients, target_correlations),
            "top_negative_coefficients": feature_rank_records(top_negative_indices, feature_names, target_coefficients, target_correlations),
            "top_absolute_coefficients": feature_rank_records(top_abs_indices, feature_names, target_coefficients, target_correlations),
            "max_abs_correlation": max_abs_correlation,
            "top_correlated_features": feature_rank_records(rank_absolute_values(target_correlations, TOP_FEATURE_LIMIT), feature_names, target_coefficients, target_correlations),
            "top_feature_groups": target_group_summary(feature_names, target_coefficients, target_correlations)[:5],
        }

    near_constant_features = near_constant_feature_names(feature_names, feature_stds)
    if near_constant_features:
        warnings.append(f"Near-constant features detected: {', '.join(near_constant_features[:10])}.")
    duplicate_groups = repeated_feature_groups(feature_names, coefficients)
    if duplicate_groups:
        warnings.append(f"Repeated coefficient patterns detected for {len(duplicate_groups)} feature groups.")

    return {
        "experiment": str(experiment_path),
        "dataset": str(dataset_path),
        "sample_count": sample_count,
        "feature_count": len(feature_names),
        "target_columns": target_columns,
        "per_target": per_target,
        "feature_groups": global_feature_group_summary(feature_names, coefficients, correlations),
        "near_constant_features": near_constant_features,
        "repeated_coefficient_feature_groups": duplicate_groups,
        "warnings": warnings,
    }


def rank_positive_values(values: np.ndarray, limit: int) -> list[int]:
    return [int(index) for index in np.argsort(values)[::-1][:limit]]


def rank_negative_values(values: np.ndarray, limit: int) -> list[int]:
    return [int(index) for index in np.argsort(values)[:limit]]


def rank_absolute_values(values: np.ndarray, limit: int) -> list[int]:
    return [int(index) for index in np.argsort(np.abs(values))[::-1][:limit]]


def feature_rank_records(
    indices: list[int],
    feature_names: list[str],
    coefficients: np.ndarray,
    correlations: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "feature_name": feature_names[index],
            "feature_group": feature_group(feature_names[index]),
            "coefficient": float(coefficients[index]),
            "abs_coefficient": abs(float(coefficients[index])),
            "feature_target_correlation": float(correlations[index]),
            "abs_feature_target_correlation": abs(float(correlations[index])),
        }
        for index in indices
    ]


def feature_target_correlations(feature_matrix: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    correlations = np.zeros((feature_matrix.shape[1], target_matrix.shape[1]), dtype=np.float64)
    for feature_index in range(feature_matrix.shape[1]):
        feature_values = feature_matrix[:, feature_index]
        for target_index in range(target_matrix.shape[1]):
            correlations[feature_index, target_index] = pearson_correlation(feature_values, target_matrix[:, target_index])
    return correlations


def pearson_correlation(first_values: np.ndarray, second_values: np.ndarray) -> float:
    if len(first_values) != len(second_values) or len(first_values) < 2:
        return 0.0
    first_std = float(np.std(first_values))
    second_std = float(np.std(second_values))
    if first_std <= NEAR_CONSTANT_STD_THRESHOLD or second_std <= NEAR_CONSTANT_STD_THRESHOLD:
        return 0.0
    return float(np.corrcoef(first_values, second_values)[0, 1])


def feature_group(feature_name: str) -> str:
    if "volume_proxy" in feature_name or "width_depth_proxy" in feature_name:
        return "volume_proxy"
    if "integrated_width" in feature_name:
        return "integrated_profile"
    if "area_to_height" in feature_name or "area_to_bbox" in feature_name:
        return "area_scale_ratio"
    if "min_torso" in feature_name or "waist_min" in feature_name:
        return "torso_waist_geometry"
    if "shoulder_peak" in feature_name or "shoulder_slope" in feature_name:
        return "shoulder_geometry"
    if "neck_" in feature_name or "neck_to" in feature_name:
        return "neck_geometry"
    if "calf_" in feature_name or "lower_leg" in feature_name:
        return "lower_leg_geometry"
    if feature_name.startswith("front_to_side"):
        return "cross_view_ratio"
    if "arm_span" in feature_name or "extension" in feature_name:
        return "arm_span_extension"
    if "bbox" in feature_name or "body_top" in feature_name or "body_bottom" in feature_name or "body_left" in feature_name or "body_right" in feature_name:
        return "bbox_scale_position"
    if "area" in feature_name:
        return "area"
    if "to_height" in feature_name:
        return "height_normalized_ratio"
    if "to_waist" in feature_name:
        return "body_band_ratio"
    if any(token in feature_name for token in ("width_ratio", "depth_ratio")):
        return "band_width_profile"
    if any(token in feature_name for token in ("extent_ratio", "center_x_ratio", "asymmetry_ratio")):
        return "band_position_asymmetry"
    if feature_name.startswith("front_"):
        return "front_other"
    if feature_name.startswith("side_"):
        return "side_other"
    return "other"


def target_group_summary(
    feature_names: list[str],
    coefficients: np.ndarray,
    correlations: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[float]]] = {}
    for index, name in enumerate(feature_names):
        group = feature_group(name)
        groups.setdefault(group, {"abs_coefficients": [], "abs_correlations": []})
        groups[group]["abs_coefficients"].append(abs(float(coefficients[index])))
        groups[group]["abs_correlations"].append(abs(float(correlations[index])))
    rows = [
        {
            "feature_group": group,
            "feature_count": len(values["abs_coefficients"]),
            "mean_abs_coefficient": _mean(values["abs_coefficients"]),
            "max_abs_coefficient": max(values["abs_coefficients"]),
            "mean_abs_correlation": _mean(values["abs_correlations"]),
            "max_abs_correlation": max(values["abs_correlations"]),
        }
        for group, values in groups.items()
    ]
    return sorted(rows, key=lambda row: row["mean_abs_coefficient"], reverse=True)


def global_feature_group_summary(feature_names: list[str], coefficients: np.ndarray, correlations: np.ndarray) -> list[dict[str, Any]]:
    target_count = coefficients.shape[1]
    rows = []
    for group in sorted({feature_group(name) for name in feature_names}):
        indices = [index for index, name in enumerate(feature_names) if feature_group(name) == group]
        abs_coefficients = np.abs(coefficients[indices, :]).reshape(-1)
        abs_correlations = np.abs(correlations[indices, :]).reshape(-1)
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(indices),
                "target_count": target_count,
                "mean_abs_coefficient": float(abs_coefficients.mean()) if len(abs_coefficients) else 0.0,
                "max_abs_coefficient": float(abs_coefficients.max()) if len(abs_coefficients) else 0.0,
                "mean_abs_correlation": float(abs_correlations.mean()) if len(abs_correlations) else 0.0,
                "max_abs_correlation": float(abs_correlations.max()) if len(abs_correlations) else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["mean_abs_coefficient"], reverse=True)


def near_constant_feature_names(feature_names: list[str], feature_stds: np.ndarray) -> list[str]:
    return [feature_names[index] for index, std in enumerate(feature_stds) if float(std) <= NEAR_CONSTANT_STD_THRESHOLD]


def repeated_feature_groups(feature_names: list[str], coefficients: np.ndarray) -> list[dict[str, Any]]:
    groups: dict[tuple[float, ...], list[str]] = {}
    for index, name in enumerate(feature_names):
        key = tuple(round(float(value), 10) for value in coefficients[index, :])
        groups.setdefault(key, []).append(name)
    return [
        {"features": names, "count": len(names)}
        for names in groups.values()
        if len(names) > 1
    ]


def dominant_feature_warning(target: str, feature_names: list[str], abs_coefficients: np.ndarray) -> str | None:
    total = float(abs_coefficients.sum())
    if total <= 0:
        return None
    top_index = int(np.argmax(abs_coefficients))
    share = float(abs_coefficients[top_index] / total)
    if share >= DOMINANT_FEATURE_SHARE_THRESHOLD:
        return f"Dominant feature: {feature_names[top_index]} contributes {share:.3f} of absolute coefficient mass for {target}."
    return None


def build_top_feature_rows(
    summary: dict[str, Any],
    feature_names: list[str],
    target_columns: list[str],
    coefficients: np.ndarray,
    correlations: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(target_columns):
        for rank_type, indices in (
            ("positive_coefficient", rank_positive_values(coefficients[:, target_index], TOP_FEATURE_LIMIT)),
            ("negative_coefficient", rank_negative_values(coefficients[:, target_index], TOP_FEATURE_LIMIT)),
            ("absolute_coefficient", rank_absolute_values(coefficients[:, target_index], TOP_FEATURE_LIMIT)),
            ("absolute_correlation", rank_absolute_values(correlations[:, target_index], TOP_FEATURE_LIMIT)),
        ):
            for rank, feature_index in enumerate(indices, start=1):
                rows.append(
                    {
                        "target": target,
                        "rank_type": rank_type,
                        "rank": rank,
                        "feature_name": feature_names[feature_index],
                        "feature_group": feature_group(feature_names[feature_index]),
                        "coefficient": float(coefficients[feature_index, target_index]),
                        "abs_coefficient": abs(float(coefficients[feature_index, target_index])),
                        "feature_target_correlation": float(correlations[feature_index, target_index]),
                        "abs_feature_target_correlation": abs(float(correlations[feature_index, target_index])),
                    }
                )
    return rows


def build_feature_group_rows(
    summary: dict[str, Any],
    feature_names: list[str],
    target_columns: list[str],
    coefficients: np.ndarray,
    correlations: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(target_columns):
        for group_row in target_group_summary(feature_names, coefficients[:, target_index], correlations[:, target_index]):
            rows.append({"target": target, **group_row})
    return rows


def format_feature_importance_report(summary: dict[str, Any]) -> str:
    hard_targets = ["weight_kg", "waist_cm", "neck_cm", "shoulder_cm", "calf_cm"]
    lines = [
        "# Feature Importance Diagnostics",
        "",
        f"Experiment: `{summary['experiment']}`",
        f"Dataset: `{summary['dataset']}`",
        f"Samples: {summary['sample_count']}",
        f"Feature count: {summary['feature_count']}",
        "",
        "## Hard Target Feature Groups",
        "",
        _markdown_table(
            ["Target", "Top Feature Groups"],
            [
                [
                    target,
                    ", ".join(group["feature_group"] for group in summary["per_target"][target]["top_feature_groups"][:3]),
                ]
                for target in hard_targets
                if target in summary["per_target"]
            ],
        ),
        "",
        "## Hard Target Top Absolute Coefficients",
        "",
        _markdown_table(
            ["Target", "Top Features"],
            [
                [
                    target,
                    ", ".join(record["feature_name"] for record in summary["per_target"][target]["top_absolute_coefficients"][:5]),
                ]
                for target in hard_targets
                if target in summary["per_target"]
            ],
        ),
        "",
        "## Global Feature Groups",
        "",
        _markdown_table(
            ["Feature Group", "Features", "Mean Abs Coef", "Max Abs Corr"],
            [
                [
                    row["feature_group"],
                    str(row["feature_count"]),
                    _format_float(row["mean_abs_coefficient"]),
                    _format_float(row["max_abs_correlation"]),
                ]
                for row in summary["feature_groups"]
            ],
        ),
        "",
        "## Warnings",
        "",
    ]
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("No feature importance warnings were emitted.")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _header in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze ridge feature importance for image-feature experiments.")
    parser.add_argument("--experiment", required=True, help="Experiment directory containing model.json and feature_names.json.")
    parser.add_argument("--dataset", required=True, help="Dataset root used for feature-target correlations.")
    parser.add_argument("--output", required=True, help="Output directory for feature diagnostics.")
    args = parser.parse_args(argv)

    result = analyze_feature_importance(args.experiment, args.dataset, args.output)
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Per-target top features: {result['per_target_top_features_path']}")
    print(f"Feature group summary: {result['feature_group_summary_path']}")
    if result["summary"]["warnings"]:
        print("Warnings:")
        for warning in result["summary"]["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
