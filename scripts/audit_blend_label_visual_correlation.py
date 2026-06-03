from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_blend_dataset_baseline import (
    DEFAULT_DATASET,
    DEFAULT_TARGET_COLUMNS,
    VIEW_COLUMNS,
    extract_blend_image_features,
    validate_blend_dataset,
)
from training.features.image_silhouette_features import (
    create_foreground_mask,
    foreground_bounding_box,
    load_rgb_image,
)

DEFAULT_OUT = "artifacts/phase_3h_f_label_visual_correlation"
DEFAULT_MIN_ABS_CORRELATION = 0.25
TOP_FEATURE_COUNT = 10
SAFE_MEASUREMENT_RANGES = {
    "height_cm": (140.0, 220.0),
    "chest_cm": (60.0, 160.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (60.0, 170.0),
    "shoulder_cm": (25.0, 80.0),
    "inseam_cm": (50.0, 110.0),
}
REQUIRED_OUTPUTS = [
    "correlation_report.json",
    "correlation_summary.md",
    "feature_label_correlation.csv",
    "target_correlation_matrix.csv",
    "visual_feature_summary.csv",
    "label_summary.csv",
    "flagged_targets.csv",
    "top_features_by_target.csv",
]
KEY_VISUAL_FEATURES = [
    "front_raw_bbox_width_ratio",
    "front_raw_bbox_height_ratio",
    "front_raw_mask_area_ratio",
    "front_width_height_ratio",
    "front_centroid_x_ratio",
    "front_centroid_y_ratio",
    "front_projection_row_width_mean",
    "front_projection_column_height_mean",
    "side_raw_bbox_width_ratio",
    "side_raw_bbox_height_ratio",
    "side_raw_mask_area_ratio",
    "side_width_height_ratio",
    "side_centroid_x_ratio",
    "side_centroid_y_ratio",
    "side_projection_row_width_mean",
    "side_projection_column_height_mean",
    "back_raw_bbox_width_ratio",
    "back_raw_bbox_height_ratio",
    "back_raw_mask_area_ratio",
    "back_width_height_ratio",
    "back_centroid_x_ratio",
    "back_centroid_y_ratio",
    "back_projection_row_width_mean",
    "back_projection_column_height_mean",
    "side_front_width_ratio",
    "back_front_width_ratio",
    "front_side_back_area_proxy",
]
EXPECTED_COVARY_TARGET_PAIRS = [
    ("chest_cm", "waist_cm"),
    ("chest_cm", "hip_cm"),
    ("waist_cm", "hip_cm"),
    ("chest_cm", "shoulder_cm"),
    ("height_cm", "inseam_cm"),
]


def audit_label_visual_correlation(
    dataset: str | Path = DEFAULT_DATASET,
    out: str | Path = DEFAULT_OUT,
    target_columns: list[str] | None = None,
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
) -> dict[str, Any]:
    targets = target_columns or [*DEFAULT_TARGET_COLUMNS]
    validation = validate_blend_dataset(dataset, targets)
    rows = validation["rows"]
    feature_names, feature_matrix = build_visual_feature_matrix(rows, dataset)
    target_matrix = build_target_matrix(rows, targets)

    label_summary = summarize_labels(target_matrix, targets)
    feature_summary = summarize_features(feature_matrix, feature_names)
    correlations = compute_feature_label_correlations(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        target_matrix=target_matrix,
        target_columns=targets,
    )
    top_features = top_features_by_target(correlations, top_n=TOP_FEATURE_COUNT)
    target_matrix_rows = compute_target_correlation_matrix(target_matrix, targets)
    flagged_targets = flag_targets(
        label_summary=label_summary,
        feature_summary=feature_summary,
        top_features=top_features,
        target_correlation_rows=target_matrix_rows,
        min_abs_correlation=min_abs_correlation,
    )
    view_flags = flag_view_feature_behavior(feature_summary)
    flagged_targets.extend(view_flags)
    weak_targets = [
        row["target"]
        for row in top_features
        if row["rank"] == 1 and abs(float(row["abs_max_correlation"])) < min_abs_correlation
    ]
    suspicious_label_behavior = [
        row for row in flagged_targets if row["category"] in {"low_label_variation", "safe_range_violation", "target_correlation"}
    ]
    suspicious_visual_behavior = [
        row for row in flagged_targets if row["category"] in {"low_visual_variation", "view_similarity"}
    ]
    strongest_by_target = {
        row["target"]: row
        for row in top_features
        if row["rank"] == 1
    }
    likely_reason = infer_likely_reason(weak_targets, suspicious_label_behavior, suspicious_visual_behavior)
    recommended_next_action = recommend_next_action(weak_targets, suspicious_label_behavior, suspicious_visual_behavior)

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "3H-F",
        "dataset": str(Path(dataset)),
        "sample_count": validation["sample_count"],
        "image_count": validation["image_count"],
        "target_columns": targets,
        "min_abs_correlation": min_abs_correlation,
        "feature_count": len(feature_names),
        "strongest_visual_correlation_by_target": strongest_by_target,
        "weakly_learnable_targets": weak_targets,
        "suspicious_label_behavior": suspicious_label_behavior,
        "suspicious_visual_feature_behavior": suspicious_visual_behavior,
        "likely_reason_phase_3h_e_mae_is_high": likely_reason,
        "recommended_next_action": recommended_next_action,
        "label_summary": label_summary,
        "visual_feature_summary": feature_summary,
        "target_correlation_matrix": target_matrix_rows,
        "flagged_targets": flagged_targets,
    }
    write_json(output_dir / "correlation_report.json", report)
    write_csv(output_dir / "feature_label_correlation.csv", correlations)
    write_csv(output_dir / "target_correlation_matrix.csv", target_matrix_rows)
    write_csv(output_dir / "visual_feature_summary.csv", feature_summary)
    write_csv(output_dir / "label_summary.csv", label_summary)
    write_csv(output_dir / "flagged_targets.csv", flagged_targets)
    write_csv(output_dir / "top_features_by_target.csv", top_features)
    write_summary(output_dir / "correlation_summary.md", report)
    return report


def build_visual_feature_matrix(rows: list[dict[str, str]], dataset: str | Path) -> tuple[list[str], np.ndarray]:
    feature_rows = [extract_visual_features(row, dataset) for row in rows]
    feature_names = sorted(feature_rows[0])
    matrix = np.asarray([[float(feature_row[name]) for name in feature_names] for feature_row in feature_rows], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Visual feature matrix contains non-finite values.")
    return feature_names, matrix


def extract_visual_features(row: dict[str, str], dataset: str | Path) -> dict[str, float]:
    features = extract_blend_image_features(row, dataset)
    dataset_path = Path(dataset)
    for view, column in VIEW_COLUMNS.items():
        image_path = dataset_path / row[column]
        image = load_rgb_image(image_path)
        mask = create_foreground_mask(image)
        x_min, y_min, x_max, y_max = foreground_bounding_box(mask)
        ys, xs = np.where(mask)
        image_height, image_width = mask.shape
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        features[f"{view}_width_height_ratio"] = _safe_ratio(float(bbox_width), float(bbox_height))
        features[f"{view}_centroid_x_ratio"] = float(xs.mean()) / float(image_width)
        features[f"{view}_centroid_y_ratio"] = float(ys.mean()) / float(image_height)
    features["side_front_width_ratio"] = _safe_ratio(
        features["side_raw_bbox_width_ratio"],
        features["front_raw_bbox_width_ratio"],
    )
    features["back_front_width_ratio"] = _safe_ratio(
        features["back_raw_bbox_width_ratio"],
        features["front_raw_bbox_width_ratio"],
    )
    return features


def build_target_matrix(rows: list[dict[str, str]], target_columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[target]) for target in target_columns] for row in rows], dtype=np.float64)


def summarize_labels(target_matrix: np.ndarray, target_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(target_columns):
        values = target_matrix[:, index]
        mean = float(values.mean())
        std = float(values.std())
        cv = _safe_ratio(std, abs(mean))
        lower, upper = SAFE_MEASUREMENT_RANGES.get(target, (-math.inf, math.inf))
        safe_range_violations = int(((values < lower) | (values > upper)).sum())
        unique_count = int(len({round(float(value), 6) for value in values}))
        rows.append(
            {
                "target": target,
                "count": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": mean,
                "std": std,
                "coefficient_of_variation": cv,
                "unique_count": unique_count,
                "safe_min": lower if math.isfinite(lower) else "",
                "safe_max": upper if math.isfinite(upper) else "",
                "safe_range_violation_count": safe_range_violations,
                "low_variation": bool(std < 1.0 or cv < 0.02 or unique_count < 5),
            }
        )
    return rows


def summarize_features(feature_matrix: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key_set = set(KEY_VISUAL_FEATURES)
    for index, feature in enumerate(feature_names):
        values = feature_matrix[:, index]
        mean = float(values.mean())
        std = float(values.std())
        rows.append(
            {
                "feature": feature,
                "count": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": mean,
                "std": std,
                "coefficient_of_variation": _safe_ratio(std, abs(mean)),
                "unique_count": int(len({round(float(value), 8) for value in values})),
                "near_zero_variation": bool(std < 1e-6),
                "key_feature": feature in key_set,
            }
        )
    return rows


def compute_feature_label_correlations(
    *,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    target_matrix: np.ndarray,
    target_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(target_columns):
        target_values = target_matrix[:, target_index]
        for feature_index, feature in enumerate(feature_names):
            feature_values = feature_matrix[:, feature_index]
            pearson = pearson_correlation(feature_values, target_values)
            spearman = spearman_correlation(feature_values, target_values)
            abs_max = max(abs(pearson) if pearson is not None else 0.0, abs(spearman) if spearman is not None else 0.0)
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_max_correlation": abs_max,
                }
            )
    return rows


def pearson_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    return pearson_correlation(rank_values(x_values), rank_values(y_values))


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def top_features_by_target(correlations: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in correlations:
        by_target.setdefault(str(row["target"]), []).append(row)
    output: list[dict[str, Any]] = []
    for target, rows in sorted(by_target.items()):
        ranked = sorted(rows, key=lambda row: (-float(row["abs_max_correlation"]), str(row["feature"])))
        for rank, row in enumerate(ranked[:top_n], start=1):
            output.append(
                {
                    "target": target,
                    "rank": rank,
                    "feature": row["feature"],
                    "pearson": row["pearson"],
                    "spearman": row["spearman"],
                    "abs_max_correlation": row["abs_max_correlation"],
                }
            )
    return output


def compute_target_correlation_matrix(target_matrix: np.ndarray, target_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(target_columns):
        for right_index, right in enumerate(target_columns):
            rows.append(
                {
                    "target_a": left,
                    "target_b": right,
                    "pearson": pearson_correlation(target_matrix[:, left_index], target_matrix[:, right_index]),
                    "spearman": spearman_correlation(target_matrix[:, left_index], target_matrix[:, right_index]),
                }
            )
    return rows


def flag_targets(
    *,
    label_summary: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    top_features: list[dict[str, Any]],
    target_correlation_rows: list[dict[str, Any]],
    min_abs_correlation: float,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for row in label_summary:
        target = str(row["target"])
        if bool(row["low_variation"]):
            flags.append(flag_row(target, "low_label_variation", "warning", "label variation is low", row["std"]))
        if int(row["safe_range_violation_count"]) > 0:
            flags.append(
                flag_row(
                    target,
                    "safe_range_violation",
                    "warning",
                    f"{row['safe_range_violation_count']} labels outside safe range",
                    row["safe_range_violation_count"],
                )
            )
    strongest = {row["target"]: row for row in top_features if row["rank"] == 1}
    for target, row in strongest.items():
        if float(row["abs_max_correlation"]) < min_abs_correlation:
            flags.append(
                flag_row(
                    str(target),
                    "weak_visual_correlation",
                    "warning",
                    f"strongest visual correlation is below {min_abs_correlation}",
                    row["abs_max_correlation"],
                )
            )
    for row in feature_summary:
        if bool(row["key_feature"]) and bool(row["near_zero_variation"]):
            flags.append(flag_row(str(row["feature"]), "low_visual_variation", "warning", "key visual feature has near-zero variation", row["std"]))
    flags.extend(flag_target_correlation_behavior(target_correlation_rows))
    return flags


def flag_target_correlation_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    off_diagonal = [
        row
        for row in rows
        if row["target_a"] != row["target_b"] and row["pearson"] is not None
    ]
    high_rows = [row for row in off_diagonal if abs(float(row["pearson"])) >= 0.98]
    if len(high_rows) >= max(4, len(off_diagonal) // 2):
        flags.append(
            flag_row(
                "all_targets",
                "target_correlation",
                "warning",
                "many measurement labels move together almost perfectly",
                len(high_rows),
            )
        )
    lookup = {(row["target_a"], row["target_b"]): row for row in rows}
    for left, right in EXPECTED_COVARY_TARGET_PAIRS:
        row = lookup.get((left, right))
        if row and row["pearson"] is not None and abs(float(row["pearson"])) < 0.10:
            flags.append(
                flag_row(
                    f"{left}:{right}",
                    "target_correlation",
                    "info",
                    "expected related measurements have very low correlation",
                    row["pearson"],
                )
            )
    return flags


def flag_view_feature_behavior(feature_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_feature = {str(row["feature"]): row for row in feature_summary}
    flags: list[dict[str, Any]] = []
    for metric in ("raw_bbox_width_ratio", "raw_bbox_height_ratio", "raw_mask_area_ratio"):
        means = [
            float(summary_by_feature[f"{view}_{metric}"]["mean"])
            for view in ("front", "side", "back")
            if f"{view}_{metric}" in summary_by_feature
        ]
        if len(means) == 3 and max(means) - min(means) < 0.005:
            flags.append(
                flag_row(
                    metric,
                    "view_similarity",
                    "warning",
                    "front/side/back view feature means are very similar",
                    max(means) - min(means),
                )
            )
    low_variation_key_count = sum(
        1
        for row in feature_summary
        if bool(row["key_feature"]) and float(row["std"]) < 1e-4
    )
    if low_variation_key_count >= 5:
        flags.append(
            flag_row(
                "key_visual_features",
                "low_visual_variation",
                "warning",
                "multiple key silhouette features barely change across samples",
                low_variation_key_count,
            )
        )
    return flags


def infer_likely_reason(
    weak_targets: list[str],
    suspicious_label_behavior: list[dict[str, Any]],
    suspicious_visual_behavior: list[dict[str, Any]],
) -> str:
    if weak_targets and suspicious_visual_behavior:
        return "Several targets have weak visual correlations and key silhouette features show limited variation, so Phase 3H-E likely lacks enough rendered visual signal for the synthetic labels."
    if weak_targets:
        return "Several labels have weak correlation with deterministic visual features, so the generated measurements may not be tightly tied to the rendered body geometry."
    if suspicious_label_behavior:
        return "Label distribution issues may be limiting learnability even though some visual features correlate with targets."
    return "The audit found usable visual-label signal; Phase 3H-E MAE may be limited by baseline model capacity, feature set, or dataset size."


def recommend_next_action(
    weak_targets: list[str],
    suspicious_label_behavior: list[dict[str, Any]],
    suspicious_visual_behavior: list[dict[str, Any]],
) -> str:
    if weak_targets:
        return "Before generating 1000+ samples, tighten label-to-render geometry coupling for weak targets and rerun the 250-sample audit."
    if suspicious_visual_behavior:
        return "Improve visual variation or camera/view-specific silhouette signal before scaling the dataset."
    if suspicious_label_behavior:
        return "Fix label distribution issues, then rerun Phase 3H-F and Phase 3H-E on the same 250 samples."
    return "Proceed cautiously to a larger dataset only after repeating this audit and confirming correlations stay stable."


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3H-F Label Visual Correlation Audit",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Sample count: `{report['sample_count']}`",
        f"- Target columns: `{', '.join(report['target_columns'])}`",
        f"- Minimum absolute correlation threshold: `{report['min_abs_correlation']}`",
        "",
        "## Strongest Visual Correlation Per Target",
    ]
    for target in report["target_columns"]:
        row = report["strongest_visual_correlation_by_target"].get(target)
        if row:
            lines.append(
                f"- {target}: `{float(row['abs_max_correlation']):.4f}` via `{row['feature']}` "
                f"(pearson={_format_optional(row['pearson'])}, spearman={_format_optional(row['spearman'])})"
            )
    lines.extend(["", "## Weakly Learnable Targets"])
    if report["weakly_learnable_targets"]:
        lines.extend(f"- {target}" for target in report["weakly_learnable_targets"])
    else:
        lines.append("- None below threshold")
    lines.extend(["", "## Suspicious Label Behavior"])
    if report["suspicious_label_behavior"]:
        lines.extend(f"- {row['target_or_feature']}: {row['detail']} ({row['value']})" for row in report["suspicious_label_behavior"])
    else:
        lines.append("- None flagged")
    lines.extend(["", "## Suspicious Visual-Feature Behavior"])
    if report["suspicious_visual_feature_behavior"]:
        lines.extend(f"- {row['target_or_feature']}: {row['detail']} ({row['value']})" for row in report["suspicious_visual_feature_behavior"])
    else:
        lines.append("- None flagged")
    lines.extend(
        [
            "",
            "## Phase 3H-E MAE Interpretation",
            report["likely_reason_phase_3h_e_mae_is_high"],
            "",
            "## Recommended Next Action",
            report["recommended_next_action"],
            "",
            "This is a diagnostic correlation audit only. It is not a production accuracy test and does not validate real-world measurement accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_audit_command(
    *,
    dataset: str,
    out: str,
    target_columns: list[str],
    min_abs_correlation: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_blend_label_visual_correlation.py",
        "--dataset",
        dataset,
        "--out",
        out,
        "--target-columns",
        *target_columns,
        "--min-abs-correlation",
        str(min_abs_correlation),
    ]


def format_audit_summary(report: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-F label visual correlation audit complete.",
        f"Dataset: {report['dataset']}",
        f"Samples: {report['sample_count']}",
        "Strongest visual correlation per target:",
    ]
    for target in report["target_columns"]:
        row = report["strongest_visual_correlation_by_target"].get(target)
        if row:
            lines.append(f"  {target}: {float(row['abs_max_correlation']):.4f} via {row['feature']}")
    lines.append("Weakly learnable targets: " + (", ".join(report["weakly_learnable_targets"]) or "none"))
    lines.append("Recommended next action: " + report["recommended_next_action"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit visual-feature to measurement-label correlation for a blend dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--target-columns", nargs="+", default=[*DEFAULT_TARGET_COLUMNS])
    parser.add_argument("--min-abs-correlation", type=float, default=DEFAULT_MIN_ABS_CORRELATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_label_visual_correlation(
        dataset=args.dataset,
        out=args.out,
        target_columns=args.target_columns,
        min_abs_correlation=args.min_abs_correlation,
    )
    print(format_audit_summary(report))
    return 0


def flag_row(target_or_feature: str, category: str, severity: str, detail: str, value: Any) -> dict[str, Any]:
    return {
        "target_or_feature": target_or_feature,
        "category": category,
        "severity": severity,
        "detail": detail,
        "value": value,
    }


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
