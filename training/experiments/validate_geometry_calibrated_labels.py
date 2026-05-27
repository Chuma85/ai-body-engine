from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.audit_label_geometry_alignment import extract_geometry_proxy_matrix
from training.experiments.build_geometry_calibrated_labels import (
    TARGETS,
    evaluate_predictions,
    train_and_predict_target_specific,
)
from training.experiments.filter_label_geometry_ambiguity import target_proxy_indices
from training.experiments.optimize_silhouette_targets import promotion_gate
from training.experiments.select_regularized_hybrid_features import select_feature_names, validate_model_type
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import extract_sample_feature_matrix

VALIDATION_JSON = "calibrated_label_validation.json"
VALIDATION_CSV = "calibrated_label_validation.csv"
VALIDATION_MD = "calibrated_label_validation.md"
DELTA_SUMMARY_CSV = "calibration_delta_summary.csv"
PROXY_LEAKAGE_MD = "proxy_leakage_risk.md"
PROMOTION_GATE_MD = "promotion_gate_summary.md"

PLAUSIBLE_RANGES = {
    "chest_cm": (55.0, 170.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (65.0, 180.0),
    "thigh_cm": (30.0, 100.0),
}
DEFAULT_FEATURE_CONFIGS = ["raw_scale_camera", "raw_scale_camera_without_direct_proxies", "normalized_shape"]
DEFAULT_MODEL_TYPES = ["gradient_boosting", "ridge"]


def validate_geometry_calibrated_labels(
    dataset: str | Path,
    phase4a_artifacts: str | Path,
    output_dir: str | Path,
    model_types: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    artifact_path = Path(phase4a_artifacts)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_model_types = model_types or DEFAULT_MODEL_TYPES
    for model_type in selected_model_types:
        validate_model_type(model_type)

    warnings: list[str] = []
    samples = list(SyntheticBodyDataset(dataset_path, split="all"))
    if not samples:
        raise ValueError(f"No samples available for calibrated label validation: {dataset_path}")

    calibrated_rows, row_warning = load_calibrated_label_rows(artifact_path / "calibrated_labels.csv")
    if row_warning:
        warnings.append(row_warning)
    benchmark_summary, benchmark_warning = load_optional_json(artifact_path / "calibrated_benchmark_results.json")
    if benchmark_warning:
        warnings.append(benchmark_warning)

    sample_ids, proxy_names, proxy_matrix, _original_target_matrix = extract_geometry_proxy_matrix(samples)
    aligned = align_calibrated_rows(samples, calibrated_rows)
    original_matrix = label_matrix(aligned, "original")
    calibrated_matrix = label_matrix(aligned, "calibrated")

    validation_rows = build_validation_rows(proxy_names, proxy_matrix, calibrated_matrix, original_matrix)
    delta_rows = build_delta_rows(aligned)
    leakage = run_proxy_leakage_benchmark(samples, calibrated_matrix, selected_model_types, random_state=random_state)
    holdout = summarize_holdout_stability(benchmark_summary)
    promotion = build_promotion_gate_summary(benchmark_summary, leakage)
    summary = {
        "dataset": str(dataset_path),
        "phase4a_artifacts": str(artifact_path),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "sample_count": len(samples),
        "targets": TARGETS,
        "validation_rows": validation_rows,
        "delta_rows": delta_rows,
        "proxy_leakage": leakage,
        "holdout_stability": holdout,
        "promotion": promotion,
        "warnings": warnings,
    }

    paths = {
        "validation_json": output_path / VALIDATION_JSON,
        "validation_csv": output_path / VALIDATION_CSV,
        "validation_md": output_path / VALIDATION_MD,
        "delta_summary_csv": output_path / DELTA_SUMMARY_CSV,
        "proxy_leakage_md": output_path / PROXY_LEAKAGE_MD,
        "promotion_gate_md": output_path / PROMOTION_GATE_MD,
    }
    write_json(paths["validation_json"], summary)
    write_csv(paths["validation_csv"], validation_rows, validation_fieldnames())
    paths["validation_md"].write_text(format_validation_markdown(summary), encoding="utf-8")
    write_csv(paths["delta_summary_csv"], delta_rows, delta_fieldnames())
    paths["proxy_leakage_md"].write_text(format_proxy_leakage_markdown(leakage), encoding="utf-8")
    paths["promotion_gate_md"].write_text(format_promotion_markdown(promotion), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def load_calibrated_label_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        raise FileNotFoundError(f"calibrated_labels.csv does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            converted: dict[str, Any] = dict(row)
            converted["is_ambiguous_phase_3z"] = str(row.get("is_ambiguous_phase_3z", "")).lower() == "true"
            for key, value in row.items():
                if key.endswith("_cm") or key.endswith("_delta_cm"):
                    converted[key] = float(value)
            rows.append(converted)
    return rows, None


def load_optional_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"Optional benchmark artifact is missing: {path}"
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file), None


def align_calibrated_rows(samples: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"Calibrated labels are missing {len(missing)} sample IDs; first missing: {missing[0]}")
    return [by_id[sample["sample_id"]] for sample in samples]


def label_matrix(rows: list[dict[str, Any]], variant: str) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for target in TARGETS:
            prefix = target.removesuffix("_cm")
            values.append(float(row[f"{variant}_{prefix}_cm"]))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def build_validation_rows(
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    calibrated_matrix: np.ndarray,
    original_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        values = calibrated_matrix[:, target_index]
        original_values = original_matrix[:, target_index]
        low, high = PLAUSIBLE_RANGES[target]
        outlier_count = int(((values < low) | (values > high)).sum())
        best_proxy, best_corr, monotonic = best_proxy_monotonicity(target, proxy_names, proxy_matrix, values)
        bucket_counts = bucket_distribution(values)
        rows.append(
            {
                "target": target,
                "count": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "original_mean": float(original_values.mean()),
                "original_std": float(original_values.std()),
                "mean_shift_vs_original": float(values.mean() - original_values.mean()),
                "outlier_count": outlier_count,
                "plausible_min": low,
                "plausible_max": high,
                "low_bucket_count": bucket_counts["low"],
                "mid_bucket_count": bucket_counts["mid"],
                "high_bucket_count": bucket_counts["high"],
                "best_geometry_proxy": best_proxy,
                "best_geometry_correlation": best_corr,
                "monotonic_low_mid_high": monotonic["monotonic"],
                "low_bucket_proxy_mean": monotonic["low_mean"],
                "mid_bucket_proxy_mean": monotonic["mid_mean"],
                "high_bucket_proxy_mean": monotonic["high_mean"],
            }
        )
    return rows


def best_proxy_monotonicity(
    target: str,
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    label_values: np.ndarray,
) -> tuple[str, float, dict[str, Any]]:
    indices = target_proxy_indices(target, proxy_names)
    if not indices:
        return "", 0.0, {"monotonic": False, "low_mean": 0.0, "mid_mean": 0.0, "high_mean": 0.0}
    correlations = [(abs(pearson_correlation(proxy_matrix[:, index], label_values)), index) for index in indices]
    _abs_corr, best_index = max(correlations, key=lambda item: item[0])
    values = proxy_matrix[:, best_index]
    low_threshold, high_threshold = np.quantile(label_values, [1 / 3, 2 / 3])
    low_mask = label_values <= low_threshold
    mid_mask = (label_values > low_threshold) & (label_values <= high_threshold)
    high_mask = label_values > high_threshold
    low_mean = float(values[low_mask].mean())
    mid_mean = float(values[mid_mask].mean())
    high_mean = float(values[high_mask].mean())
    return (
        proxy_names[best_index],
        pearson_correlation(values, label_values),
        {
            "monotonic": bool(low_mean <= mid_mean <= high_mean),
            "low_mean": low_mean,
            "mid_mean": mid_mean,
            "high_mean": high_mean,
        },
    )


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        prefix = target.removesuffix("_cm")
        deltas = np.asarray([row[f"{prefix}_calibration_delta_cm"] for row in rows], dtype=np.float64)
        abs_deltas = np.abs(deltas)
        ambiguous_abs = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in rows if row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        clean_abs = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in rows if not row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        largest = sorted(rows, key=lambda row: row[f"{prefix}_abs_calibration_delta_cm"], reverse=True)[:10]
        output.append(
            {
                "target": target,
                "mean_delta_cm": float(deltas.mean()),
                "mean_abs_delta_cm": float(abs_deltas.mean()),
                "median_abs_delta_cm": float(np.median(abs_deltas)),
                "p90_abs_delta_cm": float(np.percentile(abs_deltas, 90.0)),
                "max_abs_delta_cm": float(abs_deltas.max()),
                "ambiguous_mean_abs_delta_cm": safe_mean(ambiguous_abs),
                "clean_mean_abs_delta_cm": safe_mean(clean_abs),
                "ambiguous_minus_clean_abs_delta_cm": safe_mean(ambiguous_abs) - safe_mean(clean_abs),
                "largest_correction_sample_ids": ";".join(row["sample_id"] for row in largest),
            }
        )
    return output


def run_proxy_leakage_benchmark(
    samples: list[dict[str, Any]],
    calibrated_matrix: np.ndarray,
    model_types: list[str],
    random_state: int,
) -> dict[str, Any]:
    all_features = get_feature_names()
    selected_features = {
        "raw_scale_camera": select_feature_names(all_features, "raw_scale_camera"),
        "raw_scale_camera_without_direct_proxies": remove_direct_calibration_proxy_features(
            select_feature_names(all_features, "raw_scale_camera")
        ),
        "normalized_shape": select_feature_names(all_features, "normalized_shape"),
    }
    split_indices = {split: [i for i, sample in enumerate(samples) if sample["dataset_split"] == split] for split in ("train", "val", "test")}
    split_samples = {split: [samples[index] for index in indices] for split, indices in split_indices.items()}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for feature_config, feature_names in selected_features.items():
        if not feature_names:
            warnings.append(f"Skipped {feature_config}: no features selected.")
            continue
        features_by_split = {split: extract_sample_feature_matrix(split_samples[split], feature_names) for split in ("train", "val", "test")}
        targets_by_split = {split: calibrated_matrix[indices, :] for split, indices in split_indices.items()}
        for model_type in model_types:
            try:
                predictions_by_split = train_and_predict_target_specific(
                    model_type,
                    features_by_split,
                    targets_by_split,
                    feature_names,
                    random_state=random_state,
                )
            except Exception as error:  # pragma: no cover - optional sklearn runtime failures
                warnings.append(f"Skipped {feature_config} {model_type}: {type(error).__name__}: {error}")
                continue
            metrics = evaluate_predictions(targets_by_split, predictions_by_split)
            rows.append(
                {
                    "feature_config": feature_config,
                    "model_type": model_type,
                    "feature_count": len(feature_names),
                    "train_mae": metrics["train"]["overall_mae"],
                    "val_mae": metrics["val"]["overall_mae"],
                    "test_mae": metrics["test"]["overall_mae"],
                    "train_test_gap": metrics["test"]["overall_mae"] - metrics["train"]["overall_mae"],
                }
            )
    best = min(rows, key=lambda row: (float(row["test_mae"]), row["feature_config"], row["model_type"])) if rows else {}
    return {
        "feature_exclusion_rules": direct_proxy_exclusion_tokens(),
        "results": sorted(rows, key=lambda row: (float(row["test_mae"]), row["feature_config"], row["model_type"])),
        "best_result": best,
        "warnings": warnings,
        "risk_level": proxy_leakage_risk_level(rows),
        "interpretation": proxy_leakage_interpretation(rows),
    }


def remove_direct_calibration_proxy_features(feature_names: list[str]) -> list[str]:
    tokens = direct_proxy_exclusion_tokens()
    return [name for name in feature_names if not any(token in name for token in tokens)]


def direct_proxy_exclusion_tokens() -> list[str]:
    return ["bbox_width", "bbox_height", "mask_area", "front_to_side", "area_ratio", "width_ratio", "height_ratio", "aspect_ratio"]


def proxy_leakage_risk_level(rows: list[dict[str, Any]]) -> str:
    by_config = {row["feature_config"]: row for row in rows if row["model_type"] == "gradient_boosting"}
    raw = by_config.get("raw_scale_camera")
    stripped = by_config.get("raw_scale_camera_without_direct_proxies")
    if not raw or not stripped:
        return "unknown"
    if float(stripped["test_mae"]) - float(raw["test_mae"]) > 1.0:
        return "high"
    if float(stripped["test_mae"]) - float(raw["test_mae"]) > 0.3:
        return "moderate"
    return "low"


def proxy_leakage_interpretation(rows: list[dict[str, Any]]) -> str:
    risk = proxy_leakage_risk_level(rows)
    if risk == "high":
        return "Performance drops strongly after removing direct geometry-size proxies; calibrated-label models may be learning proxy formulas."
    if risk == "moderate":
        return "Performance is partly dependent on direct geometry-size proxies; use calibrated labels as synthetic training targets with caution."
    if risk == "low":
        return "Removing direct geometry-size proxies does not sharply degrade MAE, reducing but not eliminating circularity concern."
    return "Leakage risk could not be fully assessed from available runs."


def summarize_holdout_stability(benchmark_summary: dict[str, Any]) -> dict[str, Any]:
    if not benchmark_summary:
        return {"available": False, "warnings": ["Phase 4A benchmark metrics were unavailable."]}
    rows = benchmark_summary.get("benchmark_results", [])
    calibrated = [row for row in rows if row.get("label_variant") == "calibrated_labels"]
    if not calibrated:
        return {"available": False, "warnings": ["No calibrated_labels rows found in Phase 4A benchmark metrics."]}
    best = min(calibrated, key=lambda row: float(row["test_group_mae"]))
    gap = float(best["test_group_mae"]) - float(best["train_group_mae"])
    warnings = []
    if abs(gap) < 0.1:
        warnings.append("Train/test gap is very small; inspect for proxy circularity or overly deterministic labels.")
    if gap > 3.0:
        warnings.append("Train/test gap is large; model may not generalize across the fixed split.")
    return {
        "available": True,
        "best_calibrated_run": best,
        "train_test_gap": gap,
        "warnings": warnings,
    }


def build_promotion_gate_summary(benchmark_summary: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    best_run = {}
    best_per_target: list[dict[str, Any]] = []
    if benchmark_summary:
        best_run = benchmark_summary.get("best_run", {})
        best_per_target = [
            row for row in benchmark_summary.get("best_per_target", [])
            if row.get("label_variant") == "calibrated_labels"
        ]
    return {
        "synthetic_gate": "synthetic_calibrated_strong_candidate" if best_run and float(best_run["test_group_mae"]) <= 3.0 else "synthetic_research_only",
        "real_world_gate": "requires_real_world_calibration_before_production",
        "product_behavior": [
            "Show chest/waist/hip/thigh as AI estimates with confidence.",
            "Require manual confirmation before custom garment production.",
            "Do not use calibrated synthetic predictions as sole cutting instructions.",
        ],
        "proxy_leakage_risk": leakage["risk_level"],
        "best_run": best_run,
        "best_per_target": best_per_target,
    }


def bucket_distribution(values: np.ndarray) -> dict[str, int]:
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    return {
        "low": int((values <= low).sum()),
        "mid": int(((values > low) & (values <= high)).sum()),
        "high": int((values > high).sum()),
    }


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def safe_mean(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(values.mean())


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validation_fieldnames() -> list[str]:
    return [
        "target",
        "count",
        "min",
        "max",
        "mean",
        "std",
        "original_mean",
        "original_std",
        "mean_shift_vs_original",
        "outlier_count",
        "plausible_min",
        "plausible_max",
        "low_bucket_count",
        "mid_bucket_count",
        "high_bucket_count",
        "best_geometry_proxy",
        "best_geometry_correlation",
        "monotonic_low_mid_high",
        "low_bucket_proxy_mean",
        "mid_bucket_proxy_mean",
        "high_bucket_proxy_mean",
    ]


def delta_fieldnames() -> list[str]:
    return [
        "target",
        "mean_delta_cm",
        "mean_abs_delta_cm",
        "median_abs_delta_cm",
        "p90_abs_delta_cm",
        "max_abs_delta_cm",
        "ambiguous_mean_abs_delta_cm",
        "clean_mean_abs_delta_cm",
        "ambiguous_minus_clean_abs_delta_cm",
        "largest_correction_sample_ids",
    ]


def format_validation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 4B Calibrated Label Validation",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Samples: {summary['sample_count']}",
        f"Feature extractor: `{summary['feature_extractor_version']}`",
        "",
        "## Calibrated Label Realism",
        "",
        "| Target | Min | Max | Mean | Std | Outliers | Best Geometry Corr | Monotonic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["validation_rows"]:
        lines.append(
            f"| {row['target']} | {float(row['min']):.2f} | {float(row['max']):.2f} | {float(row['mean']):.2f} | "
            f"{float(row['std']):.2f} | {row['outlier_count']} | {float(row['best_geometry_correlation']):.4f} | {row['monotonic_low_mid_high']} |"
        )
    lines.extend(
        [
            "",
            "## Holdout Stability",
            "",
        ]
    )
    holdout = summary["holdout_stability"]
    if holdout.get("available"):
        best = holdout["best_calibrated_run"]
        lines.extend(
            [
                f"Best calibrated run: `{best['run_name']}`",
                f"Train MAE: {float(best['train_group_mae']):.4f}",
                f"Val MAE: {float(best['val_group_mae']):.4f}",
                f"Test MAE: {float(best['test_group_mae']):.4f}",
                f"Train/test gap: {float(holdout['train_test_gap']):.4f}",
            ]
        )
    if summary["warnings"] or holdout.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in [*summary["warnings"], *holdout.get("warnings", [])])
    return "\n".join(lines) + "\n"


def format_proxy_leakage_markdown(leakage: dict[str, Any]) -> str:
    lines = [
        "# Phase 4B Proxy Leakage Risk",
        "",
        f"Risk level: `{leakage['risk_level']}`",
        "",
        leakage["interpretation"],
        "",
        "Direct proxy exclusion tokens:",
        "",
    ]
    lines.extend(f"- `{token}`" for token in leakage["feature_exclusion_rules"])
    lines.extend(["", "## Feature-Set Comparison", "", "| Feature Config | Model | Feature Count | Train MAE | Val MAE | Test MAE | Train/Test Gap |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in leakage["results"]:
        lines.append(
            f"| {row['feature_config']} | {row['model_type']} | {row['feature_count']} | {float(row['train_mae']):.4f} | "
            f"{float(row['val_mae']):.4f} | {float(row['test_mae']):.4f} | {float(row['train_test_gap']):.4f} |"
        )
    if leakage["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in leakage["warnings"])
    return "\n".join(lines) + "\n"


def format_promotion_markdown(promotion: dict[str, Any]) -> str:
    best = promotion.get("best_run") or {}
    lines = [
        "# Phase 4B Promotion Gate Summary",
        "",
        f"Synthetic gate: `{promotion['synthetic_gate']}`",
        f"Real-world gate: `{promotion['real_world_gate']}`",
        f"Proxy leakage risk: `{promotion['proxy_leakage_risk']}`",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best Phase 4A calibrated-label run: `{best['run_name']}`",
                f"Test group MAE: {float(best['test_group_mae']):.4f}",
                "",
            ]
        )
    lines.extend(["## Product Behavior", ""])
    lines.extend(f"- {item}" for item in promotion["product_behavior"])
    lines.extend(["", "## Gate Definition", "", "- 1-3 cm on synthetic calibrated labels: strong synthetic candidate, not production proof.", "- 3-5 cm: assisted/manual-confirmation candidate.", "- >5 cm: research-only."])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 4A geometry-calibrated labels and promotion gates.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase4a-artifacts", default="artifacts/phase_4a_geometry_calibrated_labels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_TYPES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = validate_geometry_calibrated_labels(
        args.dataset,
        args.phase4a_artifacts,
        args.output,
        model_types=args.models,
        random_state=args.seed,
    )
    promotion = result["summary"]["promotion"]
    print(f"Synthetic gate: {promotion['synthetic_gate']}")
    print(f"Proxy leakage risk: {promotion['proxy_leakage_risk']}")
    print(f"Summary: {result['validation_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
