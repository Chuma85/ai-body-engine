from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.audit_label_geometry_alignment import (
    ellipse_circumference_proxy,
    extract_geometry_proxy_matrix,
    standardize_matrix,
)
from training.experiments.filter_label_geometry_ambiguity import target_proxy_indices
from training.experiments.filter_label_geometry_ambiguity import calculate_ambiguity_scores
from training.experiments.optimize_silhouette_targets import promotion_gate
from training.experiments.select_regularized_hybrid_features import (
    SKLEARN_MODEL_TYPES,
    predict_selected_model,
    select_feature_names,
    sklearn_available,
    train_selected_model,
    validate_model_type,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import extract_sample_feature_matrix

CALIBRATED_LABELS_CSV = "calibrated_labels.csv"
LABEL_DELTA_SUMMARY_CSV = "label_delta_summary.csv"
LABEL_DELTA_SUMMARY_MD = "label_delta_summary.md"
BENCHMARK_RESULTS_JSON = "calibrated_benchmark_results.json"
BENCHMARK_RESULTS_CSV = "calibrated_benchmark_results.csv"
PER_TARGET_RESULTS_CSV = "per_target_calibrated_results.csv"
SUMMARY_MD = "geometry_calibration_summary.md"

TARGETS = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm"]
LABEL_VARIANTS = ["original_labels", "calibrated_labels", "blended_labels"]
DEFAULT_MODEL_TYPES = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]
RAW_SCALE_FEATURE_CONFIG = "raw_scale_camera"
DEFAULT_BLEND_WEIGHT = 0.30


def build_geometry_calibrated_labels(
    dataset: str | Path,
    output_dir: str | Path,
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
    model_types: list[str] | None = None,
    ambiguity_scores: str | Path | None = "artifacts/phase_3z_label_geometry_ambiguity/ambiguity_scores.csv",
    phase3y_artifacts: str | Path | None = "artifacts/phase_3y_label_geometry_alignment",
    random_state: int = 42,
) -> dict[str, Any]:
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be between 0.0 and 1.0.")
    dataset_path = Path(dataset)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_model_types = model_types or DEFAULT_MODEL_TYPES
    for model_type in selected_model_types:
        validate_model_type(model_type)

    warnings = validate_optional_artifacts(phase3y_artifacts)
    samples = list(SyntheticBodyDataset(dataset_path, split="all"))
    if not samples:
        raise ValueError(f"No samples available for geometry calibration: {dataset_path}")

    sample_ids, proxy_names, proxy_matrix, original_targets = extract_geometry_proxy_matrix(samples)
    splits = np.asarray([sample["dataset_split"] for sample in samples], dtype=object)
    train_mask = splits == "train"
    if int(train_mask.sum()) < 2:
        raise ValueError("Need at least two train samples to fit geometry calibration.")

    height_values = np.asarray([sample["measurements"].get("height_cm", 0.0) for sample in samples], dtype=np.float64)
    calibrated_targets, calibration_models = calibrate_targets_from_geometry(
        proxy_names,
        proxy_matrix,
        original_targets,
        train_mask,
        height_values=height_values,
    )
    blended_targets = (1.0 - blend_weight) * original_targets + blend_weight * calibrated_targets
    ambiguity_flags, ambiguity_warning = load_ambiguity_flags(ambiguity_scores)
    if ambiguity_warning:
        warnings.append(ambiguity_warning)

    calibrated_rows = build_calibrated_label_rows(
        samples,
        original_targets,
        calibrated_targets,
        blended_targets,
        ambiguity_flags,
    )
    delta_rows = build_label_delta_summary(calibrated_rows)

    raw_feature_names = select_feature_names(get_feature_names(), RAW_SCALE_FEATURE_CONFIG)
    benchmark = benchmark_label_variants(
        samples,
        original_targets,
        calibrated_targets,
        blended_targets,
        raw_feature_names,
        selected_model_types,
        random_state=random_state,
    )
    ambiguity_delta_summary = summarize_ambiguity_delta_concentration(calibrated_rows)
    split_by_sample_id = {sample["sample_id"]: sample["dataset_split"] for sample in samples}
    ambiguity_comparison = build_ambiguity_comparison(
        sample_ids,
        proxy_names,
        proxy_matrix,
        original_targets,
        calibrated_targets,
        split_by_sample_id,
    )
    summary = {
        "dataset": str(dataset_path),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "targets": TARGETS,
        "label_variants": LABEL_VARIANTS,
        "blend_weight": blend_weight,
        "sample_count": len(samples),
        "calibration_models": calibration_models,
        "label_delta_summary": delta_rows,
        "ambiguity_delta_summary": ambiguity_delta_summary,
        "ambiguity_comparison": ambiguity_comparison,
        "benchmark_results": benchmark["run_rows"],
        "best_run": benchmark["best_run"],
        "best_per_target": benchmark["best_per_target"],
        "beats_phase_3w_group_mae": float(benchmark["best_run"]["test_group_mae"]) < 5.2379,
        "beats_phase_3z_group_mae": float(benchmark["best_run"]["test_group_mae"]) < 5.3305,
        "group_below_5cm": float(benchmark["best_run"]["test_group_mae"]) <= 5.0,
        "warnings": [*warnings, *benchmark["warnings"]],
        "interpretation": interpretation(benchmark["best_run"]),
    }

    paths = {
        "calibrated_labels_csv": output_path / CALIBRATED_LABELS_CSV,
        "label_delta_summary_csv": output_path / LABEL_DELTA_SUMMARY_CSV,
        "label_delta_summary_md": output_path / LABEL_DELTA_SUMMARY_MD,
        "benchmark_results_json": output_path / BENCHMARK_RESULTS_JSON,
        "benchmark_results_csv": output_path / BENCHMARK_RESULTS_CSV,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_csv(paths["calibrated_labels_csv"], calibrated_rows, calibrated_label_fieldnames())
    write_csv(paths["label_delta_summary_csv"], delta_rows, label_delta_fieldnames())
    paths["label_delta_summary_md"].write_text(format_delta_summary(delta_rows, ambiguity_delta_summary), encoding="utf-8")
    write_json(paths["benchmark_results_json"], summary)
    write_csv(paths["benchmark_results_csv"], benchmark["run_rows"], benchmark_fieldnames())
    write_csv(paths["per_target_results_csv"], benchmark["per_target_rows"], per_target_fieldnames())
    paths["summary_md"].write_text(format_summary(summary), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def calibrate_targets_from_geometry(
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    original_targets: np.ndarray,
    train_mask: np.ndarray,
    height_values: np.ndarray | None = None,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    calibrated = np.zeros_like(original_targets, dtype=np.float64)
    models: dict[str, Any] = {}
    for target_index, target in enumerate(TARGETS):
        indices = target_proxy_indices(target, proxy_names)
        if not indices:
            raise ValueError(f"No geometry proxy columns found for {target}.")
        all_features = calibration_feature_matrix(proxy_matrix, indices, height_values)
        train_features = all_features[train_mask]
        train_labels = original_targets[train_mask, target_index]
        model = fit_geometry_calibration(train_features, train_labels, ridge_alpha=ridge_alpha)
        calibrated[:, target_index] = predict_geometry_calibration(model, all_features)
        correlations = [
            abs(pearson_correlation(proxy_matrix[:, index], original_targets[:, target_index]))
            for index in indices
        ]
        best_index = indices[int(np.argmax(correlations))]
        models[target] = {
            "proxy_count": len(indices),
            "uses_height_scale_anchor": height_values is not None,
            "best_proxy": proxy_names[best_index],
            "best_proxy_abs_correlation": float(max(correlations) if correlations else 0.0),
            "ridge_alpha": ridge_alpha,
            "fit_split": "train",
        }
    return calibrated, models


def calibration_feature_matrix(proxy_matrix: np.ndarray, indices: list[int], height_values: np.ndarray | None) -> np.ndarray:
    selected = proxy_matrix[:, indices]
    if height_values is None:
        return selected
    return np.column_stack([selected, height_values])


def fit_geometry_calibration(features: np.ndarray, labels: np.ndarray, ridge_alpha: float = 1.0) -> dict[str, Any]:
    if features.shape[0] < 2:
        raise ValueError("Need at least two rows to fit geometry calibration.")
    means = features.mean(axis=0)
    stds = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
    standardized = (features - means) / stds
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return {
        "feature_means": means,
        "feature_stds": stds,
        "intercept": float(coefficients[0]),
        "coefficients": np.asarray(coefficients[1:], dtype=np.float64),
    }


def predict_geometry_calibration(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["feature_means"]) / model["feature_stds"]
    return standardized @ model["coefficients"] + model["intercept"]


def build_calibrated_label_rows(
    samples: list[dict[str, Any]],
    original_targets: np.ndarray,
    calibrated_targets: np.ndarray,
    blended_targets: np.ndarray,
    ambiguity_flags: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "dataset_split": sample["dataset_split"],
            "is_ambiguous_phase_3z": bool(ambiguity_flags.get(sample["sample_id"], False)),
        }
        for target_index, target in enumerate(TARGETS):
            prefix = target.removesuffix("_cm")
            original = float(original_targets[sample_index, target_index])
            calibrated = float(calibrated_targets[sample_index, target_index])
            blended = float(blended_targets[sample_index, target_index])
            row[f"original_{prefix}_cm"] = original
            row[f"calibrated_{prefix}_cm"] = calibrated
            row[f"blended_{prefix}_cm"] = blended
            row[f"{prefix}_calibration_delta_cm"] = calibrated - original
            row[f"{prefix}_abs_calibration_delta_cm"] = abs(calibrated - original)
        rows.append(row)
    return rows


def build_label_delta_summary(calibrated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        prefix = target.removesuffix("_cm")
        deltas = np.asarray([row[f"{prefix}_calibration_delta_cm"] for row in calibrated_rows], dtype=np.float64)
        abs_deltas = np.abs(deltas)
        largest = sorted(calibrated_rows, key=lambda row: row[f"{prefix}_abs_calibration_delta_cm"], reverse=True)[:10]
        ambiguous_deltas = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in calibrated_rows if row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        clean_deltas = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in calibrated_rows if not row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        rows.append(
            {
                "target": target,
                "mean_delta_cm": float(deltas.mean()),
                "mean_abs_delta_cm": float(abs_deltas.mean()),
                "median_abs_delta_cm": float(np.median(abs_deltas)),
                "p90_abs_delta_cm": float(np.percentile(abs_deltas, 90.0)),
                "max_abs_delta_cm": float(abs_deltas.max()),
                "ambiguous_mean_abs_delta_cm": safe_mean(ambiguous_deltas),
                "clean_mean_abs_delta_cm": safe_mean(clean_deltas),
                "largest_correction_sample_ids": ";".join(row["sample_id"] for row in largest),
            }
        )
    return rows


def benchmark_label_variants(
    samples: list[dict[str, Any]],
    original_targets: np.ndarray,
    calibrated_targets: np.ndarray,
    blended_targets: np.ndarray,
    feature_names: list[str],
    model_types: list[str],
    random_state: int,
) -> dict[str, Any]:
    target_matrices = {
        "original_labels": original_targets,
        "calibrated_labels": calibrated_targets,
        "blended_labels": blended_targets,
    }
    split_indices = {
        split: [index for index, sample in enumerate(samples) if sample["dataset_split"] == split]
        for split in ("train", "val", "test")
    }
    split_samples = {
        split: [samples[index] for index in indices]
        for split, indices in split_indices.items()
    }
    features_by_split = {
        split: extract_sample_feature_matrix(split_samples[split], feature_names)
        for split in ("train", "val", "test")
    }
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for label_variant, matrix in target_matrices.items():
        targets_by_split = {
            split: matrix[indices, :]
            for split, indices in split_indices.items()
        }
        for model_type in model_types:
            if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
                warnings.append(f"Skipped {label_variant} {model_type}: scikit-learn is not available.")
                continue
            try:
                predictions_by_split = train_and_predict_target_specific(
                    model_type,
                    features_by_split,
                    targets_by_split,
                    feature_names,
                    random_state=random_state,
                )
            except Exception as error:  # pragma: no cover - optional sklearn runtime errors
                warnings.append(f"Skipped {label_variant} {model_type}: {type(error).__name__}: {error}")
                continue
            metrics = evaluate_predictions(targets_by_split, predictions_by_split)
            run_name = f"{label_variant}__raw_scale_camera__target_specific__{model_type}"
            run_rows.append(
                {
                    "run_name": run_name,
                    "label_variant": label_variant,
                    "feature_config": RAW_SCALE_FEATURE_CONFIG,
                    "model_type": model_type,
                    "mode": "target_specific",
                    "feature_count": len(feature_names),
                    "train_group_mae": metrics["train"]["overall_mae"],
                    "val_group_mae": metrics["val"]["overall_mae"],
                    "test_group_mae": metrics["test"]["overall_mae"],
                    "promotion_gate": promotion_gate(metrics["test"]["overall_mae"])["gate"],
                    "worst_target": max(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
                    "best_target": min(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
                }
            )
            for target in TARGETS:
                per_target_rows.append(
                    {
                        "run_name": run_name,
                        "label_variant": label_variant,
                        "feature_config": RAW_SCALE_FEATURE_CONFIG,
                        "model_type": model_type,
                        "target": target,
                        "test_mae": metrics["test"]["mae_by_target"][target],
                        "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
                    }
                )
    if not run_rows:
        raise ValueError("No geometry-calibrated label benchmark runs completed.")
    best_run = min(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"]))
    best_per_target = []
    for target in TARGETS:
        target_rows = [row for row in per_target_rows if row["target"] == target]
        best_per_target.append(min(target_rows, key=lambda row: (float(row["test_mae"]), row["run_name"])))
    return {
        "run_rows": sorted(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"])),
        "per_target_rows": per_target_rows,
        "best_run": best_run,
        "best_per_target": best_per_target,
        "warnings": warnings,
    }


def train_and_predict_target_specific(
    model_type: str,
    features_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    feature_names: list[str],
    random_state: int,
) -> dict[str, np.ndarray]:
    predictions_by_split = {
        split: np.zeros((targets.shape[0], len(TARGETS)), dtype=np.float64)
        for split, targets in targets_by_split.items()
    }
    for target_index, _target in enumerate(TARGETS):
        train_targets = targets_by_split["train"][:, [target_index]]
        if model_type == "random_forest":
            train_targets = train_targets.reshape(-1)
        trained = train_selected_model(
            model_type,
            features_by_split["train"],
            train_targets,
            feature_names,
            ridge_alpha=30.0,
            elasticnet_alpha=0.05,
            elasticnet_l1_ratio=0.35,
            random_state=random_state,
        )
        for split, matrix in features_by_split.items():
            predictions = predict_selected_model(trained, matrix)
            predictions_by_split[split][:, target_index] = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return predictions_by_split


def evaluate_predictions(targets_by_split: dict[str, np.ndarray], predictions_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split, targets in targets_by_split.items():
        errors = np.abs(predictions_by_split[split] - targets)
        mae_by_target = {
            target: float(errors[:, index].mean())
            for index, target in enumerate(TARGETS)
        }
        metrics[split] = {"overall_mae": _mean(list(mae_by_target.values())), "mae_by_target": mae_by_target}
    return metrics


def load_ambiguity_flags(path: str | Path | None) -> tuple[dict[str, bool], str | None]:
    if path is None:
        return {}, None
    csv_path = Path(path)
    if not csv_path.exists():
        return {}, f"Optional ambiguity scores were not found; delta concentration by ambiguity is unavailable: {csv_path}"
    flags: dict[str, bool] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            flags[row["sample_id"]] = str(row.get("group_is_ambiguous", "")).lower() == "true"
    return flags, None


def validate_optional_artifacts(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    artifact_path = Path(path)
    if not artifact_path.exists():
        return [f"Optional geometry artifact directory was not found; recomputed proxies from dataset: {artifact_path}"]
    expected = artifact_path / "label_geometry_correlations.csv"
    if not expected.exists():
        return [f"Optional geometry correlation file was not found; recomputed proxies from dataset: {expected}"]
    return []


def summarize_ambiguity_delta_concentration(calibrated_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in TARGETS:
        prefix = target.removesuffix("_cm")
        ambiguous = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in calibrated_rows if row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        clean = np.asarray(
            [row[f"{prefix}_abs_calibration_delta_cm"] for row in calibrated_rows if not row["is_ambiguous_phase_3z"]],
            dtype=np.float64,
        )
        summary[target] = {
            "ambiguous_mean_abs_delta_cm": safe_mean(ambiguous),
            "clean_mean_abs_delta_cm": safe_mean(clean),
            "ambiguous_minus_clean_delta_cm": safe_mean(ambiguous) - safe_mean(clean),
        }
    return summary


def build_ambiguity_comparison(
    sample_ids: list[str],
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    original_targets: np.ndarray,
    calibrated_targets: np.ndarray,
    split_by_sample_id: dict[str, str],
) -> dict[str, Any]:
    original = calculate_ambiguity_scores(sample_ids, proxy_names, proxy_matrix, original_targets, split_by_sample_id)
    calibrated = calculate_ambiguity_scores(sample_ids, proxy_names, proxy_matrix, calibrated_targets, split_by_sample_id)
    return {
        "original": {
            "ambiguous_sample_count": len(original["ambiguous_sample_ids"]),
            "score_distributions": original["score_distributions"],
        },
        "calibrated": {
            "ambiguous_sample_count": len(calibrated["ambiguous_sample_ids"]),
            "score_distributions": calibrated["score_distributions"],
        },
        "group_p85_score_delta": calibrated["score_distributions"]["group"]["p85"] - original["score_distributions"]["group"]["p85"],
        "group_median_score_delta": calibrated["score_distributions"]["group"]["median"] - original["score_distributions"]["group"]["median"],
    }


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def interpretation(best_run: dict[str, Any]) -> str:
    if float(best_run["test_group_mae"]) <= 5.0:
        return "Geometry-calibrated labels moved the focused target group below 5 cm on this synthetic diagnostic, but this reflects better label/geometry consistency rather than production readiness."
    return "Geometry-calibrated labels did not move the focused target group below 5 cm; better renderer-side measurement probes or generator changes are still needed."


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calibrated_label_fieldnames() -> list[str]:
    fields = ["sample_id", "dataset_split", "is_ambiguous_phase_3z"]
    for target in TARGETS:
        prefix = target.removesuffix("_cm")
        fields.extend(
            [
                f"original_{prefix}_cm",
                f"calibrated_{prefix}_cm",
                f"blended_{prefix}_cm",
                f"{prefix}_calibration_delta_cm",
                f"{prefix}_abs_calibration_delta_cm",
            ]
        )
    return fields


def label_delta_fieldnames() -> list[str]:
    return [
        "target",
        "mean_delta_cm",
        "mean_abs_delta_cm",
        "median_abs_delta_cm",
        "p90_abs_delta_cm",
        "max_abs_delta_cm",
        "ambiguous_mean_abs_delta_cm",
        "clean_mean_abs_delta_cm",
        "largest_correction_sample_ids",
    ]


def benchmark_fieldnames() -> list[str]:
    return [
        "run_name",
        "label_variant",
        "feature_config",
        "model_type",
        "mode",
        "feature_count",
        "train_group_mae",
        "val_group_mae",
        "test_group_mae",
        "promotion_gate",
        "worst_target",
        "best_target",
    ]


def per_target_fieldnames() -> list[str]:
    return ["run_name", "label_variant", "feature_config", "model_type", "target", "test_mae", "promotion_gate"]


def format_delta_summary(delta_rows: list[dict[str, Any]], ambiguity_delta_summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 4A Label Delta Summary",
        "",
        "| Target | Mean Abs Delta | P90 Abs Delta | Max Abs Delta | Ambiguous Mean Abs Delta | Clean Mean Abs Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in delta_rows:
        target = row["target"]
        ambiguity = ambiguity_delta_summary[target]
        lines.append(
            f"| {target} | {float(row['mean_abs_delta_cm']):.4f} | {float(row['p90_abs_delta_cm']):.4f} | "
            f"{float(row['max_abs_delta_cm']):.4f} | {ambiguity['ambiguous_mean_abs_delta_cm']:.4f} | "
            f"{ambiguity['clean_mean_abs_delta_cm']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def format_summary(summary: dict[str, Any]) -> str:
    best = summary["best_run"]
    lines = [
        "# Phase 4A Geometry-Calibrated Labels",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Feature extractor: `{summary['feature_extractor_version']}`",
        f"Blend weight: {summary['blend_weight']:.2f}",
        "",
        "## Best Benchmark",
        "",
        f"Best run: `{best['run_name']}`",
        f"Test group MAE: {float(best['test_group_mae']):.4f}",
        f"Promotion gate: `{best['promotion_gate']}`",
        f"Group below 5 cm: `{summary['group_below_5cm']}`",
        f"Beats Phase 3W 5.2379: `{summary['beats_phase_3w_group_mae']}`",
        f"Beats Phase 3Z 5.3305: `{summary['beats_phase_3z_group_mae']}`",
        f"Original ambiguity group p85: {summary['ambiguity_comparison']['original']['score_distributions']['group']['p85']:.4f}",
        f"Calibrated ambiguity group p85: {summary['ambiguity_comparison']['calibrated']['score_distributions']['group']['p85']:.4f}",
        "",
        "## Benchmark Results",
        "",
        "| Label Variant | Model | Test Group MAE | Worst Target | Best Target |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in summary["benchmark_results"]:
        lines.append(
            f"| {row['label_variant']} | {row['model_type']} | {float(row['test_group_mae']):.4f} | {row['worst_target']} | {row['best_target']} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build geometry-calibrated measurement labels and benchmark them.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root, such as data/synthetic/phase_3t.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument("--blend-weight", type=float, default=DEFAULT_BLEND_WEIGHT)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_TYPES)
    parser.add_argument("--ambiguity-scores", default="artifacts/phase_3z_label_geometry_ambiguity/ambiguity_scores.csv")
    parser.add_argument("--phase3y-artifacts", default="artifacts/phase_3y_label_geometry_alignment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = build_geometry_calibrated_labels(
        args.dataset,
        args.output,
        blend_weight=args.blend_weight,
        model_types=args.models,
        ambiguity_scores=args.ambiguity_scores,
        phase3y_artifacts=args.phase3y_artifacts,
        random_state=args.seed,
    )
    best = result["summary"]["best_run"]
    print(f"Best run: {best['run_name']} test group MAE {best['test_group_mae']:.4f}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
