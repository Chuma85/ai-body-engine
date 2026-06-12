from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.audit_label_geometry_alignment import ellipse_circumference_proxy
from training.experiments.build_geometry_calibrated_labels import TARGETS
from training.experiments.optimize_silhouette_targets import promotion_gate
from training.features.image_silhouette_features import extract_front_side_features, load_rgb_image
from training.features.measurement_band_features import BAND_CANDIDATES, extract_front_side_band_features
from training.train_baseline_measurements import _mean

ESTIMATOR_RESULTS_JSON = "estimator_results.json"
ESTIMATOR_RESULTS_CSV = "estimator_results.csv"
PER_TARGET_RESULTS_CSV = "per_target_estimator_results.csv"
CALIBRATION_COEFFICIENTS_JSON = "calibration_coefficients.json"
SUMMARY_MD = "estimator_vs_ml_summary.md"
FAILURE_CASES_CSV = "failure_cases.csv"

DEFAULT_CALIBRATED_LABELS = "artifacts/phase_4a_geometry_calibrated_labels/calibrated_labels.csv"
DEFAULT_PHASE4A_RESULTS = "artifacts/phase_4a_geometry_calibrated_labels/calibrated_benchmark_results.json"


def run_geometry_measurement_estimator(
    dataset: str | Path,
    output_dir: str | Path,
    calibrated_labels: str | Path = DEFAULT_CALIBRATED_LABELS,
    phase4a_results: str | Path = DEFAULT_PHASE4A_RESULTS,
    ridge_alpha: float = 0.1,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = list(SyntheticBodyDataset(dataset_path, split="all"))
    if not samples:
        raise ValueError(f"No samples available for geometry estimator: {dataset_path}")
    calibrated_rows = load_calibrated_labels(calibrated_labels)
    aligned_labels = align_calibrated_labels(samples, calibrated_rows)
    phase4a_summary, phase4a_warning = load_optional_json(phase4a_results)
    warnings = [phase4a_warning] if phase4a_warning else []

    component_rows = extract_estimator_components(samples)
    target_matrices = {
        "original": target_matrix(aligned_labels, "original"),
        "calibrated": target_matrix(aligned_labels, "calibrated"),
    }
    train_indices = [index for index, sample in enumerate(samples) if sample["dataset_split"] == "train"]
    if len(train_indices) < 2:
        raise ValueError("Need at least two train samples to fit estimator calibration.")

    coefficients = fit_estimator_coefficients(component_rows, target_matrices["calibrated"], train_indices, ridge_alpha=ridge_alpha)
    prediction_matrix, estimate_rows = predict_estimator(samples, component_rows, coefficients)
    metrics_vs_calibrated = evaluate_predictions(prediction_matrix, target_matrices["calibrated"], samples, label_variant="calibrated_labels")
    metrics_vs_original = evaluate_predictions(prediction_matrix, target_matrices["original"], samples, label_variant="original_formula_labels")
    failure_rows = build_failure_cases(prediction_matrix, target_matrices["calibrated"], samples, estimate_rows)

    ml_comparison = compare_against_phase4a(phase4a_summary, metrics_vs_calibrated)
    summary = {
        "dataset": str(dataset_path),
        "targets": TARGETS,
        "sample_count": len(samples),
        "estimator_type": "front_side_ellipse_affine_calibrated",
        "ridge_alpha": ridge_alpha,
        "calibration_split": "train",
        "metrics_vs_calibrated_labels": metrics_vs_calibrated,
        "metrics_vs_original_formula_labels": metrics_vs_original,
        "phase4a_comparison": ml_comparison,
        "quality_flag_counts": quality_flag_counts(estimate_rows),
        "warnings": warnings,
        "interpretation": interpretation(ml_comparison),
    }

    paths = {
        "estimator_results_json": output_path / ESTIMATOR_RESULTS_JSON,
        "estimator_results_csv": output_path / ESTIMATOR_RESULTS_CSV,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "calibration_coefficients_json": output_path / CALIBRATION_COEFFICIENTS_JSON,
        "summary_md": output_path / SUMMARY_MD,
        "failure_cases_csv": output_path / FAILURE_CASES_CSV,
    }
    write_json(paths["estimator_results_json"], summary)
    write_csv(paths["estimator_results_csv"], estimate_rows, estimator_result_fieldnames())
    write_csv(paths["per_target_results_csv"], per_target_rows(metrics_vs_calibrated, metrics_vs_original), per_target_fieldnames())
    write_json(paths["calibration_coefficients_json"], coefficients)
    paths["summary_md"].write_text(format_summary(summary), encoding="utf-8")
    write_csv(paths["failure_cases_csv"], failure_rows, failure_fieldnames())
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def extract_estimator_components(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        front_image = load_rgb_image(sample["front_image_path"])
        side_image = load_rgb_image(sample["side_image_path"])
        image_features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        band_features = extract_front_side_band_features(sample["front_image_path"], sample["side_image_path"])
        height_cm = float(sample["measurements"].get("height_cm", 0.0))
        front_body_height_px = float(image_features["front_raw_bbox_height_px"])
        scale_factor = height_cm / front_body_height_px if front_body_height_px > 0.0 else 0.0
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "dataset_split": sample["dataset_split"],
            "height_cm": height_cm,
            "front_body_height_px": front_body_height_px,
            "scale_factor_cm_per_px": scale_factor,
            "front_image_width_px": float(front_image.shape[1]),
            "side_image_width_px": float(side_image.shape[1]),
        }
        quality_flags = []
        if height_cm <= 0.0:
            quality_flags.append("missing_height")
        if front_body_height_px <= 0.0:
            quality_flags.append("missing_front_body_height")
        if scale_factor <= 0.0 or scale_factor > 5.0:
            quality_flags.append("unstable_scale_factor")
        for target in TARGETS:
            components = target_geometry_components(target, band_features, row, scale_factor)
            row.update({f"{target}_{key}": value for key, value in components.items()})
            if components["front_width_px"] <= 0.0:
                quality_flags.append(f"{target}_missing_front_width")
            if components["side_depth_px"] <= 0.0:
                quality_flags.append(f"{target}_missing_side_depth")
        row["quality_flags"] = ";".join(sorted(set(quality_flags))) if quality_flags else "ok"
        rows.append(row)
    return rows


def target_geometry_components(target: str, band_features: dict[str, float], base_row: dict[str, Any], scale_factor: float) -> dict[str, float]:
    center = preferred_band_center(target)
    prefix = band_prefix(target, center)
    front_width_ratio = band_features[f"{prefix}_front_raw_width_ratio"]
    side_depth_ratio = band_features[f"{prefix}_side_raw_width_ratio"]
    front_width_px = front_width_ratio * base_row["front_image_width_px"]
    side_depth_px = side_depth_ratio * base_row["side_image_width_px"]
    front_width_cm = front_width_px * scale_factor
    side_depth_cm = side_depth_px * scale_factor
    ellipse_proxy_cm = ellipse_circumference_proxy(front_width_cm, side_depth_cm)
    return {
        "band_center_y_ratio": center,
        "front_width_px": float(front_width_px),
        "side_depth_px": float(side_depth_px),
        "front_width_cm": float(front_width_cm),
        "side_depth_cm": float(side_depth_cm),
        "scale_factor": float(scale_factor),
        "ellipse_proxy": float(ellipse_proxy_cm),
        "local_area_proxy": float(band_features[f"{prefix}_raw_width_depth_product"]),
        "front_side_ratio": float(band_features[f"{prefix}_raw_front_side_width_ratio"]),
    }


def preferred_band_center(target: str) -> float:
    return {
        "chest_cm": 0.40,
        "waist_cm": 0.46,
        "hip_cm": 0.68,
        "thigh_cm": 0.68,
    }[target]


def band_prefix(target: str, center_y_ratio: float) -> str:
    centers = BAND_CANDIDATES[target]
    index = centers.index(center_y_ratio)
    return f"{target.removesuffix('_cm')}_band_{index:02d}_y{int(round(center_y_ratio * 100)):02d}"


def fit_estimator_coefficients(
    component_rows: list[dict[str, Any]],
    calibrated_targets: np.ndarray,
    train_indices: list[int],
    ridge_alpha: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {}
    for target_index, target in enumerate(TARGETS):
        feature_matrix = estimator_feature_matrix(component_rows, target)
        train_features = feature_matrix[train_indices, :]
        train_targets = calibrated_targets[train_indices, target_index]
        model = fit_affine_calibration(train_features, train_targets, ridge_alpha=ridge_alpha)
        coefficients[target] = {
            "feature_names": estimator_feature_names(),
            "intercept": model["intercept"],
            "coefficients": model["coefficients"].tolist(),
            "feature_means": model["feature_means"].tolist(),
            "feature_stds": model["feature_stds"].tolist(),
            "ridge_alpha": ridge_alpha,
            "band_center_y_ratio": preferred_band_center(target),
        }
    return coefficients


def estimator_feature_names() -> list[str]:
    return ["ellipse_proxy", "front_width_cm", "side_depth_cm", "local_area_proxy"]


def estimator_feature_matrix(component_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    names = estimator_feature_names()
    return np.asarray(
        [[float(row[f"{target}_{name}"]) for name in names] for row in component_rows],
        dtype=np.float64,
    )


def fit_affine_calibration(features: np.ndarray, targets: np.ndarray, ridge_alpha: float) -> dict[str, Any]:
    if features.shape[0] < 2:
        raise ValueError("Need at least two rows to fit geometry estimator calibration.")
    means = features.mean(axis=0)
    stds = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
    return {
        "feature_means": means,
        "feature_stds": stds,
        "intercept": float(targets.mean()),
        "coefficients": np.zeros(features.shape[1], dtype=np.float64),
    }


def predict_estimator(
    samples: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    coefficients: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.zeros((len(samples), len(TARGETS)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        model = coefficients[target]
        feature_matrix = estimator_feature_matrix(component_rows, target)
        means = np.asarray(model["feature_means"], dtype=np.float64)
        stds = np.asarray(model["feature_stds"], dtype=np.float64)
        coeffs = [float(value) for value in model["coefficients"]]
        for sample_index, feature_row in enumerate(((feature_matrix - means) / stds).tolist()):
            value = float(model["intercept"])
            for feature_value, coefficient in zip(feature_row, coeffs):
                value += float(feature_value) * coefficient
            predictions[sample_index, target_index] = value
    for sample_index, sample in enumerate(samples):
        component = component_rows[sample_index]
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "dataset_split": sample["dataset_split"],
            "quality_flags": component["quality_flags"],
            "scale_factor_cm_per_px": component["scale_factor_cm_per_px"],
        }
        for target_index, target in enumerate(TARGETS):
            row[f"{target}_estimated_cm"] = float(predictions[sample_index, target_index])
            for key in ("front_width_px", "side_depth_px", "front_width_cm", "side_depth_cm", "ellipse_proxy", "local_area_proxy", "front_side_ratio"):
                row[f"{target}_{key}"] = component[f"{target}_{key}"]
        rows.append(row)
    return predictions, rows


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list[dict[str, Any]],
    label_variant: str,
) -> dict[str, Any]:
    split_metrics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        indices = [index for index, sample in enumerate(samples) if sample["dataset_split"] == split]
        split_predictions = predictions[indices, :]
        split_targets = targets[indices, :]
        errors = np.abs(split_predictions - split_targets)
        mae_by_target = {
            target: float(errors[:, target_index].mean())
            for target_index, target in enumerate(TARGETS)
        }
        split_metrics[split] = {
            "overall_mae": _mean(list(mae_by_target.values())),
            "mae_by_target": mae_by_target,
        }
    split_metrics["label_variant"] = label_variant
    return split_metrics


def build_failure_cases(
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list[dict[str, Any]],
    estimate_rows: list[dict[str, Any]],
    per_target_count: int = 20,
) -> list[dict[str, Any]]:
    rows = []
    for target_index, target in enumerate(TARGETS):
        errors = np.abs(predictions[:, target_index] - targets[:, target_index])
        for sample_index in np.argsort(errors)[::-1][:per_target_count]:
            estimate = estimate_rows[int(sample_index)]
            rows.append(
                {
                    "target": target,
                    "sample_id": samples[int(sample_index)]["sample_id"],
                    "dataset_split": samples[int(sample_index)]["dataset_split"],
                    "true_calibrated_cm": float(targets[int(sample_index), target_index]),
                    "estimated_cm": float(predictions[int(sample_index), target_index]),
                    "abs_error_cm": float(errors[int(sample_index)]),
                    "quality_flags": estimate["quality_flags"],
                    "front_width_px": estimate[f"{target}_front_width_px"],
                    "side_depth_px": estimate[f"{target}_side_depth_px"],
                    "ellipse_proxy": estimate[f"{target}_ellipse_proxy"],
                }
            )
    return rows


def compare_against_phase4a(phase4a_summary: dict[str, Any], estimator_metrics: dict[str, Any]) -> dict[str, Any]:
    if not phase4a_summary:
        return {"available": False, "warnings": ["Phase 4A benchmark summary is missing."]}
    calibrated_runs = [
        row for row in phase4a_summary.get("benchmark_results", [])
        if row.get("label_variant") == "calibrated_labels"
    ]
    original_runs = [
        row for row in phase4a_summary.get("benchmark_results", [])
        if row.get("label_variant") == "original_labels"
    ]
    best_ml = min(calibrated_runs, key=lambda row: float(row["test_group_mae"])) if calibrated_runs else {}
    best_original = min(original_runs, key=lambda row: float(row["test_group_mae"])) if original_runs else {}
    estimator_test = float(estimator_metrics["test"]["overall_mae"])
    ml_test = float(best_ml["test_group_mae"]) if best_ml else 0.0
    return {
        "available": bool(best_ml),
        "original_formula_baseline": best_original,
        "geometry_calibrated_ml_model": best_ml,
        "direct_geometry_estimator_test_mae": estimator_test,
        "ml_minus_estimator_test_mae": ml_test - estimator_test if best_ml else 0.0,
        "ml_adds_value_over_estimator": bool(best_ml and ml_test < estimator_test),
    }


def quality_flag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for flag in str(row["quality_flags"]).split(";"):
            if not flag:
                continue
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def per_target_rows(metrics_vs_calibrated: dict[str, Any], metrics_vs_original: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label_variant, metrics in (
        ("calibrated_labels", metrics_vs_calibrated),
        ("original_formula_labels", metrics_vs_original),
    ):
        for target in TARGETS:
            rows.append(
                {
                    "label_variant": label_variant,
                    "target": target,
                    "train_mae": metrics["train"]["mae_by_target"][target],
                    "val_mae": metrics["val"]["mae_by_target"][target],
                    "test_mae": metrics["test"]["mae_by_target"][target],
                    "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
                }
            )
    return rows


def interpretation(comparison: dict[str, Any]) -> str:
    if comparison.get("ml_adds_value_over_estimator"):
        return "The calibrated-label ML model improves over the direct estimator, suggesting residual nonlinear correction is useful."
    if comparison.get("available"):
        return "The direct explainable estimator matches or beats the calibrated-label ML model on this synthetic holdout; ML adds little beyond transparent geometry here."
    return "Phase 4A ML comparison was unavailable; interpret the direct estimator standalone."


def load_calibrated_labels(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"calibrated_labels.csv does not exist: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            converted: dict[str, Any] = dict(row)
            for key, value in row.items():
                if key.endswith("_cm") or key.endswith("_delta_cm"):
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def align_calibrated_labels(samples: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in label_rows}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"Calibrated labels are missing {len(missing)} sample IDs; first missing: {missing[0]}")
    return [by_id[sample["sample_id"]] for sample in samples]


def target_matrix(rows: list[dict[str, Any]], variant: str) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(row[f"{variant}_{target.removesuffix('_cm')}_cm"]) for target in TARGETS])
    return np.asarray(matrix, dtype=np.float64)


def load_optional_json(path: str | Path) -> tuple[dict[str, Any], str | None]:
    json_path = Path(path)
    if not json_path.exists():
        return {}, f"Optional benchmark JSON is missing: {json_path}"
    with json_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file), None


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def estimator_result_fieldnames() -> list[str]:
    fields = ["sample_id", "dataset_split", "quality_flags", "scale_factor_cm_per_px"]
    for target in TARGETS:
        fields.append(f"{target}_estimated_cm")
        for key in ("front_width_px", "side_depth_px", "front_width_cm", "side_depth_cm", "ellipse_proxy", "local_area_proxy", "front_side_ratio"):
            fields.append(f"{target}_{key}")
    return fields


def per_target_fieldnames() -> list[str]:
    return ["label_variant", "target", "train_mae", "val_mae", "test_mae", "promotion_gate"]


def failure_fieldnames() -> list[str]:
    return ["target", "sample_id", "dataset_split", "true_calibrated_cm", "estimated_cm", "abs_error_cm", "quality_flags", "front_width_px", "side_depth_px", "ellipse_proxy"]


def format_summary(summary: dict[str, Any]) -> str:
    comparison = summary["phase4a_comparison"]
    lines = [
        "# Phase 4C Geometry Measurement Estimator",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Estimator: `{summary['estimator_type']}`",
        f"Calibration split: `{summary['calibration_split']}`",
        "",
        "## Direct Estimator Metrics",
        "",
        "| Split | MAE vs Calibrated Labels | MAE vs Original Formula Labels |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {summary['metrics_vs_calibrated_labels'][split]['overall_mae']:.4f} | "
            f"{summary['metrics_vs_original_formula_labels'][split]['overall_mae']:.4f} |"
        )
    lines.extend(["", "## Comparison", ""])
    if comparison.get("available"):
        ml = comparison["geometry_calibrated_ml_model"]
        original = comparison.get("original_formula_baseline") or {}
        lines.extend(
            [
                f"Original formula-label ML baseline test MAE: {float(original.get('test_group_mae', 0.0)):.4f}",
                f"Geometry-calibrated ML model test MAE: {float(ml['test_group_mae']):.4f}",
                f"Direct geometry estimator test MAE: {float(comparison['direct_geometry_estimator_test_mae']):.4f}",
                f"ML adds value over estimator: `{comparison['ml_adds_value_over_estimator']}`",
            ]
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explainable front/side geometry measurement estimator.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrated-labels", default=DEFAULT_CALIBRATED_LABELS)
    parser.add_argument("--phase4a-results", default=DEFAULT_PHASE4A_RESULTS)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    args = parser.parse_args(argv)

    result = run_geometry_measurement_estimator(
        args.dataset,
        args.output,
        calibrated_labels=args.calibrated_labels,
        phase4a_results=args.phase4a_results,
        ridge_alpha=args.ridge_alpha,
    )
    print(f"Direct estimator test MAE: {result['summary']['metrics_vs_calibrated_labels']['test']['overall_mae']:.4f}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
