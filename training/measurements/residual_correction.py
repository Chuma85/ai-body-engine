from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.build_geometry_calibrated_labels import TARGETS
from training.experiments.optimize_silhouette_targets import promotion_gate
from training.experiments.select_regularized_hybrid_features import (
    SKLEARN_MODEL_TYPES,
    predict_selected_model,
    select_feature_names,
    sklearn_available,
    train_selected_model,
    validate_model_type,
)
from training.features.image_silhouette_features import get_feature_names
from training.measurements.geometry_measurement_estimator import (
    DEFAULT_CALIBRATED_LABELS,
    DEFAULT_PHASE4A_RESULTS,
    align_calibrated_labels,
    compare_against_phase4a,
    extract_estimator_components,
    fit_estimator_coefficients,
    load_calibrated_labels,
    load_optional_json,
    predict_estimator,
    quality_flag_counts,
    target_matrix,
)
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import extract_sample_feature_matrix

RESIDUAL_TRAINING_SUMMARY_JSON = "residual_training_summary.json"
RESIDUAL_TRAINING_SUMMARY_CSV = "residual_training_summary.csv"
RESIDUAL_BENCHMARK_JSON = "residual_benchmark_results.json"
RESIDUAL_BENCHMARK_CSV = "residual_benchmark_results.csv"
PER_TARGET_RESULTS_CSV = "per_target_residual_results.csv"
RESIDUAL_DISTRIBUTION_CSV = "residual_distribution.csv"
SUMMARY_MD = "estimator_plus_residual_summary.md"

DEFAULT_MODEL_TYPES = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]
REALISTIC_RANGES = {
    "chest_cm": (55.0, 170.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (65.0, 180.0),
    "thigh_cm": (30.0, 100.0),
}
LARGE_RESIDUAL_THRESHOLD_CM = 10.0


def run_residual_correction(
    dataset: str | Path,
    output_dir: str | Path,
    calibrated_labels: str | Path = DEFAULT_CALIBRATED_LABELS,
    phase4a_results: str | Path = DEFAULT_PHASE4A_RESULTS,
    model_types: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_models = model_types or DEFAULT_MODEL_TYPES
    for model_type in selected_models:
        validate_model_type(model_type)

    samples = list(SyntheticBodyDataset(dataset_path, split="all"))
    if not samples:
        raise ValueError(f"No samples available for residual correction: {dataset_path}")
    calibrated_rows = align_calibrated_labels(samples, load_calibrated_labels(calibrated_labels))
    calibrated_targets = target_matrix(calibrated_rows, "calibrated")
    phase4a_summary, phase4a_warning = load_optional_json(phase4a_results)
    warnings = [phase4a_warning] if phase4a_warning else []

    component_rows = extract_estimator_components(samples)
    train_indices = split_indices(samples)["train"]
    geometry_coefficients = fit_estimator_coefficients(component_rows, calibrated_targets, train_indices, ridge_alpha=0.1)
    geometry_predictions, geometry_rows = predict_estimator(samples, component_rows, geometry_coefficients)
    residual_targets = calculate_residuals(geometry_predictions, calibrated_targets)

    feature_names, feature_matrix = build_residual_feature_matrix(samples, geometry_rows)
    split_index_map = split_indices(samples)
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = build_residual_distribution_rows(samples, residual_targets, geometry_predictions)
    skipped_runs: list[dict[str, Any]] = []

    direct_metrics = evaluate_final_predictions(geometry_predictions, calibrated_targets, samples)
    phase4a_comparison = compare_against_phase4a(phase4a_summary, direct_metrics)

    for model_type in selected_models:
        if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
            skipped_runs.append({"model_type": model_type, "reason": "scikit-learn is not available"})
            continue
        try:
            residual_predictions = train_predict_residuals(
                model_type,
                feature_matrix,
                residual_targets,
                split_index_map,
                feature_names,
                random_state=random_state,
            )
        except Exception as error:  # pragma: no cover - optional sklearn runtime failures
            skipped_runs.append({"model_type": model_type, "reason": f"{type(error).__name__}: {error}"})
            continue
        final_predictions = final_estimates(geometry_predictions, residual_predictions)
        metrics = evaluate_final_predictions(final_predictions, calibrated_targets, samples)
        residual_metrics = evaluate_residual_predictions(residual_predictions, residual_targets, samples)
        run_name = f"geometry_plus_residual__{model_type}"
        run_rows.append(
            {
                "run_name": run_name,
                "model_type": model_type,
                "feature_count": len(feature_names),
                "train_group_mae": metrics["train"]["overall_mae"],
                "val_group_mae": metrics["val"]["overall_mae"],
                "test_group_mae": metrics["test"]["overall_mae"],
                "test_residual_mae": residual_metrics["test"]["overall_mae"],
                "direct_estimator_test_mae": direct_metrics["test"]["overall_mae"],
                "phase4a_ml_test_mae": phase4a_comparison.get("geometry_calibrated_ml_model", {}).get("test_group_mae", ""),
                "beats_direct_estimator": metrics["test"]["overall_mae"] < direct_metrics["test"]["overall_mae"],
                "beats_phase4a_ml": bool(phase4a_comparison.get("geometry_calibrated_ml_model")) and metrics["test"]["overall_mae"] < float(phase4a_comparison["geometry_calibrated_ml_model"]["test_group_mae"]),
                "promotion_gate": promotion_gate(metrics["test"]["overall_mae"])["gate"],
                "worst_target": max(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
                "best_target": min(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
            }
        )
        per_target_rows.extend(build_per_target_rows(run_name, model_type, metrics, residual_metrics))
        prediction_rows.extend(
            build_prediction_rows(
                run_name,
                model_type,
                samples,
                geometry_rows,
                geometry_predictions,
                calibrated_targets,
                residual_targets,
                residual_predictions,
                final_predictions,
            )
        )

    if not run_rows:
        raise ValueError("No residual correction runs completed.")
    best_run = min(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"]))
    summary = {
        "dataset": str(dataset_path),
        "targets": TARGETS,
        "sample_count": len(samples),
        "residual_feature_names": feature_names,
        "direct_estimator_metrics": direct_metrics,
        "phase4a_comparison": phase4a_comparison,
        "best_run": best_run,
        "benchmark_results": sorted(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"])),
        "residual_distribution": distribution_rows,
        "quality_flag_counts": quality_flag_counts(geometry_rows),
        "skipped_runs": skipped_runs,
        "warnings": warnings,
        "interpretation": interpretation(best_run, phase4a_comparison),
    }

    paths = {
        "residual_training_summary_json": output_path / RESIDUAL_TRAINING_SUMMARY_JSON,
        "residual_training_summary_csv": output_path / RESIDUAL_TRAINING_SUMMARY_CSV,
        "residual_benchmark_results_json": output_path / RESIDUAL_BENCHMARK_JSON,
        "residual_benchmark_results_csv": output_path / RESIDUAL_BENCHMARK_CSV,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "residual_distribution_csv": output_path / RESIDUAL_DISTRIBUTION_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_json(paths["residual_training_summary_json"], summary)
    write_csv(paths["residual_training_summary_csv"], prediction_rows, prediction_fieldnames())
    write_json(paths["residual_benchmark_results_json"], summary)
    write_csv(paths["residual_benchmark_results_csv"], summary["benchmark_results"], benchmark_fieldnames())
    write_csv(paths["per_target_results_csv"], per_target_rows, per_target_fieldnames())
    write_csv(paths["residual_distribution_csv"], distribution_rows, residual_distribution_fieldnames())
    paths["summary_md"].write_text(format_summary(summary), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def calculate_residuals(geometry_estimates: np.ndarray, calibrated_labels: np.ndarray) -> np.ndarray:
    return calibrated_labels - geometry_estimates


def final_estimates(geometry_estimates: np.ndarray, predicted_residuals: np.ndarray) -> np.ndarray:
    return geometry_estimates + predicted_residuals


def split_indices(samples: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        split: [index for index, sample in enumerate(samples) if sample["dataset_split"] == split]
        for split in ("train", "val", "test")
    }


def build_residual_feature_matrix(samples: list[dict[str, Any]], geometry_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    raw_feature_names = select_feature_names(get_feature_names(), "raw_scale_camera")
    raw_features = extract_sample_feature_matrix(samples, raw_feature_names)
    geometry_feature_names = residual_geometry_feature_names()
    geometry_features = np.asarray(
        [[float(row[name]) for name in geometry_feature_names] for row in geometry_rows],
        dtype=np.float64,
    )
    feature_names = [f"raw__{name}" for name in raw_feature_names] + [f"geometry__{name}" for name in geometry_feature_names]
    return feature_names, np.column_stack([raw_features, geometry_features])


def residual_geometry_feature_names() -> list[str]:
    names = ["scale_factor_cm_per_px"]
    for target in TARGETS:
        names.extend(
            [
                f"{target}_estimated_cm",
                f"{target}_front_width_cm",
                f"{target}_side_depth_cm",
                f"{target}_ellipse_proxy",
                f"{target}_local_area_proxy",
                f"{target}_front_side_ratio",
            ]
        )
    return names


def train_predict_residuals(
    model_type: str,
    feature_matrix: np.ndarray,
    residual_targets: np.ndarray,
    split_index_map: dict[str, list[int]],
    feature_names: list[str],
    random_state: int,
) -> np.ndarray:
    predictions = np.zeros_like(residual_targets, dtype=np.float64)
    train_features = feature_matrix[split_index_map["train"], :]
    for target_index, _target in enumerate(TARGETS):
        train_targets = residual_targets[split_index_map["train"], target_index]
        fit_targets = train_targets.reshape(-1, 1)
        if model_type == "random_forest":
            fit_targets = train_targets
        trained = train_selected_model(
            model_type,
            train_features,
            fit_targets,
            feature_names,
            ridge_alpha=30.0,
            elasticnet_alpha=0.05,
            elasticnet_l1_ratio=0.35,
            random_state=random_state,
        )
        predicted = predict_selected_model(trained, feature_matrix)
        predictions[:, target_index] = np.asarray(predicted, dtype=np.float64).reshape(-1)
    return predictions


def evaluate_final_predictions(predictions: np.ndarray, targets: np.ndarray, samples: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split, indices in split_indices(samples).items():
        errors = np.abs(predictions[indices, :] - targets[indices, :])
        mae_by_target = {target: float(errors[:, target_index].mean()) for target_index, target in enumerate(TARGETS)}
        metrics[split] = {"overall_mae": _mean(list(mae_by_target.values())), "mae_by_target": mae_by_target}
    return metrics


def evaluate_residual_predictions(predictions: np.ndarray, targets: np.ndarray, samples: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_final_predictions(predictions, targets, samples)


def build_per_target_rows(run_name: str, model_type: str, metrics: dict[str, Any], residual_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        rows.append(
            {
                "run_name": run_name,
                "model_type": model_type,
                "target": target,
                "train_mae": metrics["train"]["mae_by_target"][target],
                "val_mae": metrics["val"]["mae_by_target"][target],
                "test_mae": metrics["test"]["mae_by_target"][target],
                "test_residual_mae": residual_metrics["test"]["mae_by_target"][target],
                "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
            }
        )
    return rows


def build_prediction_rows(
    run_name: str,
    model_type: str,
    samples: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    geometry_estimates: np.ndarray,
    calibrated_targets: np.ndarray,
    residual_targets: np.ndarray,
    predicted_residuals: np.ndarray,
    final_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for sample_index, sample in enumerate(samples):
        for target_index, target in enumerate(TARGETS):
            predicted_residual = float(predicted_residuals[sample_index, target_index])
            final_estimate = float(final_predictions[sample_index, target_index])
            flags = residual_quality_flags(target, geometry_rows[sample_index]["quality_flags"], predicted_residual, final_estimate)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset_split": sample["dataset_split"],
                    "target": target,
                    "model_name": model_type,
                    "run_name": run_name,
                    "geometry_estimate_cm": float(geometry_estimates[sample_index, target_index]),
                    "calibrated_label_cm": float(calibrated_targets[sample_index, target_index]),
                    "residual_cm": float(residual_targets[sample_index, target_index]),
                    "predicted_residual_cm": predicted_residual,
                    "final_estimate_cm": final_estimate,
                    "abs_error_cm": abs(final_estimate - float(calibrated_targets[sample_index, target_index])),
                    "confidence_flags": flags,
                    "geometry_quality_flags": geometry_rows[sample_index]["quality_flags"],
                }
            )
    return rows


def residual_quality_flags(target: str, geometry_flags: str, predicted_residual: float, final_estimate: float) -> str:
    flags = [] if geometry_flags == "ok" else [f"geometry_{flag}" for flag in geometry_flags.split(";") if flag]
    if abs(predicted_residual) > LARGE_RESIDUAL_THRESHOLD_CM:
        flags.append("large_residual_correction")
    low, high = REALISTIC_RANGES[target]
    if final_estimate < low or final_estimate > high:
        flags.append("final_estimate_out_of_range")
    return ";".join(sorted(set(flags))) if flags else "ok"


def build_residual_distribution_rows(
    samples: list[dict[str, Any]],
    residual_targets: np.ndarray,
    geometry_estimates: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for split, indices in split_indices(samples).items():
        for target_index, target in enumerate(TARGETS):
            residuals = residual_targets[indices, target_index]
            geometry_values = geometry_estimates[indices, target_index]
            rows.append(
                {
                    "dataset_split": split,
                    "target": target,
                    "residual_mean_cm": float(residuals.mean()),
                    "residual_std_cm": float(residuals.std()),
                    "residual_mean_abs_cm": float(np.abs(residuals).mean()),
                    "residual_p90_abs_cm": float(np.percentile(np.abs(residuals), 90.0)),
                    "geometry_estimate_mean_cm": float(geometry_values.mean()),
                    "large_residual_count": int((np.abs(residuals) > LARGE_RESIDUAL_THRESHOLD_CM).sum()),
                }
            )
    return rows


def interpretation(best_run: dict[str, Any], phase4a_comparison: dict[str, Any]) -> str:
    if not phase4a_comparison.get("available"):
        return "Residual correction improved the geometry estimator, but Phase 4A ML comparison was unavailable."
    phase4a_mae = float(phase4a_comparison["geometry_calibrated_ml_model"]["test_group_mae"])
    best_mae = float(best_run["test_group_mae"])
    if best_mae <= phase4a_mae:
        return "The hybrid geometry plus residual model matches or beats the Phase 4A calibrated ML benchmark while preserving per-sample geometry explanations."
    return "Residual correction improves the direct geometry estimator, but the Phase 4A calibrated ML benchmark still performs better."


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_fieldnames() -> list[str]:
    return [
        "run_name",
        "model_type",
        "feature_count",
        "train_group_mae",
        "val_group_mae",
        "test_group_mae",
        "test_residual_mae",
        "direct_estimator_test_mae",
        "phase4a_ml_test_mae",
        "beats_direct_estimator",
        "beats_phase4a_ml",
        "promotion_gate",
        "worst_target",
        "best_target",
    ]


def per_target_fieldnames() -> list[str]:
    return ["run_name", "model_type", "target", "train_mae", "val_mae", "test_mae", "test_residual_mae", "promotion_gate"]


def prediction_fieldnames() -> list[str]:
    return [
        "sample_id",
        "dataset_split",
        "target",
        "model_name",
        "run_name",
        "geometry_estimate_cm",
        "calibrated_label_cm",
        "residual_cm",
        "predicted_residual_cm",
        "final_estimate_cm",
        "abs_error_cm",
        "confidence_flags",
        "geometry_quality_flags",
    ]


def residual_distribution_fieldnames() -> list[str]:
    return [
        "dataset_split",
        "target",
        "residual_mean_cm",
        "residual_std_cm",
        "residual_mean_abs_cm",
        "residual_p90_abs_cm",
        "geometry_estimate_mean_cm",
        "large_residual_count",
    ]


def format_summary(summary: dict[str, Any]) -> str:
    direct = summary["direct_estimator_metrics"]["test"]["overall_mae"]
    best = summary["best_run"]
    phase4a = summary["phase4a_comparison"].get("geometry_calibrated_ml_model", {})
    phase4a_mae = phase4a.get("test_group_mae", "")
    lines = [
        "# Phase 4D Geometry Residual Correction",
        "",
        f"Dataset: `{summary['dataset']}`",
        "",
        "## Benchmark Results",
        "",
        f"Direct geometry estimator test MAE: {float(direct):.4f}",
    ]
    if phase4a_mae != "":
        lines.append(f"Phase 4A calibrated ML test MAE: {float(phase4a_mae):.4f}")
    lines.extend(
        [
            f"Best residual run: `{best['run_name']}`",
            f"Best residual test MAE: {float(best['test_group_mae']):.4f}",
            "",
            "| Model | Train MAE | Val MAE | Test MAE | Residual MAE | Beats Direct | Beats Phase 4A ML |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["benchmark_results"]:
        lines.append(
            f"| {row['model_type']} | {float(row['train_group_mae']):.4f} | {float(row['val_group_mae']):.4f} | "
            f"{float(row['test_group_mae']):.4f} | {float(row['test_residual_mae']):.4f} | {row['beats_direct_estimator']} | {row['beats_phase4a_ml']} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    if summary["skipped_runs"] or summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {row['model_type']}: {row['reason']}" for row in summary["skipped_runs"])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train learned residual correction on top of geometry measurements.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrated-labels", default=DEFAULT_CALIBRATED_LABELS)
    parser.add_argument("--phase4a-results", default=DEFAULT_PHASE4A_RESULTS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_TYPES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = run_residual_correction(
        args.dataset,
        args.output,
        calibrated_labels=args.calibrated_labels,
        phase4a_results=args.phase4a_results,
        model_types=args.models,
        random_state=args.seed,
    )
    best = result["summary"]["best_run"]
    print(f"Best residual run: {best['run_name']} test MAE {best['test_group_mae']:.4f}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
