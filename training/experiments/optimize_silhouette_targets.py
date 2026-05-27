from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.experiments.select_regularized_hybrid_features import (
    SKLEARN_MODEL_TYPES,
    load_drift_scores,
    load_split_samples,
    predict_selected_model,
    select_feature_names,
    sklearn_available,
    train_selected_model,
    validate_feature_config,
    validate_model_type,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix

SILHOUETTE_TARGETS = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm", "shoulder_cm", "calf_cm"]
EXCLUDED_TARGETS = ["height_cm", "weight_kg", "inseam_cm", "sleeve_cm", "neck_cm"]
PHASE_3V_SILHOUETTE_MAE = 5.3132
BENCHMARK_RESULTS_JSON = "benchmark_results.json"
BENCHMARK_RESULTS_CSV = "benchmark_results.csv"
PER_TARGET_RESULTS_CSV = "per_target_results.csv"
BEST_MODEL_PER_TARGET_CSV = "best_model_per_target.csv"
ERROR_ANALYSIS_CSV = "error_analysis.csv"
PROMOTION_GATE_MD = "promotion_gate_summary.md"

DEFAULT_MODEL_FEATURE_COMBOS = [
    ("raw_scale_camera", "ridge"),
    ("raw_scale_camera", "elasticnet"),
    ("raw_scale_camera", "random_forest"),
    ("raw_scale_camera", "gradient_boosting"),
    ("normalized_shape", "ridge"),
    ("combined_hybrid_without_area_ratios", "random_forest"),
    ("selected_low_drift_features", "elasticnet"),
]


def optimize_silhouette_targets(
    dataset: str | Path,
    output_dir: str | Path,
    drift_csv: str | Path | None = None,
    cnn_metrics: str | Path | None = "artifacts/deep/phase_3t_dual_branch_augmented/metrics.json",
    ridge_alpha: float = 30.0,
    elasticnet_alpha: float = 0.05,
    elasticnet_l1_ratio: float = 0.35,
    random_state: int = 42,
) -> dict[str, Any]:
    feature_names = get_feature_names()
    drift_scores = load_drift_scores(drift_csv) if drift_csv else {}
    selected_features = {
        config_name: select_feature_names(feature_names, config_name, drift_scores=drift_scores, drift_threshold=1.0)
        for config_name, _model_type in DEFAULT_MODEL_FEATURE_COMBOS
    }
    split_samples = load_split_samples(Path(dataset))
    targets_by_split = {split: _target_matrix(samples, SILHOUETTE_TARGETS) for split, samples in split_samples.items()}

    feature_matrices: dict[str, dict[str, np.ndarray]] = {}
    for config_name, names in selected_features.items():
        feature_matrices[config_name] = {
            split: extract_sample_feature_matrix(samples, names)
            for split, samples in split_samples.items()
        }

    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []

    for feature_config, model_type in DEFAULT_MODEL_FEATURE_COMBOS:
        validate_feature_config(feature_config)
        validate_model_type(model_type)
        if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
            skipped_runs.append({"feature_config": feature_config, "model_type": model_type, "reason": "scikit-learn is not available"})
            continue
        names = selected_features[feature_config]
        matrices = feature_matrices[feature_config]

        multi_result = train_and_evaluate_run(
            mode="multi_output",
            feature_config=feature_config,
            model_type=model_type,
            feature_names=names,
            feature_matrices=matrices,
            targets_by_split=targets_by_split,
            split_samples=split_samples,
            ridge_alpha=ridge_alpha,
            elasticnet_alpha=elasticnet_alpha,
            elasticnet_l1_ratio=elasticnet_l1_ratio,
            random_state=random_state,
        )
        run_rows.append(multi_result["run_row"])
        per_target_rows.extend(multi_result["per_target_rows"])
        error_rows.extend(multi_result["error_rows"])

        target_result = train_and_evaluate_target_specific_run(
            feature_config=feature_config,
            model_type=model_type,
            feature_names=names,
            feature_matrices=matrices,
            targets_by_split=targets_by_split,
            split_samples=split_samples,
            ridge_alpha=ridge_alpha,
            elasticnet_alpha=elasticnet_alpha,
            elasticnet_l1_ratio=elasticnet_l1_ratio,
            random_state=random_state,
        )
        run_rows.append(target_result["run_row"])
        per_target_rows.extend(target_result["per_target_rows"])
        error_rows.extend(target_result["error_rows"])

    cnn_rows = load_existing_cnn_metrics(cnn_metrics)
    if cnn_rows:
        run_rows.append(cnn_rows["run_row"])
        per_target_rows.extend(cnn_rows["per_target_rows"])

    best_per_target = select_best_per_target(per_target_rows)
    best_run = min(run_rows, key=lambda row: (float(row["silhouette_group_mae"]), row["run_name"]))
    summary = {
        "dataset": str(dataset),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "silhouette_targets": SILHOUETTE_TARGETS,
        "excluded_targets": EXCLUDED_TARGETS,
        "phase_3v_silhouette_mae": PHASE_3V_SILHOUETTE_MAE,
        "best_run": best_run,
        "beats_phase_3v_silhouette_mae": float(best_run["silhouette_group_mae"]) < PHASE_3V_SILHOUETTE_MAE,
        "best_model_per_target": best_per_target,
        "benchmark_results": sorted(run_rows, key=lambda row: (float(row["silhouette_group_mae"]), row["run_name"])),
        "skipped_runs": skipped_runs,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(output_path, summary, per_target_rows, error_rows)
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def train_and_evaluate_run(
    mode: str,
    feature_config: str,
    model_type: str,
    feature_names: list[str],
    feature_matrices: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    split_samples: dict[str, list[dict[str, Any]]],
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    random_state: int,
) -> dict[str, Any]:
    trained = train_selected_model(
        model_type,
        feature_matrices["train"],
        targets_by_split["train"],
        feature_names,
        ridge_alpha=ridge_alpha,
        elasticnet_alpha=elasticnet_alpha,
        elasticnet_l1_ratio=elasticnet_l1_ratio,
        random_state=random_state,
    )
    predictions_by_split = {
        split: predict_selected_model(trained, matrix)
        for split, matrix in feature_matrices.items()
    }
    return evaluate_predictions(
        run_name=f"{mode}__{feature_config}__{model_type}",
        mode=mode,
        feature_config=feature_config,
        model_type=model_type,
        feature_count=len(feature_names),
        predictions_by_split=predictions_by_split,
        targets_by_split=targets_by_split,
        split_samples=split_samples,
    )


def train_and_evaluate_target_specific_run(
    feature_config: str,
    model_type: str,
    feature_names: list[str],
    feature_matrices: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    split_samples: dict[str, list[dict[str, Any]]],
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    random_state: int,
) -> dict[str, Any]:
    predictions_by_split = {
        split: np.zeros((targets.shape[0], len(SILHOUETTE_TARGETS)), dtype=np.float64)
        for split, targets in targets_by_split.items()
    }
    for target_index, _target in enumerate(SILHOUETTE_TARGETS):
        trained = train_selected_model(
            model_type,
            feature_matrices["train"],
            targets_by_split["train"][:, [target_index]],
            feature_names,
            ridge_alpha=ridge_alpha,
            elasticnet_alpha=elasticnet_alpha,
            elasticnet_l1_ratio=elasticnet_l1_ratio,
            random_state=random_state,
        )
        for split, matrix in feature_matrices.items():
            predictions = predict_selected_model(trained, matrix)
            predictions_by_split[split][:, target_index] = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return evaluate_predictions(
        run_name=f"target_specific__{feature_config}__{model_type}",
        mode="target_specific",
        feature_config=feature_config,
        model_type=model_type,
        feature_count=len(feature_names),
        predictions_by_split=predictions_by_split,
        targets_by_split=targets_by_split,
        split_samples=split_samples,
    )


def evaluate_predictions(
    run_name: str,
    mode: str,
    feature_config: str,
    model_type: str,
    feature_count: int,
    predictions_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    split_samples: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metrics = {}
    for split, targets in targets_by_split.items():
        errors = np.abs(predictions_by_split[split] - targets)
        metrics[split] = {
            "overall_mae": _mean([float(errors[:, index].mean()) for index in range(len(SILHOUETTE_TARGETS))]),
            "mae_by_target": {
                target: float(errors[:, index].mean())
                for index, target in enumerate(SILHOUETTE_TARGETS)
            },
        }
    run_row = {
        "run_name": run_name,
        "mode": mode,
        "feature_config": feature_config,
        "model_type": model_type,
        "feature_count": feature_count,
        "train_mae": metrics["train"]["overall_mae"],
        "val_mae": metrics["val"]["overall_mae"],
        "silhouette_group_mae": metrics["test"]["overall_mae"],
        "promotion_gate": promotion_gate(metrics["test"]["overall_mae"])["gate"],
        "worst_target": max(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
        "best_target": min(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
    }
    per_target_rows = [
        {
            "run_name": run_name,
            "mode": mode,
            "feature_config": feature_config,
            "model_type": model_type,
            "target": target,
            "test_mae": metrics["test"]["mae_by_target"][target],
            "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
        }
        for target in SILHOUETTE_TARGETS
    ]
    error_rows = build_error_analysis_rows(run_name, mode, feature_config, model_type, predictions_by_split["test"], targets_by_split["test"], split_samples["test"])
    return {"run_row": run_row, "per_target_rows": per_target_rows, "error_rows": error_rows}


def build_error_analysis_rows(
    run_name: str,
    mode: str,
    feature_config: str,
    model_type: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    heights = np.asarray([sample["measurements"].get("height_cm", 0.0) for sample in samples], dtype=np.float64)
    body_shapes = [sample.get("body_shape", "") for sample in samples]
    for target_index, target in enumerate(SILHOUETTE_TARGETS):
        target_values = targets[:, target_index]
        pred_values = predictions[:, target_index]
        signed_errors = pred_values - target_values
        abs_errors = np.abs(signed_errors)
        worst_indices = list(np.argsort(abs_errors)[::-1][:20])
        row = {
            "run_name": run_name,
            "mode": mode,
            "feature_config": feature_config,
            "model_type": model_type,
            "target": target,
            "mae": float(abs_errors.mean()),
            "signed_bias": float(signed_errors.mean()),
            "underprediction_count": int((signed_errors < 0).sum()),
            "overprediction_count": int((signed_errors > 0).sum()),
            "small_measurement_mae": bucket_mae(target_values, abs_errors, "low"),
            "mid_measurement_mae": bucket_mae(target_values, abs_errors, "mid"),
            "large_measurement_mae": bucket_mae(target_values, abs_errors, "high"),
            "short_height_mae": bucket_mae(heights, abs_errors, "low"),
            "mid_height_mae": bucket_mae(heights, abs_errors, "mid"),
            "tall_height_mae": bucket_mae(heights, abs_errors, "high"),
            "worst_sample_ids": ";".join(samples[index]["sample_id"] for index in worst_indices),
        }
        for shape in sorted({shape for shape in body_shapes if shape}):
            shape_errors = [float(abs_errors[index]) for index, value in enumerate(body_shapes) if value == shape]
            row[f"body_shape_{shape}_mae"] = _mean(shape_errors) if shape_errors else ""
        rows.append(row)
    return rows


def bucket_mae(bucket_values: np.ndarray, errors: np.ndarray, bucket: str) -> float:
    low, high = np.quantile(bucket_values, [1 / 3, 2 / 3])
    if bucket == "low":
        mask = bucket_values <= low
    elif bucket == "mid":
        mask = (bucket_values > low) & (bucket_values <= high)
    elif bucket == "high":
        mask = bucket_values > high
    else:
        raise ValueError(f"Unknown bucket: {bucket}")
    if not bool(mask.any()):
        return 0.0
    return float(errors[mask].mean())


def load_existing_cnn_metrics(cnn_metrics: str | Path | None) -> dict[str, Any] | None:
    if cnn_metrics is None:
        return None
    path = Path(cnn_metrics)
    if not path.exists():
        return None
    metrics = read_json(path)
    test_by_target = metrics.get("test", {}).get("mae_by_target", {})
    if not all(target in test_by_target for target in SILHOUETTE_TARGETS):
        return None
    run_name = path.parent.name
    per_target_rows = [
        {
            "run_name": run_name,
            "mode": "existing_cnn_filtered",
            "feature_config": "front_side_images",
            "model_type": metrics.get("model_type", "cnn"),
            "target": target,
            "test_mae": float(test_by_target[target]),
            "promotion_gate": promotion_gate(float(test_by_target[target]))["gate"],
        }
        for target in SILHOUETTE_TARGETS
    ]
    group_mae = _mean([float(test_by_target[target]) for target in SILHOUETTE_TARGETS])
    run_row = {
        "run_name": run_name,
        "mode": "existing_cnn_filtered",
        "feature_config": "front_side_images",
        "model_type": metrics.get("model_type", "cnn"),
        "feature_count": "",
        "train_mae": "",
        "val_mae": "",
        "silhouette_group_mae": group_mae,
        "promotion_gate": promotion_gate(group_mae)["gate"],
        "worst_target": max(per_target_rows, key=lambda row: row["test_mae"])["target"],
        "best_target": min(per_target_rows, key=lambda row: row["test_mae"])["target"],
    }
    return {"run_row": run_row, "per_target_rows": per_target_rows}


def select_best_per_target(per_target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows = []
    for target in SILHOUETTE_TARGETS:
        rows = [row for row in per_target_rows if row["target"] == target]
        if rows:
            best = min(rows, key=lambda row: (float(row["test_mae"]), row["run_name"]))
            best_rows.append({**best, "recommendation": target_recommendation(float(best["test_mae"]))})
    return best_rows


def promotion_gate(mae: float) -> dict[str, str]:
    if mae > 5.0:
        return {"gate": "research_only", "description": "Research-only: MAE is above 5 cm."}
    if mae > 3.0:
        return {"gate": "assisted_manual_confirmation", "description": "Assisted/manual confirmation candidate: MAE is 3-5 cm."}
    if mae >= 1.0:
        return {"gate": "stronger_candidate", "description": "Stronger production candidate: MAE is 1-3 cm."}
    return {"gate": "below_defined_gate", "description": "Verify evaluation quality before interpreting."}


def target_recommendation(mae: float) -> str:
    gate = promotion_gate(mae)["gate"]
    if gate == "research_only":
        return "Keep research-only for this target."
    if gate == "assisted_manual_confirmation":
        return "Candidate for assisted measurement with manual confirmation."
    if gate == "stronger_candidate":
        return "Promising target; validate on real data before promotion."
    return "Verify metrics before using."


def write_outputs(
    output_path: Path,
    summary: dict[str, Any],
    per_target_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "benchmark_results_json": output_path / BENCHMARK_RESULTS_JSON,
        "benchmark_results_csv": output_path / BENCHMARK_RESULTS_CSV,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "best_model_per_target_csv": output_path / BEST_MODEL_PER_TARGET_CSV,
        "error_analysis_csv": output_path / ERROR_ANALYSIS_CSV,
        "promotion_gate_summary_md": output_path / PROMOTION_GATE_MD,
    }
    write_json(paths["benchmark_results_json"], summary)
    write_csv(paths["benchmark_results_csv"], summary["benchmark_results"], benchmark_fieldnames())
    write_csv(paths["per_target_results_csv"], per_target_rows, per_target_fieldnames())
    write_csv(paths["best_model_per_target_csv"], summary["best_model_per_target"], best_fieldnames())
    write_csv(paths["error_analysis_csv"], error_rows, error_fieldnames(error_rows))
    paths["promotion_gate_summary_md"].write_text(format_promotion_summary(summary), encoding="utf-8")
    return paths


def format_promotion_summary(summary: dict[str, Any]) -> str:
    best = summary["best_run"]
    lines = [
        "# Phase 3W Silhouette Target Optimization",
        "",
        f"Best run: `{best['run_name']}`",
        f"Best silhouette group MAE: {float(best['silhouette_group_mae']):.4f}",
        f"Phase 3V silhouette group MAE: {summary['phase_3v_silhouette_mae']:.4f}",
        f"Beats Phase 3V: `{summary['beats_phase_3v_silhouette_mae']}`",
        f"Group gate: `{best['promotion_gate']}`",
        "",
        "## Best Per Target",
        "",
        "| Target | Best Run | Model | Feature Config | MAE | Gate |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in summary["best_model_per_target"]:
        lines.append(
            f"| {row['target']} | {row['run_name']} | {row['model_type']} | {row['feature_config']} | {float(row['test_mae']):.4f} | {row['promotion_gate']} |"
        )
    if summary["skipped_runs"]:
        lines.extend(["", "## Skipped Runs", ""])
        lines.extend(f"- {row['feature_config']} + {row['model_type']}: {row['reason']}" for row in summary["skipped_runs"])
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def benchmark_fieldnames() -> list[str]:
    return [
        "run_name",
        "mode",
        "feature_config",
        "model_type",
        "feature_count",
        "train_mae",
        "val_mae",
        "silhouette_group_mae",
        "promotion_gate",
        "worst_target",
        "best_target",
    ]


def per_target_fieldnames() -> list[str]:
    return ["run_name", "mode", "feature_config", "model_type", "target", "test_mae", "promotion_gate"]


def best_fieldnames() -> list[str]:
    return [*per_target_fieldnames(), "recommendation"]


def error_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    base = [
        "run_name",
        "mode",
        "feature_config",
        "model_type",
        "target",
        "mae",
        "signed_bias",
        "underprediction_count",
        "overprediction_count",
        "small_measurement_mae",
        "mid_measurement_mae",
        "large_measurement_mae",
        "short_height_mae",
        "mid_height_mae",
        "tall_height_mae",
        "worst_sample_ids",
    ]
    extras = sorted({key for row in rows for key in row if key.startswith("body_shape_")})
    return [*base, *extras]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize models for silhouette-learnable measurement targets only.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root, such as data/synthetic/phase_3t.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument("--drift-csv", help="Optional feature drift CSV for selected_low_drift_features.")
    parser.add_argument("--cnn-metrics", default="artifacts/deep/phase_3t_dual_branch_augmented/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = optimize_silhouette_targets(
        args.dataset,
        args.output,
        drift_csv=args.drift_csv,
        cnn_metrics=args.cnn_metrics,
        random_state=args.seed,
    )
    best = result["summary"]["best_run"]
    print(f"Best run: {best['run_name']} silhouette MAE {best['silhouette_group_mae']:.4f}")
    print(f"Summary: {result['benchmark_results_json']}")
    print(f"Promotion gates: {result['promotion_gate_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
