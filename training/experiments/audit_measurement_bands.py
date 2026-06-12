from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.optimize_silhouette_targets import bucket_mae, promotion_gate
from training.experiments.select_regularized_hybrid_features import (
    SKLEARN_MODEL_TYPES,
    load_split_samples,
    predict_selected_model,
    sklearn_available,
    train_selected_model,
)
from training.features.image_silhouette_features import (
    FEATURE_EXTRACTOR_VERSION as V5_FEATURE_VERSION,
    extract_front_side_features,
    feature_vector,
    get_feature_names,
)
from training.features.measurement_band_features import (
    FEATURE_EXTRACTOR_VERSION as BAND_FEATURE_VERSION,
    MEASUREMENT_BAND_TARGETS,
    candidate_band_definitions,
    extract_front_side_band_features,
    get_band_feature_names,
)
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import _target_matrix

BAND_CORRELATIONS_JSON = "band_correlations.json"
BAND_CORRELATIONS_CSV = "band_correlations.csv"
BAND_CORRELATIONS_MD = "band_correlations.md"
BAND_BENCHMARK_JSON = "band_feature_benchmark_results.json"
BAND_BENCHMARK_CSV = "band_feature_benchmark_results.csv"
PER_TARGET_BAND_RESULTS_CSV = "per_target_band_results.csv"
WORST_PREDICTIONS_CSV = "worst_predictions.csv"
SUMMARY_MD = "measurement_band_summary.md"
CONTACT_SHEET_DIR = "contact_sheets"

MODEL_TYPES = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]
FEATURE_SETS = ["v5_existing", "v6_bands", "v5_plus_v6"]
PHASE_3W_TARGET_BASELINES = {
    "chest_cm": 5.4918,
    "waist_cm": 5.6725,
    "hip_cm": 6.1704,
    "thigh_cm": 5.7530,
}


def audit_measurement_bands(
    dataset: str | Path,
    output_dir: str | Path,
    prediction_csv: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_samples = list(SyntheticBodyDataset(dataset, split="all"))
    split_samples = load_split_samples(Path(dataset))
    band_feature_names = get_band_feature_names()

    sample_ids, band_matrix, target_matrix = extract_band_matrix(all_samples, band_feature_names)
    correlation_rows = build_band_correlation_rows(band_matrix, target_matrix, band_feature_names)
    correlation_summary = summarize_band_correlations(correlation_rows)

    benchmark = run_band_feature_benchmark(split_samples, random_state=random_state)
    contact_warnings = write_measurement_band_contact_sheets(all_samples, output_path / CONTACT_SHEET_DIR)
    prediction_warnings = optional_prediction_warnings(prediction_csv)
    summary = {
        "dataset": str(dataset),
        "v5_feature_version": V5_FEATURE_VERSION,
        "band_feature_version": BAND_FEATURE_VERSION,
        "targets": MEASUREMENT_BAND_TARGETS,
        "sample_count": len(all_samples),
        "feature_sets": FEATURE_SETS,
        "model_types": MODEL_TYPES,
        "correlation_summary": correlation_summary,
        "benchmark": benchmark["summary"],
        "warnings": [*contact_warnings, *prediction_warnings],
    }

    paths = write_outputs(output_path, summary, correlation_rows, benchmark)
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def extract_band_matrix(
    samples: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids: list[str] = []
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    for sample in samples:
        features = extract_front_side_band_features(sample["front_image_path"], sample["side_image_path"])
        sample_ids.append(sample["sample_id"])
        feature_rows.append([float(features[name]) for name in feature_names])
        target_rows.append([float(sample["measurements"][target]) for target in MEASUREMENT_BAND_TARGETS])
    return sample_ids, np.asarray(feature_rows, dtype=np.float64), np.asarray(target_rows, dtype=np.float64)


def build_band_correlation_rows(
    band_matrix: np.ndarray,
    target_matrix: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = {definition["band_name"]: definition for definition in candidate_band_definitions()}
    for target_index, target in enumerate(MEASUREMENT_BAND_TARGETS):
        target_values = target_matrix[:, target_index]
        target_prefix = target.removesuffix("_cm")
        for feature_index, feature_name in enumerate(feature_names):
            if not feature_name.startswith(f"{target_prefix}_band_"):
                continue
            band_key = "_".join(feature_name.split("_")[:4])
            definition = definitions[band_key]
            corr = pearson_correlation(band_matrix[:, feature_index], target_values)
            rows.append(
                {
                    "target": target,
                    "band_name": band_key,
                    "center_y_ratio": definition["center_y_ratio"],
                    "feature": feature_name,
                    "feature_role": band_feature_role(feature_name),
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    return rows


def pearson_correlation(values: np.ndarray, target_values: np.ndarray) -> float:
    left = [float(value) for value in values.tolist()]
    right = [float(value) for value in target_values.tolist()]
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_sum = sum(value * value for value in left_centered)
    right_sum = sum(value * value for value in right_centered)
    if left_sum < 1e-12 or right_sum < 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(left_centered, right_centered)) / ((left_sum * right_sum) ** 0.5)


def band_feature_role(feature_name: str) -> str:
    if "_front_" in feature_name:
        return "front"
    if "_side_" in feature_name:
        return "side"
    if "_product" in feature_name or "_front_side_" in feature_name:
        return "front_side_combined"
    return "band_metadata"


def summarize_band_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in MEASUREMENT_BAND_TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        strongest = sorted(target_rows, key=lambda row: row["abs_correlation"], reverse=True)
        role_best = {}
        for role in ("front", "side", "front_side_combined"):
            role_rows = [row for row in target_rows if row["feature_role"] == role]
            if role_rows:
                role_best[role] = max(role_rows, key=lambda row: row["abs_correlation"])
        band_scores = []
        for band_name in sorted({row["band_name"] for row in target_rows}):
            band_rows = [row for row in target_rows if row["band_name"] == band_name]
            best = max(band_rows, key=lambda row: row["abs_correlation"])
            band_scores.append(
                {
                    "band_name": band_name,
                    "center_y_ratio": best["center_y_ratio"],
                    "best_abs_correlation": best["abs_correlation"],
                    "best_feature": best["feature"],
                    "weak_band": best["abs_correlation"] < 0.20,
                }
            )
        summary[target] = {
            "best_overall": strongest[0] if strongest else None,
            "best_by_role": role_best,
            "band_scores": sorted(band_scores, key=lambda row: row["best_abs_correlation"], reverse=True),
            "weak_or_unstable_bands": [row for row in band_scores if row["weak_band"]],
        }
    return summary


def run_band_feature_benchmark(
    split_samples: dict[str, list[dict[str, Any]]],
    random_state: int,
) -> dict[str, Any]:
    matrices_by_feature_set = build_feature_set_matrices(split_samples)
    targets_by_split = {split: _target_matrix(samples, MEASUREMENT_BAND_TARGETS) for split, samples in split_samples.items()}
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []

    for feature_set in FEATURE_SETS:
        feature_names = matrices_by_feature_set[feature_set]["feature_names"]
        for model_type in MODEL_TYPES:
            if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
                skipped_runs.append({"feature_set": feature_set, "model_type": model_type, "reason": "scikit-learn is not available"})
                continue
            predictions_by_split = {
                split: np.zeros((targets.shape[0], len(MEASUREMENT_BAND_TARGETS)), dtype=np.float64)
                for split, targets in targets_by_split.items()
            }
            for target_index, _target in enumerate(MEASUREMENT_BAND_TARGETS):
                trained = train_selected_model(
                    model_type,
                    matrices_by_feature_set[feature_set]["train"],
                    targets_by_split["train"][:, [target_index]],
                    feature_names,
                    ridge_alpha=30.0,
                    elasticnet_alpha=0.05,
                    elasticnet_l1_ratio=0.35,
                    random_state=random_state,
                )
                for split in ("train", "val", "test"):
                    predictions = predict_selected_model(trained, matrices_by_feature_set[feature_set][split])
                    predictions_by_split[split][:, target_index] = np.asarray(predictions, dtype=np.float64).reshape(-1)
            evaluated = evaluate_band_predictions(feature_set, model_type, predictions_by_split, targets_by_split, split_samples)
            run_rows.append(evaluated["run_row"])
            per_target_rows.extend(evaluated["per_target_rows"])
            worst_rows.extend(evaluated["worst_rows"])

    best_run = min(run_rows, key=lambda row: (row["test_group_mae"], row["feature_set"], row["model_type"]))
    best_per_target = select_best_per_target(per_target_rows)
    return {
        "summary": {
            "best_run": best_run,
            "best_per_target": best_per_target,
            "benchmark_results": sorted(run_rows, key=lambda row: (row["test_group_mae"], row["feature_set"], row["model_type"])),
            "skipped_runs": skipped_runs,
        },
        "run_rows": run_rows,
        "per_target_rows": per_target_rows,
        "worst_rows": worst_rows,
    }


def build_feature_set_matrices(split_samples: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    v5_names = get_feature_names()
    v6_names = get_band_feature_names()
    matrices: dict[str, dict[str, Any]] = {
        "v5_existing": {"feature_names": v5_names},
        "v6_bands": {"feature_names": v6_names},
        "v5_plus_v6": {"feature_names": [*v5_names, *v6_names]},
    }
    for split, samples in split_samples.items():
        v5_matrix = extract_v5_matrix(samples, v5_names)
        v6_matrix = extract_v6_matrix(samples, v6_names)
        matrices["v5_existing"][split] = v5_matrix
        matrices["v6_bands"][split] = v6_matrix
        matrices["v5_plus_v6"][split] = np.hstack([v5_matrix, v6_matrix])
    return matrices


def extract_v5_matrix(samples: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        rows.append(feature_vector(features, feature_names))
    return np.asarray(rows, dtype=np.float64)


def extract_v6_matrix(samples: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        features = extract_front_side_band_features(sample["front_image_path"], sample["side_image_path"])
        rows.append([float(features[name]) for name in feature_names])
    return np.asarray(rows, dtype=np.float64)


def evaluate_band_predictions(
    feature_set: str,
    model_type: str,
    predictions_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    split_samples: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metrics = {}
    for split, targets in targets_by_split.items():
        errors = np.abs(predictions_by_split[split] - targets)
        metrics[split] = {
            "overall_mae": _mean([float(errors[:, index].mean()) for index in range(len(MEASUREMENT_BAND_TARGETS))]),
            "mae_by_target": {
                target: float(errors[:, index].mean())
                for index, target in enumerate(MEASUREMENT_BAND_TARGETS)
            },
        }
    run_name = f"{feature_set}__target_specific__{model_type}"
    run_row = {
        "run_name": run_name,
        "feature_set": feature_set,
        "model_type": model_type,
        "train_group_mae": metrics["train"]["overall_mae"],
        "val_group_mae": metrics["val"]["overall_mae"],
        "test_group_mae": metrics["test"]["overall_mae"],
        "promotion_gate": promotion_gate(metrics["test"]["overall_mae"])["gate"],
        "beats_phase_3w_best_group": metrics["test"]["overall_mae"] < 5.2379,
        "worst_target": max(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
        "best_target": min(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
    }
    per_target_rows = [
        {
            "run_name": run_name,
            "feature_set": feature_set,
            "model_type": model_type,
            "target": target,
            "test_mae": metrics["test"]["mae_by_target"][target],
            "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
            "beats_phase_3w_target": metrics["test"]["mae_by_target"][target] < PHASE_3W_TARGET_BASELINES[target],
        }
        for target in MEASUREMENT_BAND_TARGETS
    ]
    worst_rows = build_worst_prediction_rows(run_name, feature_set, model_type, predictions_by_split["test"], targets_by_split["test"], split_samples["test"])
    return {"run_row": run_row, "per_target_rows": per_target_rows, "worst_rows": worst_rows}


def build_worst_prediction_rows(
    run_name: str,
    feature_set: str,
    model_type: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for target_index, target in enumerate(MEASUREMENT_BAND_TARGETS):
        errors = predictions[:, target_index] - targets[:, target_index]
        absolute_errors = np.abs(errors)
        target_values = targets[:, target_index]
        worst_indices = list(np.argsort(absolute_errors)[::-1][:20])
        for rank, sample_index in enumerate(worst_indices, start=1):
            rows.append(
                {
                    "run_name": run_name,
                    "feature_set": feature_set,
                    "model_type": model_type,
                    "target": target,
                    "rank": rank,
                    "sample_id": samples[sample_index]["sample_id"],
                    "true_value": float(targets[sample_index, target_index]),
                    "predicted_value": float(predictions[sample_index, target_index]),
                    "signed_error": float(errors[sample_index]),
                    "abs_error": float(absolute_errors[sample_index]),
                    "body_shape": samples[sample_index].get("body_shape", ""),
                    "small_measurement_mae": bucket_mae(target_values, absolute_errors, "low"),
                    "mid_measurement_mae": bucket_mae(target_values, absolute_errors, "mid"),
                    "large_measurement_mae": bucket_mae(target_values, absolute_errors, "high"),
                }
            )
    return rows


def select_best_per_target(per_target_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best = {}
    for target in MEASUREMENT_BAND_TARGETS:
        rows = [row for row in per_target_rows if row["target"] == target]
        if rows:
            best[target] = min(rows, key=lambda row: (row["test_mae"], row["run_name"]))
    return best


def write_measurement_band_contact_sheets(samples: list[dict[str, Any]], output_dir: Path, per_bucket: int = 4) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    for target in MEASUREMENT_BAND_TARGETS:
        ordered = sorted(samples, key=lambda sample: sample["measurements"][target])
        mid_start = max(0, len(ordered) // 2 - per_bucket // 2)
        selected = [*ordered[:per_bucket], *ordered[mid_start : mid_start + per_bucket], *ordered[-per_bucket:]]
        warnings.extend(write_contact_sheet(output_dir / f"{target}_low_mid_high.png", selected, f"{target}: low / mid / high"))
    return warnings


def write_contact_sheet(path: Path, samples: list[dict[str, Any]], title: str) -> list[str]:
    warnings = []
    thumb_width = 88
    thumb_height = 124
    label_height = 24
    columns = min(6, len(samples))
    rows = int(np.ceil(len(samples) / columns))
    sheet = Image.new("RGB", (columns * thumb_width * 2, rows * (thumb_height + label_height) + 24), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(20, 20, 20))
    for index, sample in enumerate(samples):
        row = index // columns
        col = index % columns
        x = col * thumb_width * 2
        y = 24 + row * (thumb_height + label_height)
        try:
            front = thumbnail(sample["front_image_path"], thumb_width, thumb_height)
            side = thumbnail(sample["side_image_path"], thumb_width, thumb_height)
        except (OSError, ValueError, FileNotFoundError) as error:
            warnings.append(f"Could not add sample {sample.get('sample_id', '')} to contact sheet {path.name}: {error}")
            continue
        sheet.paste(front, (x, y))
        sheet.paste(side, (x + thumb_width, y))
        draw.text((x + 2, y + thumb_height + 2), sample["sample_id"], fill=(20, 20, 20))
    sheet.save(path)
    return warnings


def thumbnail(path: str | Path, width: int, height: int) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    with Image.open(image_path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((width, height))
        canvas = Image.new("RGB", (width, height), (230, 230, 230))
        canvas.paste(thumb, ((width - thumb.width) // 2, (height - thumb.height) // 2))
        return canvas


def optional_prediction_warnings(prediction_csv: str | Path | None) -> list[str]:
    if prediction_csv is None:
        return []
    path = Path(prediction_csv)
    if not path.exists():
        return [f"Optional prediction file is missing: {path}"]
    return []


def write_outputs(output_path: Path, summary: dict[str, Any], correlation_rows: list[dict[str, Any]], benchmark: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "band_correlations_json": output_path / BAND_CORRELATIONS_JSON,
        "band_correlations_csv": output_path / BAND_CORRELATIONS_CSV,
        "band_correlations_md": output_path / BAND_CORRELATIONS_MD,
        "band_benchmark_json": output_path / BAND_BENCHMARK_JSON,
        "band_benchmark_csv": output_path / BAND_BENCHMARK_CSV,
        "per_target_band_results_csv": output_path / PER_TARGET_BAND_RESULTS_CSV,
        "worst_predictions_csv": output_path / WORST_PREDICTIONS_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_json(paths["band_correlations_json"], summary["correlation_summary"])
    write_csv(paths["band_correlations_csv"], correlation_rows, ["target", "band_name", "center_y_ratio", "feature", "feature_role", "correlation", "abs_correlation"])
    paths["band_correlations_md"].write_text(format_correlation_report(summary), encoding="utf-8")
    write_json(paths["band_benchmark_json"], benchmark["summary"])
    write_csv(paths["band_benchmark_csv"], benchmark["run_rows"], ["run_name", "feature_set", "model_type", "train_group_mae", "val_group_mae", "test_group_mae", "promotion_gate", "beats_phase_3w_best_group", "worst_target", "best_target"])
    write_csv(paths["per_target_band_results_csv"], benchmark["per_target_rows"], ["run_name", "feature_set", "model_type", "target", "test_mae", "promotion_gate", "beats_phase_3w_target"])
    write_csv(paths["worst_predictions_csv"], benchmark["worst_rows"], ["run_name", "feature_set", "model_type", "target", "rank", "sample_id", "true_value", "predicted_value", "signed_error", "abs_error", "body_shape", "small_measurement_mae", "mid_measurement_mae", "large_measurement_mae"])
    paths["summary_md"].write_text(format_summary_report(summary), encoding="utf-8")
    return paths


def format_correlation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3X Band Correlations",
        "",
        f"Band feature version: `{summary['band_feature_version']}`",
        "",
        "| Target | Best Band | Center Y | Best Feature | Role | Abs Corr |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for target, target_summary in summary["correlation_summary"].items():
        best = target_summary["best_overall"]
        lines.append(
            f"| {target} | {best['band_name']} | {float(best['center_y_ratio']):.2f} | {best['feature']} | {best['feature_role']} | {float(best['abs_correlation']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def format_summary_report(summary: dict[str, Any]) -> str:
    best = summary["benchmark"]["best_run"]
    lines = [
        "# Phase 3X Measurement Band Diagnostics",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"V5 feature version: `{summary['v5_feature_version']}`",
        f"Band feature version: `{summary['band_feature_version']}`",
        f"Best run: `{best['run_name']}`",
        f"Best group MAE: {float(best['test_group_mae']):.4f}",
        f"Gate: `{best['promotion_gate']}`",
        "",
        "## Best Per Target",
        "",
        "| Target | Run | Feature Set | Model | MAE | Gate |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for target, row in summary["benchmark"]["best_per_target"].items():
        lines.append(
            f"| {target} | {row['run_name']} | {row['feature_set']} | {row['model_type']} | {float(row['test_mae']):.4f} | {row['promotion_gate']} |"
        )
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and benchmark localized chest/waist/hip/thigh measurement bands.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-csv", help="Optional prediction CSV; missing files produce warnings.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = audit_measurement_bands(args.dataset, args.output, prediction_csv=args.prediction_csv, random_state=args.seed)
    best = result["summary"]["benchmark"]["best_run"]
    print(f"Best run: {best['run_name']} test group MAE {best['test_group_mae']:.4f}")
    print(f"Band correlations: {result['band_correlations_json']}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
