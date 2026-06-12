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
from training.experiments.analyze_feature_drift import feature_drift_group
from training.features.image_silhouette_features import (
    FEATURE_EXTRACTOR_VERSION,
    extract_front_side_features,
    feature_vector,
    get_feature_names,
)
from training.train_baseline_measurements import TARGET_COLUMNS

SIGNAL_JSON = "signal_correlations.json"
SIGNAL_CSV = "signal_correlations.csv"
SIGNAL_MD = "signal_correlations.md"
AMBIGUOUS_PAIRS_CSV = "ambiguous_pairs.csv"
ERROR_ANALYSIS_CSV = "per_target_error_analysis.csv"
VISUAL_SUMMARY_MD = "visual_audit_summary.md"
CONTACT_SHEET_DIR = "contact_sheets"

DEFAULT_CONTACT_TARGETS = ["waist_cm", "chest_cm", "hip_cm", "inseam_cm"]
WEAK_SIGNAL_THRESHOLD = 0.20
MISMATCH_THRESHOLD = 0.12


def audit_measurement_signal(
    dataset: str | Path,
    output_dir: str | Path,
    prediction_csvs: list[str | Path] | None = None,
    limit_samples: int | None = None,
    top_features: int = 10,
    ambiguous_pairs_per_target: int = 10,
    contact_sheet_count: int = 6,
) -> dict[str, Any]:
    if top_features <= 0:
        raise ValueError("top_features must be positive.")
    if ambiguous_pairs_per_target <= 0:
        raise ValueError("ambiguous_pairs_per_target must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples = load_samples(dataset, limit_samples=limit_samples)
    feature_names = get_feature_names()
    sample_ids, feature_matrix, target_matrix = extract_feature_and_target_matrices(samples, feature_names)

    correlation_rows = build_correlation_rows(feature_matrix, target_matrix, feature_names, TARGET_COLUMNS)
    signal_summary = build_signal_summary(correlation_rows, len(samples), len(feature_names))
    ambiguous_rows = find_ambiguous_pairs(
        sample_ids,
        feature_matrix,
        target_matrix,
        TARGET_COLUMNS,
        ambiguous_pairs_per_target=ambiguous_pairs_per_target,
    )

    prediction_warnings: list[str] = []
    error_rows: list[dict[str, Any]] = []
    best_worst_prediction_samples: dict[str, list[str]] = {}
    for prediction_csv in prediction_csvs or []:
        rows, warnings, sample_groups = analyze_prediction_errors(prediction_csv)
        error_rows.extend(rows)
        prediction_warnings.extend(warnings)
        for sheet_name, ids in sample_groups.items():
            best_worst_prediction_samples[sheet_name] = ids

    contact_sheet_warnings = write_contact_sheets(
        samples,
        output_path / CONTACT_SHEET_DIR,
        contact_targets=DEFAULT_CONTACT_TARGETS,
        contact_sheet_count=contact_sheet_count,
        extra_sample_groups=best_worst_prediction_samples,
    )

    signal_payload = {
        "dataset": str(dataset),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "sample_count": len(samples),
        "feature_count": len(feature_names),
        "target_columns": TARGET_COLUMNS,
        "weak_signal_threshold": WEAK_SIGNAL_THRESHOLD,
        "likely_label_image_mismatch_threshold": MISMATCH_THRESHOLD,
        "target_summaries": signal_summary,
        "warnings": [*prediction_warnings, *contact_sheet_warnings],
    }

    paths = write_audit_outputs(output_path, signal_payload, correlation_rows, ambiguous_rows, error_rows)
    return {
        **{key: str(value) for key, value in paths.items()},
        "signal": signal_payload,
        "correlation_rows": correlation_rows,
        "ambiguous_pairs": ambiguous_rows,
        "error_rows": error_rows,
    }


def load_samples(dataset: str | Path, limit_samples: int | None = None) -> list[dict[str, Any]]:
    loaded = list(SyntheticBodyDataset(dataset, split="all"))
    if limit_samples is not None:
        loaded = loaded[:limit_samples]
    if not loaded:
        raise ValueError(f"No samples available for measurement signal audit: {dataset}")
    return loaded


def extract_feature_and_target_matrices(
    samples: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids: list[str] = []
    feature_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    for sample in samples:
        features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        sample_ids.append(sample["sample_id"])
        feature_rows.append(feature_vector(features, feature_names))
        target_rows.append([float(sample["measurements"][target]) for target in TARGET_COLUMNS])
    return sample_ids, np.asarray(feature_rows, dtype=np.float64), np.asarray(target_rows, dtype=np.float64)


def build_correlation_rows(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    feature_names: list[str],
    target_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(target_columns):
        target_values = target_matrix[:, target_index]
        for feature_index, feature_name in enumerate(feature_names):
            corr = pearson_correlation(feature_matrix[:, feature_index], target_values)
            rows.append(
                {
                    "target": target,
                    "feature": feature_name,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "feature_group": feature_drift_group(feature_name),
                    "view_group": feature_view_group(feature_name),
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


def feature_view_group(feature_name: str) -> str:
    if feature_name.startswith("front_to_side_") or feature_name.startswith("front_side_"):
        return "front_side_ratio_or_cross_view"
    if feature_name.startswith("front_"):
        return "front_only"
    if feature_name.startswith("side_"):
        return "side_only"
    return "other"


def build_signal_summary(correlation_rows: list[dict[str, Any]], sample_count: int, feature_count: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        rows = [row for row in correlation_rows if row["target"] == target]
        strongest = sorted(rows, key=lambda row: row["abs_correlation"], reverse=True)
        positive = sorted(rows, key=lambda row: row["correlation"], reverse=True)
        negative = sorted(rows, key=lambda row: row["correlation"])
        group_max = {}
        for group in sorted({row["feature_group"] for row in rows}):
            group_rows = [row for row in rows if row["feature_group"] == group]
            group_max[group] = max(group_rows, key=lambda row: row["abs_correlation"])
        view_max = {}
        for group in sorted({row["view_group"] for row in rows}):
            group_rows = [row for row in rows if row["view_group"] == group]
            view_max[group] = max(group_rows, key=lambda row: row["abs_correlation"])
        max_abs = float(strongest[0]["abs_correlation"]) if strongest else 0.0
        likely_mismatch = max_abs < MISMATCH_THRESHOLD
        summaries[target] = {
            "sample_count": sample_count,
            "feature_count": feature_count,
            "max_abs_correlation": max_abs,
            "weak_visual_signal": max_abs < WEAK_SIGNAL_THRESHOLD,
            "likely_label_image_mismatch": likely_mismatch,
            "top_features": strongest[:10],
            "strongest_positive": positive[:5],
            "strongest_negative": negative[:5],
            "best_by_feature_group": group_max,
            "best_by_view_group": view_max,
            "front_view_best_abs_correlation": float(view_max.get("front_only", {}).get("abs_correlation", 0.0)),
            "side_view_best_abs_correlation": float(view_max.get("side_only", {}).get("abs_correlation", 0.0)),
        }
    return summaries


def find_ambiguous_pairs(
    sample_ids: list[str],
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    target_columns: list[str],
    ambiguous_pairs_per_target: int,
) -> list[dict[str, Any]]:
    standardized_features = standardize_matrix(feature_matrix)
    standardized_targets = standardize_matrix(target_matrix)
    feature_distances = pairwise_euclidean(standardized_features)
    np.fill_diagonal(feature_distances, np.inf)
    nearest_indices = np.argsort(feature_distances, axis=1)[:, : min(10, len(sample_ids) - 1)]

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target_index, target in enumerate(target_columns):
        candidates: list[dict[str, Any]] = []
        for left_index, neighbor_indices in enumerate(nearest_indices):
            for right_index in neighbor_indices:
                left_id = sample_ids[left_index]
                right_id = sample_ids[int(right_index)]
                pair_key = tuple(sorted((left_id, right_id)))
                key = (target, pair_key[0], pair_key[1])
                if key in seen:
                    continue
                seen.add(key)
                label_diff = abs(float(target_matrix[left_index, target_index] - target_matrix[int(right_index), target_index]))
                standardized_label_diff = abs(
                    float(standardized_targets[left_index, target_index] - standardized_targets[int(right_index), target_index])
                )
                feature_distance = float(feature_distances[left_index, int(right_index)])
                ambiguity_score = standardized_label_diff / max(feature_distance, 1e-9)
                candidates.append(
                    {
                        "target": target,
                        "sample_id_a": left_id,
                        "sample_id_b": right_id,
                        "feature_distance": feature_distance,
                        "label_diff": label_diff,
                        "standardized_label_diff": standardized_label_diff,
                        "ambiguity_score": ambiguity_score,
                    }
                )
        candidates.sort(key=lambda row: (row["ambiguity_score"], row["label_diff"]), reverse=True)
        rows.extend(candidates[:ambiguous_pairs_per_target])
    return rows


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    return (matrix - matrix.mean(axis=0)) / stds


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    rows = matrix.tolist()
    distances: list[list[float]] = []
    for left in rows:
        distance_row = []
        for right in rows:
            squared = sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
            distance_row.append(squared ** 0.5)
        distances.append(distance_row)
    return np.asarray(distances, dtype=np.float64)


def analyze_prediction_errors(prediction_csv: str | Path) -> tuple[list[dict[str, Any]], list[str], dict[str, list[str]]]:
    path = Path(prediction_csv)
    if not path.exists():
        return [], [f"Optional prediction file is missing: {path}"], {}

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return [], [f"Optional prediction file has no rows: {path}"], {}

    analysis_rows: list[dict[str, Any]] = []
    sample_groups: dict[str, list[str]] = {}
    source_name = path.parent.name
    for target in TARGET_COLUMNS:
        error_column = f"abs_error_{target}"
        true_column = f"true_{target}"
        if error_column not in rows[0] or true_column not in rows[0]:
            continue
        target_rows = [row for row in rows if row.get(error_column, "") not in ("", None)]
        if not target_rows:
            continue
        errors = np.asarray([float(row[error_column]) for row in target_rows], dtype=np.float64)
        truths = np.asarray([float(row[true_column]) for row in target_rows], dtype=np.float64)
        worst_rows = sorted(target_rows, key=lambda row: float(row[error_column]), reverse=True)[:20]
        best_rows = sorted(target_rows, key=lambda row: float(row[error_column]))[:20]
        sample_groups[f"{source_name}_{target}_worst_predictions"] = [row["sample_id"] for row in worst_rows[:8]]
        sample_groups[f"{source_name}_{target}_best_predictions"] = [row["sample_id"] for row in best_rows[:8]]
        analysis_rows.append(
            {
                "source": str(path),
                "target": target,
                "sample_count": len(target_rows),
                "mae": float(errors.mean()),
                "median_abs_error": float(np.median(errors)),
                "max_abs_error": float(errors.max()),
                "worst_sample_ids": ";".join(row["sample_id"] for row in worst_rows),
                "best_sample_ids": ";".join(row["sample_id"] for row in best_rows),
                "true_min": float(truths.min()),
                "true_max": float(truths.max()),
            }
        )
    return analysis_rows, [], sample_groups


def write_contact_sheets(
    samples: list[dict[str, Any]],
    output_dir: Path,
    contact_targets: list[str],
    contact_sheet_count: int,
    extra_sample_groups: dict[str, list[str]] | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    warnings: list[str] = []
    for target in contact_targets:
        if target not in samples[0]["measurements"]:
            continue
        sorted_samples = sorted(samples, key=lambda sample: sample["measurements"][target])
        low = sorted_samples[:contact_sheet_count]
        high = sorted_samples[-contact_sheet_count:]
        warnings.extend(write_sample_contact_sheet(output_dir / f"{target}_lowest_highest.png", [*low, *high], f"{target}: low to high"))

    for group_name, sample_ids in (extra_sample_groups or {}).items():
        selected = [samples_by_id[sample_id] for sample_id in sample_ids if sample_id in samples_by_id]
        if selected:
            warnings.extend(write_sample_contact_sheet(output_dir / f"{safe_filename(group_name)}.png", selected, group_name))
    return warnings


def write_sample_contact_sheet(path: Path, samples: list[dict[str, Any]], title: str) -> list[str]:
    warnings: list[str] = []
    if not samples:
        return warnings
    thumb_width = 96
    thumb_height = 134
    label_height = 26
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
        except (OSError, ValueError) as error:
            warnings.append(f"Could not add sample {sample['sample_id']} to contact sheet {path.name}: {error}")
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
        x = (width - thumb.width) // 2
        y = (height - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        return canvas


def write_audit_outputs(
    output_path: Path,
    signal_payload: dict[str, Any],
    correlation_rows: list[dict[str, Any]],
    ambiguous_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "signal_json": output_path / SIGNAL_JSON,
        "signal_csv": output_path / SIGNAL_CSV,
        "signal_md": output_path / SIGNAL_MD,
        "ambiguous_pairs_csv": output_path / AMBIGUOUS_PAIRS_CSV,
        "per_target_error_analysis_csv": output_path / ERROR_ANALYSIS_CSV,
        "visual_audit_summary_md": output_path / VISUAL_SUMMARY_MD,
    }
    write_json(paths["signal_json"], signal_payload)
    write_csv(paths["signal_csv"], correlation_rows, ["target", "feature", "correlation", "abs_correlation", "feature_group", "view_group"])
    write_csv(paths["ambiguous_pairs_csv"], ambiguous_rows, ["target", "sample_id_a", "sample_id_b", "feature_distance", "label_diff", "standardized_label_diff", "ambiguity_score"])
    write_csv(paths["per_target_error_analysis_csv"], error_rows, ["source", "target", "sample_count", "mae", "median_abs_error", "max_abs_error", "worst_sample_ids", "best_sample_ids", "true_min", "true_max"])
    paths["signal_md"].write_text(format_signal_report(signal_payload), encoding="utf-8")
    paths["visual_audit_summary_md"].write_text(format_visual_summary(signal_payload), encoding="utf-8")
    return paths


def format_signal_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Measurement Signal Correlations",
        "",
        f"Dataset: `{payload['dataset']}`",
        f"Feature extractor: `{payload['feature_extractor_version']}`",
        f"Samples: {payload['sample_count']}",
        f"Features: {payload['feature_count']}",
        "",
        "| Target | Max Abs Corr | Weak Signal | Likely Mismatch | Best Feature | Best Group | Best View |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for target, summary in payload["target_summaries"].items():
        best = summary["top_features"][0]
        view_best = max(summary["best_by_view_group"].values(), key=lambda row: row["abs_correlation"])
        lines.append(
            f"| {target} | {summary['max_abs_correlation']:.4f} | {summary['weak_visual_signal']} | "
            f"{summary['likely_label_image_mismatch']} | {best['feature']} | {best['feature_group']} | {view_best['view_group']} |"
        )
    weak = [target for target, summary in payload["target_summaries"].items() if summary["weak_visual_signal"]]
    if weak:
        lines.extend(["", "Weak visual/feature signal targets:", ""])
        lines.extend(f"- {target}" for target in weak)
    if payload.get("warnings"):
        lines.extend(["", "Warnings:", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    return "\n".join(lines) + "\n"


def format_visual_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Visual Audit Summary",
        "",
        "Contact sheets are written under `contact_sheets/` for low/high measurement samples and optional prediction examples.",
        "",
        "Targets audited by low/high contact sheet:",
    ]
    lines.extend(f"- {target}" for target in DEFAULT_CONTACT_TARGETS)
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- If low/high sheets do not show visible geometry differences for a target, label-image alignment may be weak.",
            "- If ambiguous pairs have nearly identical silhouettes but large label differences, the target may require metadata or better rendering.",
        ]
    )
    if payload.get("warnings"):
        lines.extend(["", "Warnings:", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
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


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)[:120]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether synthetic measurement labels align with visible geometry.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root.")
    parser.add_argument("--output", required=True, help="Audit output directory.")
    parser.add_argument("--prediction-csv", action="append", default=[], help="Optional predictions CSV for error analysis.")
    parser.add_argument("--limit-samples", type=int, help="Optional sample limit for quick audits.")
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--ambiguous-pairs-per-target", type=int, default=10)
    parser.add_argument("--contact-sheet-count", type=int, default=6)
    args = parser.parse_args(argv)

    result = audit_measurement_signal(
        args.dataset,
        args.output,
        prediction_csvs=args.prediction_csv,
        limit_samples=args.limit_samples,
        top_features=args.top_features,
        ambiguous_pairs_per_target=args.ambiguous_pairs_per_target,
        contact_sheet_count=args.contact_sheet_count,
    )
    print(f"Signal correlations: {result['signal_json']}")
    print(f"Ambiguous pairs: {result['ambiguous_pairs_csv']}")
    print(f"Visual summary: {result['visual_audit_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
