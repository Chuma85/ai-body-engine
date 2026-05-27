from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.features.image_silhouette_features import (
    FEATURE_EXTRACTOR_VERSION,
    extract_front_side_features,
    get_feature_names,
)

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
RESULTS_FILENAME = "feature_drift.csv"


def analyze_feature_drift(
    datasets: list[str | Path],
    output_dir: str | Path,
    clean_dataset: str | Path | None = None,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one dataset is required.")

    dataset_paths = [Path(dataset) for dataset in datasets]
    clean_path = Path(clean_dataset) if clean_dataset is not None else dataset_paths[0]
    feature_names = get_feature_names()
    clean_features = extract_dataset_features(clean_path, feature_names, limit_samples=limit_samples)
    dataset_features = {
        ablation_name(path): extract_dataset_features(path, feature_names, limit_samples=limit_samples)
        for path in dataset_paths
    }
    clean_name = ablation_name(clean_path)
    if clean_name not in dataset_features:
        dataset_features[clean_name] = clean_features

    summary = build_feature_drift_summary(dataset_features, clean_name, feature_names)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    results_path = output_path / RESULTS_FILENAME

    _write_json(summary_path, summary)
    report_path.write_text(format_feature_drift_report(summary), encoding="utf-8")
    write_feature_drift_csv(results_path, summary["rows"])

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "results_path": str(results_path),
        "summary": summary,
    }


def extract_dataset_features(
    dataset_root: str | Path,
    feature_names: list[str] | None = None,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    dataset = SyntheticBodyDataset(dataset_root, split="all")
    names = feature_names or get_feature_names()
    sample_ids: list[str] = []
    rows: list[list[float]] = []
    for index, sample in enumerate(dataset):
        if limit_samples is not None and index >= limit_samples:
            break
        features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        sample_ids.append(sample["sample_id"])
        rows.append([float(features[name]) for name in names])
    if not rows:
        raise ValueError(f"No samples available for feature drift analysis: {dataset_root}")
    return {
        "dataset": str(dataset_root),
        "sample_ids": sample_ids,
        "feature_names": names,
        "matrix": np.asarray(rows, dtype=np.float64),
    }


def build_feature_drift_summary(
    dataset_features: dict[str, dict[str, Any]],
    clean_name: str,
    feature_names: list[str],
) -> dict[str, Any]:
    if clean_name not in dataset_features:
        raise ValueError(f"Clean dataset '{clean_name}' was not found.")
    clean = dataset_features[clean_name]
    rows: list[dict[str, Any]] = []
    top_by_ablation: dict[str, list[dict[str, Any]]] = {}

    for name, data in dataset_features.items():
        matched_clean, matched_current = matched_feature_matrices(clean, data)
        drift_rows = feature_drift_rows(name, matched_clean, matched_current, feature_names)
        rows.extend(drift_rows)
        top_by_ablation[name] = sorted(drift_rows, key=lambda row: row["mean_abs_drift"], reverse=True)[:10]

    return {
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "clean_dataset": clean_name,
        "sample_count": len(clean["sample_ids"]),
        "feature_count": len(feature_names),
        "ablation_names": list(dataset_features),
        "rows": rows,
        "top_drift_by_ablation": top_by_ablation,
        "recommendations": recommendations(top_by_ablation, clean_name),
    }


def matched_feature_matrices(clean: dict[str, Any], current: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    clean_index = {sample_id: index for index, sample_id in enumerate(clean["sample_ids"])}
    current_index = {sample_id: index for index, sample_id in enumerate(current["sample_ids"])}
    shared_ids = [sample_id for sample_id in clean["sample_ids"] if sample_id in current_index]
    if not shared_ids:
        raise ValueError("No matching sample IDs found between clean and ablation datasets.")
    clean_rows = [clean["matrix"][clean_index[sample_id]] for sample_id in shared_ids]
    current_rows = [current["matrix"][current_index[sample_id]] for sample_id in shared_ids]
    return np.asarray(clean_rows, dtype=np.float64), np.asarray(current_rows, dtype=np.float64)


def feature_drift_rows(
    ablation: str,
    clean_matrix: np.ndarray,
    current_matrix: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    deltas = current_matrix - clean_matrix
    for feature_index, feature_name in enumerate(feature_names):
        clean_values = clean_matrix[:, feature_index]
        current_values = current_matrix[:, feature_index]
        delta_values = deltas[:, feature_index]
        rows.append(
            {
                "ablation": ablation,
                "feature": feature_name,
                "clean_mean": float(clean_values.mean()),
                "clean_std": float(clean_values.std()),
                "clean_min": float(clean_values.min()),
                "clean_max": float(clean_values.max()),
                "ablation_mean": float(current_values.mean()),
                "ablation_std": float(current_values.std()),
                "ablation_min": float(current_values.min()),
                "ablation_max": float(current_values.max()),
                "mean_signed_drift": float(delta_values.mean()),
                "mean_abs_drift": float(np.abs(delta_values).mean()),
                "max_abs_drift": float(np.abs(delta_values).max()),
            }
        )
    return rows


def format_feature_drift_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Feature Drift Analysis",
        "",
        f"Feature extractor version: `{summary['feature_extractor_version']}`",
        f"Clean dataset: `{summary['clean_dataset']}`",
        f"Samples: {summary['sample_count']}",
        f"Features: {summary['feature_count']}",
        "",
    ]
    for ablation, rows in summary["top_drift_by_ablation"].items():
        if ablation == summary["clean_dataset"]:
            continue
        lines.extend(
            [
                f"## {ablation}",
                "",
                _markdown_table(
                    ["Feature", "Mean Abs Drift", "Mean Signed Drift", "Clean Mean", "Ablation Mean"],
                    [
                        [
                            row["feature"],
                            _format_float(row["mean_abs_drift"]),
                            _format_float(row["mean_signed_drift"]),
                            _format_float(row["clean_mean"]),
                            _format_float(row["ablation_mean"]),
                        ]
                        for row in rows[:10]
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Recommendations", "", *[f"- {note}" for note in summary["recommendations"]], ""])
    return "\n".join(lines)


def write_feature_drift_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ablation",
        "feature",
        "clean_mean",
        "clean_std",
        "clean_min",
        "clean_max",
        "ablation_mean",
        "ablation_std",
        "ablation_min",
        "ablation_max",
        "mean_signed_drift",
        "mean_abs_drift",
        "max_abs_drift",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ablation_name(dataset_path: str | Path) -> str:
    name = Path(dataset_path).name
    if name.startswith("phase_3n_"):
        name = name[len("phase_3n_") :]
    return name


def recommendations(top_by_ablation: dict[str, list[dict[str, Any]]], clean_name: str) -> list[str]:
    notes = []
    for ablation, rows in top_by_ablation.items():
        if ablation == clean_name or not rows:
            continue
        top_feature = rows[0]
        notes.append(
            f"{ablation}: largest drift is {top_feature['feature']} ({_format_float(top_feature['mean_abs_drift'])})."
        )
    if not notes:
        notes.append("No ablation feature drift was detected.")
    return notes


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
    return f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze feature drift across same-body render ablation datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset roots to compare.")
    parser.add_argument("--clean-dataset", required=True, help="Clean baseline dataset root.")
    parser.add_argument("--output", required=True, help="Output directory for feature drift reports.")
    parser.add_argument("--limit-samples", type=int)
    args = parser.parse_args(argv)

    result = analyze_feature_drift(
        args.datasets,
        args.output,
        clean_dataset=args.clean_dataset,
        limit_samples=args.limit_samples,
    )
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Results: {result['results_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
