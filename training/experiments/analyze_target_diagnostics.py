from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
WORST_SAMPLES_FILENAME = "worst_samples_test.csv"
SPLITS = ("train", "val", "test")


def analyze_target_diagnostics(experiment_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_dir)
    metrics_path = experiment_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")

    metrics = _read_json(metrics_path)
    target_columns = list(metrics.get("target_columns", []))
    if not target_columns:
        raise ValueError(f"Experiment metrics has no target_columns: {metrics_path}")

    prediction_rows = {
        split: read_prediction_rows(experiment_path / f"predictions_{split}.csv")
        for split in SPLITS
    }
    summary = build_diagnostics_summary(experiment_path, target_columns, prediction_rows)
    worst_rows = worst_sample_rows(prediction_rows["test"], target_columns, limit_per_target=10)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    worst_samples_path = output_path / WORST_SAMPLES_FILENAME
    _write_json(summary_path, summary)
    report_path.write_text(format_diagnostics_report(summary), encoding="utf-8")
    write_worst_samples_csv(worst_samples_path, worst_rows)

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "worst_samples_path": str(worst_samples_path),
        "summary": summary,
    }


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def build_diagnostics_summary(
    experiment_path: Path,
    target_columns: list[str],
    prediction_rows: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    per_target = {
        target: {
            split: calculate_target_diagnostics(prediction_rows[split], target)
            for split in SPLITS
        }
        for target in target_columns
    }
    ranked_by_test_percent = sorted(
        target_columns,
        key=lambda target: per_target[target]["test"]["mae_percent_of_mean_true"],
    )
    ranked_by_test_mae = sorted(
        target_columns,
        key=lambda target: per_target[target]["test"]["mae"],
    )

    return {
        "experiment": str(experiment_path),
        "target_columns": target_columns,
        "sample_counts": {split: len(rows) for split, rows in prediction_rows.items()},
        "per_target": per_target,
        "target_rank_easiest_to_hardest_percent": ranked_by_test_percent,
        "target_rank_easiest_to_hardest_mae": ranked_by_test_mae,
        "hardest_targets": list(reversed(ranked_by_test_percent))[:5],
        "easiest_targets": ranked_by_test_percent[:5],
    }


def calculate_target_diagnostics(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Cannot calculate diagnostics for {target} with zero rows.")

    true_values = [float(row[f"true_{target}"]) for row in rows]
    pred_values = [float(row[f"pred_{target}"]) for row in rows]
    signed_errors = [pred - true for true, pred in zip(true_values, pred_values)]
    absolute_errors = [abs(error) for error in signed_errors]
    mean_true = _mean(true_values)
    mae = _mean(absolute_errors)
    return {
        "count": len(rows),
        "mae": mae,
        "mean_true": mean_true,
        "mae_percent_of_mean_true": percent_mae(mae, mean_true),
        "signed_error_mean": signed_bias(signed_errors),
        "underprediction_count": sum(1 for error in signed_errors if error < 0),
        "overprediction_count": sum(1 for error in signed_errors if error > 0),
        "exact_prediction_count": sum(1 for error in signed_errors if error == 0),
        "correlation": pearson_correlation(true_values, pred_values),
    }


def percent_mae(mae: float, mean_true: float) -> float:
    if mean_true == 0:
        return 0.0
    return (mae / abs(mean_true)) * 100


def signed_bias(signed_errors: list[float]) -> float:
    if not signed_errors:
        raise ValueError("Cannot calculate signed bias for an empty error list.")
    return _mean(signed_errors)


def worst_sample_rows(
    rows: list[dict[str, str]],
    target_columns: list[str],
    limit_per_target: int = 10,
) -> list[dict[str, Any]]:
    worst_rows: list[dict[str, Any]] = []
    for target in target_columns:
        ranked = sorted(rows, key=lambda row: float(row[f"abs_error_{target}"]), reverse=True)
        for row in ranked[:limit_per_target]:
            true_value = float(row[f"true_{target}"])
            pred_value = float(row[f"pred_{target}"])
            worst_rows.append(
                {
                    "target": target,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "true_value": true_value,
                    "pred_value": pred_value,
                    "signed_error": pred_value - true_value,
                    "abs_error": float(row[f"abs_error_{target}"]),
                }
            )
    return worst_rows


def write_worst_samples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["target", "sample_id", "split", "true_value", "pred_value", "signed_error", "abs_error"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_diagnostics_report(summary: dict[str, Any]) -> str:
    test_rows = [
        [
            target,
            _format_float(summary["per_target"][target]["test"]["mae"]),
            _format_float(summary["per_target"][target]["test"]["mean_true"]),
            _format_float(summary["per_target"][target]["test"]["mae_percent_of_mean_true"]),
            _format_float(summary["per_target"][target]["test"]["signed_error_mean"]),
            str(summary["per_target"][target]["test"]["underprediction_count"]),
            str(summary["per_target"][target]["test"]["overprediction_count"]),
            _format_float(summary["per_target"][target]["test"]["correlation"]),
        ]
        for target in summary["target_columns"]
    ]
    lines = [
        "# Target Diagnostics",
        "",
        f"Experiment: `{summary['experiment']}`",
        (
            "Samples: "
            f"train={summary['sample_counts']['train']} "
            f"val={summary['sample_counts']['val']} "
            f"test={summary['sample_counts']['test']}"
        ),
        "",
        "## Test Diagnostics",
        "",
        _markdown_table(
            ["Target", "MAE", "Mean True", "MAE %", "Bias", "Under", "Over", "Corr"],
            test_rows,
        ),
        "",
        "## Easiest Targets",
        "",
        ", ".join(summary["easiest_targets"]),
        "",
        "## Hardest Targets",
        "",
        ", ".join(summary["hardest_targets"]),
        "",
    ]
    return "\n".join(lines)


def pearson_correlation(first_values: list[float], second_values: list[float]) -> float:
    if len(first_values) != len(second_values) or len(first_values) < 2:
        return 0.0

    first_mean = _mean(first_values)
    second_mean = _mean(second_values)
    numerator = sum((first - first_mean) * (second - second_mean) for first, second in zip(first_values, second_values))
    first_denominator = math.sqrt(sum((first - first_mean) ** 2 for first in first_values))
    second_denominator = math.sqrt(sum((second - second_mean) ** 2 for second in second_values))
    denominator = first_denominator * second_denominator
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate mean of empty values.")
    return sum(values) / len(values)


def _read_json(path: Path) -> dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze per-target diagnostics from experiment prediction CSVs.")
    parser.add_argument("--experiment", required=True, help="Experiment directory containing metrics.json and prediction CSVs.")
    parser.add_argument("--output", required=True, help="Output directory for diagnostics reports.")
    args = parser.parse_args(argv)

    result = analyze_target_diagnostics(args.experiment, args.output)
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Worst samples: {result['worst_samples_path']}")
    print("Hardest targets: " + ", ".join(result["summary"]["hardest_targets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
