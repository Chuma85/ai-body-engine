from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from synthetic.validate_synthetic_dataset import _read_label_rows

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
MEASUREMENT_COLUMNS = [
    "height_cm",
    "weight_kg",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
    "sleeve_cm",
    "neck_cm",
    "thigh_cm",
    "calf_cm",
]
PLAUSIBLE_BOUNDS = {
    "height_cm": (140.0, 220.0),
    "weight_kg": (35.0, 180.0),
    "chest_cm": (60.0, 160.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (60.0, 170.0),
    "shoulder_cm": (25.0, 80.0),
    "inseam_cm": (50.0, 110.0),
    "sleeve_cm": (35.0, 90.0),
    "neck_cm": (25.0, 65.0),
    "thigh_cm": (30.0, 100.0),
    "calf_cm": (20.0, 75.0),
}
LOW_VARIATION_MIN_RANGES = {
    "height_cm": 20.0,
    "weight_kg": 25.0,
    "chest_cm": 20.0,
    "waist_cm": 20.0,
    "hip_cm": 20.0,
    "shoulder_cm": 8.0,
    "inseam_cm": 10.0,
    "sleeve_cm": 8.0,
    "neck_cm": 5.0,
    "thigh_cm": 10.0,
    "calf_cm": 8.0,
}
HIGH_CORRELATION_THRESHOLD = 0.95
DEFAULT_BUCKET_COUNT = 5


def audit_synthetic_variation(
    dataset: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset)
    labels_path = dataset_root / "labels" / "labels.csv"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    parse_result: dict[str, Any] = {"errors": [], "warnings": []}
    label_rows = _read_label_rows(labels_path, parse_result)
    if parse_result["errors"]:
        raise ValueError(f"Could not parse labels.csv: {'; '.join(parse_result['errors'])}")

    summary = build_variation_summary(dataset_root, label_rows, parse_result["warnings"])
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / SUMMARY_FILENAME
        report_path = output_path / REPORT_FILENAME
        _write_json(summary_path, summary)
        report_path.write_text(format_variation_report(summary), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        summary["report_path"] = str(report_path)

    return summary


def build_variation_summary(
    dataset_root: Path,
    label_rows: list[dict[str, str]],
    parse_warnings: list[str] | None = None,
) -> dict[str, Any]:
    numeric_values, missing_fields, non_numeric_fields = collect_numeric_values(label_rows, MEASUREMENT_COLUMNS)
    numeric_columns = [
        column
        for column in MEASUREMENT_COLUMNS
        if numeric_values.get(column)
    ]
    measurement_stats = {
        column: numeric_summary(numeric_values[column])
        for column in numeric_columns
    }
    buckets = {
        column: bucket_values(numeric_values[column], DEFAULT_BUCKET_COUNT)
        for column in numeric_columns
    }
    warnings = [*(parse_warnings or [])]
    warnings.extend(low_variation_warnings(measurement_stats))
    warnings.extend(outlier_warnings(numeric_values))
    warnings.extend(correlation_warnings(numeric_values))

    return {
        "dataset": str(dataset_root),
        "sample_count": len(label_rows),
        "numeric_columns": numeric_columns,
        "measurement_columns": MEASUREMENT_COLUMNS,
        "measurement_stats": measurement_stats,
        "buckets": buckets,
        "missing_fields": missing_fields,
        "non_numeric_fields": non_numeric_fields,
        "warnings": warnings,
    }


def collect_numeric_values(
    label_rows: list[dict[str, str]],
    columns: list[str],
) -> tuple[dict[str, list[float]], dict[str, int], list[dict[str, Any]]]:
    numeric_values = {column: [] for column in columns}
    missing_fields = {column: 0 for column in columns}
    non_numeric_fields: list[dict[str, Any]] = []

    for row_index, row in enumerate(label_rows):
        sample_id = row.get("sample_id", "")
        for column in columns:
            value = row.get(column, "")
            if value in ("", None):
                missing_fields[column] += 1
                continue
            try:
                numeric_values[column].append(float(value))
            except ValueError:
                non_numeric_fields.append(
                    {
                        "row_index": row_index,
                        "sample_id": sample_id,
                        "column": column,
                        "value": value,
                    }
                )

    return numeric_values, missing_fields, non_numeric_fields


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty numeric value list.")

    value_count = len(values)
    value_min = min(values)
    value_max = max(values)
    value_mean = sum(values) / value_count
    variance = sum((value - value_mean) ** 2 for value in values) / value_count
    return {
        "count": value_count,
        "min": value_min,
        "max": value_max,
        "mean": value_mean,
        "std": math.sqrt(variance),
        "range": value_max - value_min,
    }


def bucket_values(values: list[float], bucket_count: int) -> list[dict[str, float | int]]:
    if not values:
        return []

    value_min = min(values)
    value_max = max(values)
    if value_min == value_max:
        return [{"min": value_min, "max": value_max, "count": len(values)}]

    width = (value_max - value_min) / bucket_count
    buckets = [
        {"min": value_min + index * width, "max": value_min + (index + 1) * width, "count": 0}
        for index in range(bucket_count)
    ]
    buckets[-1]["max"] = value_max
    for value in values:
        bucket_index = min(int((value - value_min) / width), bucket_count - 1)
        buckets[bucket_index]["count"] = int(buckets[bucket_index]["count"]) + 1
    return buckets


def low_variation_warnings(measurement_stats: dict[str, dict[str, float | int]]) -> list[str]:
    warnings = []
    for column, minimum_range in LOW_VARIATION_MIN_RANGES.items():
        stats = measurement_stats.get(column)
        if not stats:
            continue
        observed_range = float(stats["range"])
        if observed_range < minimum_range:
            warnings.append(
                f"Low variation: {column} range {observed_range:.2f} is below recommended minimum {minimum_range:.2f}."
            )
    return warnings


def outlier_warnings(numeric_values: dict[str, list[float]]) -> list[str]:
    warnings = []
    for column, values in numeric_values.items():
        lower, upper = PLAUSIBLE_BOUNDS[column]
        outliers = [value for value in values if value < lower or value > upper]
        if outliers:
            warnings.append(
                f"Outlier values: {column} has {len(outliers)} values outside plausible bounds [{lower:.1f}, {upper:.1f}]."
            )
    return warnings


def correlation_warnings(numeric_values: dict[str, list[float]]) -> list[str]:
    warnings = []
    columns = [column for column, values in numeric_values.items() if len(values) >= 3]
    for first_index, first_column in enumerate(columns):
        for second_column in columns[first_index + 1 :]:
            first_values = numeric_values[first_column]
            second_values = numeric_values[second_column]
            if len(first_values) != len(second_values):
                continue
            correlation = pearson_correlation(first_values, second_values)
            if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:
                warnings.append(
                    f"High coupling: {first_column} and {second_column} correlation is {correlation:.3f}."
                )
    return warnings


def pearson_correlation(first_values: list[float], second_values: list[float]) -> float:
    if len(first_values) != len(second_values) or len(first_values) < 2:
        return 0.0

    first_mean = sum(first_values) / len(first_values)
    second_mean = sum(second_values) / len(second_values)
    numerator = sum((first - first_mean) * (second - second_mean) for first, second in zip(first_values, second_values))
    first_denominator = math.sqrt(sum((first - first_mean) ** 2 for first in first_values))
    second_denominator = math.sqrt(sum((second - second_mean) ** 2 for second in second_values))
    denominator = first_denominator * second_denominator
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_variation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Variation Audit",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Samples: {summary['sample_count']}",
        "",
        "## Measurement Ranges",
        "",
        _markdown_table(
            ["Measurement", "Min", "Max", "Mean", "Std", "Range"],
            [
                [
                    column,
                    _format_float(stats["min"]),
                    _format_float(stats["max"]),
                    _format_float(stats["mean"]),
                    _format_float(stats["std"]),
                    _format_float(stats["range"]),
                ]
                for column, stats in summary["measurement_stats"].items()
            ],
        ),
        "",
        "## Warnings",
        "",
    ]

    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("No variation, coupling, or outlier warnings were emitted.")

    lines.extend(
        [
            "",
            "## Missing And Non-Numeric Fields",
            "",
            _markdown_table(
                ["Measurement", "Missing Count"],
                [[column, str(count)] for column, count in summary["missing_fields"].items()],
            ),
            "",
        ]
    )

    if summary["non_numeric_fields"]:
        lines.append("Non-numeric measurement values were found.")
    else:
        lines.append("No non-numeric measurement values were found.")

    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _header in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_float(value: float | int) -> str:
    return f"{float(value):.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit synthetic body-label variation.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing labels/labels.csv.")
    parser.add_argument("--output", required=True, help="Directory for summary.json and report.md.")
    args = parser.parse_args(argv)

    summary = audit_synthetic_variation(args.dataset, args.output)
    print(f"Summary: {summary['summary_path']}")
    print(f"Report: {summary['report_path']}")
    print(f"Samples: {summary['sample_count']}")
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    else:
        print("Warnings: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
