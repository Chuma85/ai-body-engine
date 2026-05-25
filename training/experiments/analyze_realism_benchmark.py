from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
PER_TARGET_FILENAME = "per_target_comparison.csv"
SPLITS = ("train", "val", "test")


def analyze_realism_benchmark(
    phase_2v_run_dir: str | Path,
    phase_3h_ridge_run_dir: str | Path,
    phase_3h_cnn_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    phase_2v = load_metrics_run(phase_2v_run_dir, label="phase_2v_ridge")
    phase_3h_ridge = load_metrics_run(phase_3h_ridge_run_dir, label="phase_3h_ridge")
    phase_3h_cnn = load_metrics_run(phase_3h_cnn_run_dir, label="phase_3h_cnn")
    summary = build_realism_summary(phase_2v, phase_3h_ridge, phase_3h_cnn)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    per_target_path = output_path / PER_TARGET_FILENAME
    _write_json(summary_path, summary)
    report_path.write_text(format_realism_report(summary), encoding="utf-8")
    write_per_target_csv(per_target_path, summary["per_target_comparison"])

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "per_target_path": str(per_target_path),
        "summary": summary,
    }


def load_metrics_run(run_dir: str | Path, label: str) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for {label}: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return {
        "label": label,
        "run_name": run_path.name,
        "run_dir": str(run_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def build_realism_summary(phase_2v: dict[str, Any], phase_3h_ridge: dict[str, Any], phase_3h_cnn: dict[str, Any]) -> dict[str, Any]:
    target_columns = _target_columns([phase_2v, phase_3h_ridge, phase_3h_cnn])
    overall_mae = {
        run["label"]: {
            split: float(run["metrics"][split]["overall_mae"])
            for split in SPLITS
        }
        for run in (phase_2v, phase_3h_ridge, phase_3h_cnn)
    }
    per_target = per_target_comparison_rows(phase_2v, phase_3h_ridge, phase_3h_cnn, target_columns)
    ridge_improvements = sorted(per_target, key=lambda row: row["ridge_improvement_mae"], reverse=True)
    ridge_regressions = sorted(per_target, key=lambda row: row["ridge_improvement_mae"])
    cnn_gaps = sorted(per_target, key=lambda row: row["cnn_gap_vs_ridge_mae"], reverse=True)
    hardest_after_realism = sorted(per_target, key=lambda row: row["phase_3h_ridge_mae"], reverse=True)
    split_gaps = split_gap_summary(overall_mae)

    return {
        "run_names": {
            "phase_2v_ridge": phase_2v["run_name"],
            "phase_3h_ridge": phase_3h_ridge["run_name"],
            "phase_3h_cnn": phase_3h_cnn["run_name"],
        },
        "run_dirs": {
            "phase_2v_ridge": phase_2v["run_dir"],
            "phase_3h_ridge": phase_3h_ridge["run_dir"],
            "phase_3h_cnn": phase_3h_cnn["run_dir"],
        },
        "target_columns": target_columns,
        "overall_mae": overall_mae,
        "overall_improvement_phase_2v_to_3h_ridge": overall_mae["phase_2v_ridge"]["test"] - overall_mae["phase_3h_ridge"]["test"],
        "overall_gap_phase_3h_cnn_vs_ridge": overall_mae["phase_3h_cnn"]["test"] - overall_mae["phase_3h_ridge"]["test"],
        "per_target_comparison": per_target,
        "top_improved_targets": [row for row in ridge_improvements if row["ridge_improvement_mae"] > 0][:5],
        "regressed_targets_after_realism": [row for row in ridge_regressions if row["ridge_improvement_mae"] < 0],
        "largest_cnn_gaps": cnn_gaps[:5],
        "hardest_targets_after_realism": hardest_after_realism[:5],
        "split_gaps": split_gaps,
        "cnn_underperformance": cnn_underperformance_summary(per_target, overall_mae),
        "recommendations": recommendations(per_target, overall_mae),
    }


def per_target_comparison_rows(
    phase_2v: dict[str, Any],
    phase_3h_ridge: dict[str, Any],
    phase_3h_cnn: dict[str, Any],
    target_columns: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for target in target_columns:
        phase_2v_mae = float(phase_2v["metrics"]["test"]["mae_by_target"][target])
        phase_3h_ridge_mae = float(phase_3h_ridge["metrics"]["test"]["mae_by_target"][target])
        phase_3h_cnn_mae = float(phase_3h_cnn["metrics"]["test"]["mae_by_target"][target])
        improvement = phase_2v_mae - phase_3h_ridge_mae
        cnn_gap = phase_3h_cnn_mae - phase_3h_ridge_mae
        rows.append(
            {
                "target": target,
                "phase_2v_ridge_mae": phase_2v_mae,
                "phase_3h_ridge_mae": phase_3h_ridge_mae,
                "phase_3h_cnn_mae": phase_3h_cnn_mae,
                "ridge_improvement_mae": improvement,
                "ridge_improvement_percent": (improvement / phase_2v_mae) * 100 if phase_2v_mae else 0.0,
                "cnn_gap_vs_ridge_mae": cnn_gap,
                "cnn_gap_vs_ridge_percent": (cnn_gap / phase_3h_ridge_mae) * 100 if phase_3h_ridge_mae else 0.0,
                "ridge_improved": improvement > 0,
                "cnn_trails_ridge": cnn_gap > 0,
            }
        )
    return rows


def split_gap_summary(overall_mae: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        run_name: {
            "train_to_val": values["val"] - values["train"],
            "val_to_test": values["test"] - values["val"],
            "train_to_test": values["test"] - values["train"],
        }
        for run_name, values in overall_mae.items()
    }


def cnn_underperformance_summary(per_target_rows: list[dict[str, Any]], overall_mae: dict[str, dict[str, float]]) -> dict[str, Any]:
    trailing_targets = [row for row in per_target_rows if row["cnn_trails_ridge"]]
    leading_targets = [row for row in per_target_rows if not row["cnn_trails_ridge"]]
    return {
        "cnn_test_mae_gap_vs_ridge": overall_mae["phase_3h_cnn"]["test"] - overall_mae["phase_3h_ridge"]["test"],
        "targets_trailing_ridge_count": len(trailing_targets),
        "targets_beating_or_matching_ridge_count": len(leading_targets),
        "targets_trailing_ridge": [row["target"] for row in trailing_targets],
        "targets_beating_or_matching_ridge": [row["target"] for row in leading_targets],
        "interpretation": "global_underperformance" if len(trailing_targets) > len(per_target_rows) / 2 else "target_specific_underperformance",
    }


def recommendations(per_target_rows: list[dict[str, Any]], overall_mae: dict[str, dict[str, float]]) -> list[str]:
    hardest = sorted(per_target_rows, key=lambda row: row["phase_3h_ridge_mae"], reverse=True)[:3]
    largest_gaps = sorted(per_target_rows, key=lambda row: row["cnn_gap_vs_ridge_mae"], reverse=True)[:3]
    notes = [
        "Use Phase 3H ridge as the current benchmark; it remains strongest overall.",
        "Realism-enabled rendering is worth keeping because it substantially improved ridge image-feature performance.",
        "CNN underperforms ridge globally on the Phase 3H split; improve training/modeling before scaling CNN runs.",
    ]
    notes.append("Prioritize hardest current targets: " + ", ".join(row["target"] for row in hardest) + ".")
    notes.append("If improving CNN next, focus on largest ridge gaps: " + ", ".join(row["target"] for row in largest_gaps) + ".")
    return notes


def format_realism_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 3J Realism Benchmark Analysis",
            "",
            "## Overall MAE",
            "",
            _markdown_table(
                ["Run", "Train MAE", "Val MAE", "Test MAE", "Train-Test Gap"],
                [
                    [
                        run_name,
                        _format_float(values["train"]),
                        _format_float(values["val"]),
                        _format_float(values["test"]),
                        _format_float(summary["split_gaps"][run_name]["train_to_test"]),
                    ]
                    for run_name, values in summary["overall_mae"].items()
                ],
            ),
            "",
            f"Phase 2V ridge to Phase 3H ridge test MAE improvement: {_format_float(summary['overall_improvement_phase_2v_to_3h_ridge'])}",
            f"Phase 3H CNN test MAE gap versus Phase 3H ridge: {_format_float(summary['overall_gap_phase_3h_cnn_vs_ridge'])}",
            "",
            "## Top Improved Targets",
            "",
            _target_table(summary["top_improved_targets"], include_improvement=True),
            "",
            "## Largest CNN Gaps Versus Ridge",
            "",
            _target_table(summary["largest_cnn_gaps"], include_gap=True),
            "",
            "## Hardest Targets After Realism",
            "",
            _markdown_table(
                ["Target", "Phase 3H Ridge Test MAE"],
                [[row["target"], _format_float(row["phase_3h_ridge_mae"])] for row in summary["hardest_targets_after_realism"]],
            ),
            "",
            "## CNN Underperformance",
            "",
            f"Interpretation: `{summary['cnn_underperformance']['interpretation']}`",
            f"Targets where CNN trails ridge: {summary['cnn_underperformance']['targets_trailing_ridge_count']}/{len(summary['target_columns'])}",
            "",
            "## Recommendations",
            "",
            *[f"- {recommendation}" for recommendation in summary["recommendations"]],
            "",
        ]
    )


def write_per_target_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target",
        "phase_2v_ridge_mae",
        "phase_3h_ridge_mae",
        "phase_3h_cnn_mae",
        "ridge_improvement_mae",
        "ridge_improvement_percent",
        "cnn_gap_vs_ridge_mae",
        "cnn_gap_vs_ridge_percent",
        "ridge_improved",
        "cnn_trails_ridge",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _target_columns(runs: list[dict[str, Any]]) -> list[str]:
    targets = list(runs[0]["metrics"].get("target_columns", []))
    if not targets:
        raise ValueError(f"Run '{runs[0]['run_name']}' metrics.json has no target_columns.")
    for run in runs[1:]:
        if list(run["metrics"].get("target_columns", [])) != targets:
            raise ValueError(f"Run '{run['run_name']}' target_columns do not match the first run.")
    return targets


def _target_table(rows: list[dict[str, Any]], include_improvement: bool = False, include_gap: bool = False) -> str:
    headers = ["Target", "Phase 2V Ridge", "Phase 3H Ridge", "Phase 3H CNN"]
    if include_improvement:
        headers.extend(["Ridge Improvement", "Improvement %"])
    if include_gap:
        headers.extend(["CNN Gap", "Gap %"])
    table_rows = []
    for row in rows:
        values = [
            row["target"],
            _format_float(row["phase_2v_ridge_mae"]),
            _format_float(row["phase_3h_ridge_mae"]),
            _format_float(row["phase_3h_cnn_mae"]),
        ]
        if include_improvement:
            values.extend([_format_float(row["ridge_improvement_mae"]), _format_float(row["ridge_improvement_percent"])])
        if include_gap:
            values.extend([_format_float(row["cnn_gap_vs_ridge_mae"]), _format_float(row["cnn_gap_vs_ridge_percent"])])
        table_rows.append(values)
    return _markdown_table(headers, table_rows)


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
    parser = argparse.ArgumentParser(description="Analyze Phase 3H realism benchmark improvements.")
    parser.add_argument("--phase-2v", required=True, help="Phase 2V ridge artifact directory.")
    parser.add_argument("--phase-3h-ridge", required=True, help="Phase 3H ridge artifact directory.")
    parser.add_argument("--phase-3h-cnn", required=True, help="Phase 3H CNN artifact directory.")
    parser.add_argument("--output", required=True, help="Output directory for analysis artifacts.")
    args = parser.parse_args(argv)

    result = analyze_realism_benchmark(args.phase_2v, args.phase_3h_ridge, args.phase_3h_cnn, args.output)
    summary = result["summary"]
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Per-target CSV: {result['per_target_path']}")
    print(f"Phase 2V -> Phase 3H ridge test MAE improvement: {summary['overall_improvement_phase_2v_to_3h_ridge']:.4f}")
    print(f"Phase 3H CNN gap vs ridge: {summary['overall_gap_phase_3h_cnn_vs_ridge']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
