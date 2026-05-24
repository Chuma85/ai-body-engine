from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
SPLITS = ("train", "val", "test")


def analyze_baseline_errors(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    runs = [load_metrics_run(run_dir) for run_dir in run_dirs]
    if not runs:
        raise ValueError("At least one run directory is required.")

    summary = build_comparison_summary(runs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    report_path.write_text(format_markdown_report(summary), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def load_metrics_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for run '{run_path}': {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    return {
        "run_name": run_path.name,
        "run_dir": str(run_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def build_comparison_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    target_columns = _target_columns(runs)
    overall_mae = _overall_mae_table(runs)
    per_target_mae = _per_target_mae_table(runs, target_columns, split="test")
    best_run_per_target = _best_run_per_target(per_target_mae)
    worst_targets = sorted(
        (
            {
                "target": target,
                "best_run": best_run_per_target[target]["run_name"],
                "best_mae": best_run_per_target[target]["mae"],
                "worst_mae": max(values.values()),
            }
            for target, values in per_target_mae.items()
        ),
        key=lambda row: row["best_mae"],
        reverse=True,
    )
    improvement = _pairwise_improvement(runs, target_columns, split="test") if len(runs) >= 2 else {}

    return {
        "run_names": [run["run_name"] for run in runs],
        "target_columns": target_columns,
        "overall_mae": overall_mae,
        "per_target_test_mae": per_target_mae,
        "best_run_per_target": best_run_per_target,
        "worst_targets": worst_targets,
        "pairwise_test_improvement": improvement,
        "latest_vs_previous_test_improvement": _pairwise_improvement_between(runs[-2], runs[-1], target_columns, split="test") if len(runs) >= 3 else improvement,
        "recommendations": _recommendations(worst_targets, improvement),
    }


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Baseline Error Analysis",
        "",
        f"Runs: {', '.join(summary['run_names'])}",
        "",
        "## Overall MAE",
        "",
        _markdown_table(
            ["Run", "Train MAE", "Val MAE", "Test MAE"],
            [
                [
                    run_name,
                    _format_float(summary["overall_mae"][run_name]["train"]),
                    _format_float(summary["overall_mae"][run_name]["val"]),
                    _format_float(summary["overall_mae"][run_name]["test"]),
                ]
                for run_name in summary["run_names"]
            ],
        ),
        "",
        "## Per-Target Test MAE",
        "",
        _markdown_table(
            ["Target", *summary["run_names"], "Best Run"],
            [
                [
                    target,
                    *[_format_float(summary["per_target_test_mae"][target][run_name]) for run_name in summary["run_names"]],
                    summary["best_run_per_target"][target]["run_name"],
                ]
                for target in summary["target_columns"]
            ],
        ),
        "",
        "## Top Improved Targets",
        "",
    ]

    improvement = summary["pairwise_test_improvement"]
    improved_targets = improvement.get("top_improved_targets", [])
    if improved_targets:
        lines.append(
            _markdown_table(
                ["Target", "Baseline", "Candidate", "MAE Delta", "Percent"],
                [
                    [
                        row["target"],
                        row["baseline_run"],
                        row["candidate_run"],
                        _format_float(row["mae_delta"]),
                        _format_float(row["percent_improvement"]),
                    ]
                    for row in improved_targets
                ],
            )
        )
    else:
        lines.append("No improved targets available.")

    latest_vs_previous = summary.get("latest_vs_previous_test_improvement", {})
    if latest_vs_previous and latest_vs_previous != improvement:
        lines.extend(
            [
                "",
                "## Latest Run vs Previous Run",
                "",
                f"{latest_vs_previous['candidate_run']} test MAE delta versus {latest_vs_previous['baseline_run']}: {_format_float(latest_vs_previous['overall_mae_delta'])}",
                "",
            ]
        )
        if latest_vs_previous.get("top_regressed_targets"):
            lines.append(
                _markdown_table(
                    ["Regressed Target", "Previous", "Latest", "MAE Delta"],
                    [
                        [
                            row["target"],
                            _format_float(row["baseline_mae"]),
                            _format_float(row["candidate_mae"]),
                            _format_float(row["mae_delta"]),
                        ]
                        for row in latest_vs_previous["top_regressed_targets"][:5]
                    ],
                )
            )

    lines.extend(
        [
            "",
            "## Worst Targets",
            "",
            _markdown_table(
                ["Target", "Best Run", "Best Test MAE", "Worst Test MAE"],
                [
                    [
                        row["target"],
                        row["best_run"],
                        _format_float(row["best_mae"]),
                        _format_float(row["worst_mae"]),
                    ]
                    for row in summary["worst_targets"][:5]
                ],
            ),
            "",
            "## Recommendations",
            "",
            *[f"- {recommendation}" for recommendation in summary["recommendations"]],
            "",
        ]
    )
    return "\n".join(lines)


def _target_columns(runs: list[dict[str, Any]]) -> list[str]:
    first_targets = list(runs[0]["metrics"].get("target_columns", []))
    if not first_targets:
        raise ValueError(f"Run '{runs[0]['run_name']}' metrics.json has no target_columns.")

    for run in runs[1:]:
        targets = list(run["metrics"].get("target_columns", []))
        if targets != first_targets:
            raise ValueError(f"Run '{run['run_name']}' target_columns do not match the first run.")
    return first_targets


def _overall_mae_table(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for run in runs:
        metrics = run["metrics"]
        table[run["run_name"]] = {split: float(metrics[split]["overall_mae"]) for split in SPLITS}
    return table


def _per_target_mae_table(
    runs: list[dict[str, Any]],
    target_columns: list[str],
    split: str,
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for target in target_columns:
        table[target] = {}
        for run in runs:
            table[target][run["run_name"]] = float(run["metrics"][split]["mae_by_target"][target])
    return table


def _best_run_per_target(per_target_mae: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    return {
        target: {
            "run_name": min(values, key=values.get),
            "mae": min(values.values()),
        }
        for target, values in per_target_mae.items()
    }


def _pairwise_improvement(
    runs: list[dict[str, Any]],
    target_columns: list[str],
    split: str,
) -> dict[str, Any]:
    return _pairwise_improvement_between(runs[0], runs[-1], target_columns, split)


def _pairwise_improvement_between(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    target_columns: list[str],
    split: str,
) -> dict[str, Any]:
    baseline_name = baseline["run_name"]
    candidate_name = candidate["run_name"]
    baseline_overall = float(baseline["metrics"][split]["overall_mae"])
    candidate_overall = float(candidate["metrics"][split]["overall_mae"])
    target_improvements = []

    for target in target_columns:
        baseline_mae = float(baseline["metrics"][split]["mae_by_target"][target])
        candidate_mae = float(candidate["metrics"][split]["mae_by_target"][target])
        delta = baseline_mae - candidate_mae
        target_improvements.append(
            {
                "target": target,
                "baseline_run": baseline_name,
                "candidate_run": candidate_name,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "mae_delta": delta,
                "percent_improvement": (delta / baseline_mae) * 100 if baseline_mae else 0.0,
            }
        )

    sorted_improvements = sorted(target_improvements, key=lambda row: row["mae_delta"], reverse=True)
    sorted_regressions = sorted(target_improvements, key=lambda row: row["mae_delta"])

    return {
        "baseline_run": baseline_name,
        "candidate_run": candidate_name,
        "split": split,
        "overall_mae_delta": baseline_overall - candidate_overall,
        "overall_percent_improvement": ((baseline_overall - candidate_overall) / baseline_overall) * 100 if baseline_overall else 0.0,
        "targets": target_improvements,
        "top_improved_targets": [row for row in sorted_improvements if row["mae_delta"] > 0][:5],
        "top_regressed_targets": [row for row in sorted_regressions if row["mae_delta"] < 0][:5],
    }


def _recommendations(worst_targets: list[dict[str, Any]], improvement: dict[str, Any]) -> list[str]:
    recommendations = []
    if improvement:
        if improvement["overall_mae_delta"] > 0:
            recommendations.append(
                f"Continue image-feature modeling: {improvement['candidate_run']} improves test MAE over {improvement['baseline_run']} by {_format_float(improvement['overall_mae_delta'])}."
            )
        else:
            recommendations.append("Review image features before moving deeper; the candidate did not improve overall test MAE.")

        regressed = improvement.get("top_regressed_targets", [])
        if regressed:
            target_list = ", ".join(row["target"] for row in regressed[:3])
            recommendations.append(f"Prioritize feature work for regressed targets: {target_list}.")

    if worst_targets:
        target_list = ", ".join(row["target"] for row in worst_targets[:3])
        recommendations.append(f"Prioritize next modeling pass on highest-error targets: {target_list}.")
    recommendations.append("Before deep learning, add richer silhouette/profile features or a larger synthetic validation set for error stability.")
    return recommendations


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _header in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare baseline measurement metrics and write error analysis reports.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories containing metrics.json files.")
    parser.add_argument("--output", required=True, help="Output directory for summary.json and report.md.")
    args = parser.parse_args(argv)

    result = analyze_baseline_errors(args.runs, args.output)
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    improvement = result["summary"].get("pairwise_test_improvement", {})
    if improvement:
        print(
            "Test overall MAE delta "
            f"({improvement['baseline_run']} - {improvement['candidate_run']}): "
            f"{improvement['overall_mae_delta']:.4f}"
        )
    latest = result["summary"].get("latest_vs_previous_test_improvement", {})
    if latest and latest != improvement:
        print(
            "Latest vs previous test MAE delta "
            f"({latest['baseline_run']} - {latest['candidate_run']}): "
            f"{latest['overall_mae_delta']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
