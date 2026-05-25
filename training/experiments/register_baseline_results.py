from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
SPLITS = ("train", "val", "test")


def register_baseline_results(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    runs = [load_baseline_run(run_dir) for run_dir in run_dirs]
    if not runs:
        raise ValueError("At least one run directory is required.")

    summary = build_registry_summary(runs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME

    _write_json(summary_path, summary)
    report_path.write_text(format_registry_report(summary), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def load_baseline_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    config_path = run_path / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for run '{run_path}': {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for run '{run_path}': {config_path}")

    metrics = _read_json(metrics_path)
    config = _read_json(config_path)
    model_config = config.get("model", {})
    feature_extractor = config.get("feature_extractor", {})

    return {
        "run_name": run_path.name,
        "run_dir": str(run_path),
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
        "dataset": config.get("dataset", ""),
        "model_type": model_config.get("type", metrics.get("model_family", metrics.get("model_type", ""))),
        "model_artifact_type": model_config.get("artifact_type", metrics.get("model_type", "")),
        "feature_extractor_name": feature_extractor.get("name", ""),
        "feature_extractor_version": feature_extractor.get("version", ""),
        "feature_count": int(config.get("feature_count", metrics.get("feature_count", 0))),
        "sample_counts": metrics.get("sample_counts", {}),
        "target_columns": list(metrics.get("target_columns", config.get("target_columns", []))),
        "metrics": metrics,
        "config": config,
    }


def build_registry_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    target_columns = _target_columns(runs)
    ranked_runs = sorted(
        (run_registry_record(run) for run in runs),
        key=lambda row: row["test_mae"],
    )
    best_run = ranked_runs[0]
    best_test_mae = float(best_run["test_mae"])
    for rank, run in enumerate(ranked_runs, start=1):
        run["rank"] = rank
        run["is_current_best"] = run["run_name"] == best_run["run_name"]
        run["test_mae_delta_vs_best"] = float(run["test_mae"]) - best_test_mae
        run["regression_vs_best"] = run["test_mae_delta_vs_best"] > 0

    per_target_best = best_run_per_target(runs, target_columns)
    target_win_counts = {
        run["run_name"]: sum(1 for row in per_target_best.values() if row["run_name"] == run["run_name"])
        for run in runs
    }
    best_metrics = next(run["metrics"] for run in runs if run["run_name"] == best_run["run_name"])
    hardest_targets = sorted(
        (
            {
                "target": target,
                "test_mae": float(best_metrics["test"]["mae_by_target"][target]),
            }
            for target in target_columns
        ),
        key=lambda row: row["test_mae"],
        reverse=True,
    )

    return {
        "run_names": [run["run_name"] for run in runs],
        "target_columns": target_columns,
        "ranked_runs": ranked_runs,
        "current_best": best_run,
        "per_target_best": per_target_best,
        "per_target_win_counts": target_win_counts,
        "hardest_targets_for_current_best": hardest_targets,
        "recommendations": recommendations(best_run, ranked_runs, hardest_targets),
    }


def run_registry_record(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    return {
        "run_name": run["run_name"],
        "run_dir": run["run_dir"],
        "dataset": run["dataset"],
        "model_type": run["model_type"],
        "model_artifact_type": run["model_artifact_type"],
        "feature_extractor": run["feature_extractor_name"],
        "feature_extractor_version": run["feature_extractor_version"],
        "feature_count": run["feature_count"],
        "sample_counts": run["sample_counts"],
        "train_mae": float(metrics["train"]["overall_mae"]),
        "val_mae": float(metrics["val"]["overall_mae"]),
        "test_mae": float(metrics["test"]["overall_mae"]),
    }


def best_run_per_target(runs: list[dict[str, Any]], target_columns: list[str]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for target in target_columns:
        values = {
            run["run_name"]: float(run["metrics"]["test"]["mae_by_target"][target])
            for run in runs
        }
        best_name = min(values, key=values.get)
        winners[target] = {
            "run_name": best_name,
            "test_mae": values[best_name],
            "all_runs": values,
        }
    return winners


def format_registry_report(summary: dict[str, Any]) -> str:
    best = summary["current_best"]
    lines = [
        "# Baseline Registry",
        "",
        f"Current best run: `{best['run_name']}`",
        f"Current best test MAE: {_format_float(best['test_mae'])}",
        "",
        "## Ranked Runs",
        "",
        _markdown_table(
            [
                "Rank",
                "Run",
                "Dataset",
                "Model",
                "Features",
                "Feature Version",
                "Train MAE",
                "Val MAE",
                "Test MAE",
                "Delta vs Best",
            ],
            [
                [
                    str(row["rank"]),
                    row["run_name"],
                    row["dataset"],
                    row["model_type"],
                    str(row["feature_count"]),
                    row["feature_extractor_version"],
                    _format_float(row["train_mae"]),
                    _format_float(row["val_mae"]),
                    _format_float(row["test_mae"]),
                    _format_float(row["test_mae_delta_vs_best"]),
                ]
                for row in summary["ranked_runs"]
            ],
        ),
        "",
        "## Per-Target Winners",
        "",
        _markdown_table(
            ["Target", "Best Run", "Best Test MAE"],
            [
                [
                    target,
                    row["run_name"],
                    _format_float(row["test_mae"]),
                ]
                for target, row in summary["per_target_best"].items()
            ],
        ),
        "",
        "## Target Win Counts",
        "",
        _markdown_table(
            ["Run", "Target Wins"],
            [
                [run_name, str(count)]
                for run_name, count in summary["per_target_win_counts"].items()
            ],
        ),
        "",
        "## Hardest Targets For Current Best",
        "",
        _markdown_table(
            ["Target", "Test MAE"],
            [
                [row["target"], _format_float(row["test_mae"])]
                for row in summary["hardest_targets_for_current_best"][:5]
            ],
        ),
        "",
        "## Recommendations",
        "",
        *[f"- {recommendation}" for recommendation in summary["recommendations"]],
        "",
    ]
    return "\n".join(lines)


def recommendations(
    best_run: dict[str, Any],
    ranked_runs: list[dict[str, Any]],
    hardest_targets: list[dict[str, Any]],
) -> list[str]:
    notes = [
        f"Use {best_run['run_name']} as the current benchmark unless a future run beats its test MAE of {_format_float(best_run['test_mae'])}.",
    ]
    regressions = [run for run in ranked_runs if run["test_mae_delta_vs_best"] > 0]
    if regressions:
        regressed_names = ", ".join(run["run_name"] for run in regressions)
        notes.append(f"Do not promote regressed runs without a new reason: {regressed_names}.")
    if hardest_targets:
        target_list = ", ".join(row["target"] for row in hardest_targets[:3])
        notes.append(f"Prioritize next modeling work on the current hardest targets: {target_list}.")
    notes.append("Keep the default lightweight baseline as regular ridge until a same-dataset experiment improves test MAE.")
    return notes


def _target_columns(runs: list[dict[str, Any]]) -> list[str]:
    targets = list(runs[0].get("target_columns", []))
    if not targets:
        raise ValueError(f"Run '{runs[0]['run_name']}' has no target_columns.")
    for run in runs[1:]:
        run_targets = list(run.get("target_columns", []))
        if run_targets != targets:
            raise ValueError(f"Run '{run['run_name']}' target_columns do not match the first run.")
    return targets


def _read_json(path: Path) -> Any:
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
    parser = argparse.ArgumentParser(description="Register and rank baseline experiment results.")
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment directories containing metrics.json and config.json.")
    parser.add_argument("--output", required=True, help="Output directory for summary.json and report.md.")
    args = parser.parse_args(argv)

    result = register_baseline_results(args.runs, args.output)
    best = result["summary"]["current_best"]
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Current best: {best['run_name']} test MAE {best['test_mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
