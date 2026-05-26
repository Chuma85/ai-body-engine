from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
RESULTS_FILENAME = "results.csv"
SPLITS = ("train", "val", "test")


def analyze_render_ablation(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    clean_run_name: str = "phase_3n_clean_baseline_ridge",
    label_equality_confirmed: bool = False,
    sample_count: int | None = None,
) -> dict[str, Any]:
    runs = [load_ablation_run(run_dir) for run_dir in run_dirs]
    if not runs:
        raise ValueError("At least one run directory is required.")

    summary = build_render_ablation_summary(
        runs,
        clean_run_name=clean_run_name,
        label_equality_confirmed=label_equality_confirmed,
        sample_count=sample_count,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    results_path = output_path / RESULTS_FILENAME

    _write_json(summary_path, summary)
    report_path.write_text(format_render_ablation_report(summary), encoding="utf-8")
    write_results_csv(results_path, summary["results"])

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "results_path": str(results_path),
        "summary": summary,
    }


def load_ablation_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    config_path = run_path / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for run '{run_path}': {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for run '{run_path}': {config_path}")

    metrics = _read_json(metrics_path)
    config = _read_json(config_path)

    return {
        "run_name": run_path.name,
        "run_dir": str(run_path),
        "dataset": config.get("dataset", ""),
        "metrics": metrics,
        "config": config,
    }


def build_render_ablation_summary(
    runs: list[dict[str, Any]],
    clean_run_name: str,
    label_equality_confirmed: bool,
    sample_count: int | None,
) -> dict[str, Any]:
    clean_run = next((run for run in runs if run["run_name"] == clean_run_name), None)
    if clean_run is None:
        raise ValueError(f"Clean baseline run '{clean_run_name}' was not found.")

    clean_test_mae = float(clean_run["metrics"]["test"]["overall_mae"])
    results = [
        result_row_for_run(run, clean_test_mae)
        for run in runs
    ]
    ranked = sorted(results, key=lambda row: row["test_mae"])
    best = ranked[0]
    worst = ranked[-1]

    return {
        "run_names": [run["run_name"] for run in runs],
        "clean_run_name": clean_run_name,
        "sample_count": sample_count,
        "split": _sample_counts(clean_run),
        "label_equality_confirmed": label_equality_confirmed,
        "results": results,
        "ranked_results": ranked,
        "best_ablation": best,
        "worst_ablation": worst,
        "recommendations": recommendations(best, worst, clean_test_mae, label_equality_confirmed),
    }


def result_row_for_run(run: dict[str, Any], clean_test_mae: float) -> dict[str, Any]:
    metrics = run["metrics"]
    test_mae = float(metrics["test"]["overall_mae"])
    delta = test_mae - clean_test_mae
    return {
        "ablation": ablation_name(run["run_name"]),
        "run_name": run["run_name"],
        "dataset": run["dataset"],
        "train_mae": float(metrics["train"]["overall_mae"]),
        "val_mae": float(metrics["val"]["overall_mae"]),
        "test_mae": test_mae,
        "delta_vs_clean_test_mae": delta,
        "effect_vs_clean": effect_label(delta),
    }


def ablation_name(run_name: str) -> str:
    name = run_name
    if name.startswith("phase_3n_"):
        name = name[len("phase_3n_") :]
    if name.endswith("_ridge"):
        name = name[: -len("_ridge")]
    return name


def effect_label(delta: float, tolerance: float = 1e-9) -> str:
    if delta < -tolerance:
        return "helped"
    if delta > tolerance:
        return "hurt"
    return "matched"


def format_render_ablation_report(summary: dict[str, Any]) -> str:
    best = summary["best_ablation"]
    worst = summary["worst_ablation"]
    lines = [
        "# Render Realism Ablation",
        "",
        f"Clean baseline run: `{summary['clean_run_name']}`",
        f"Sample count: {summary['sample_count'] or 'unknown'}",
        f"Split: {_format_split(summary.get('split', {}))}",
        f"Label equality confirmed: {summary['label_equality_confirmed']}",
        "",
        "## Results",
        "",
        _markdown_table(
            ["Ablation", "Train MAE", "Val MAE", "Test MAE", "Delta vs Clean", "Effect"],
            [
                [
                    row["ablation"],
                    _format_float(row["train_mae"]),
                    _format_float(row["val_mae"]),
                    _format_float(row["test_mae"]),
                    _format_float(row["delta_vs_clean_test_mae"]),
                    row["effect_vs_clean"],
                ]
                for row in summary["results"]
            ],
        ),
        "",
        "## Best And Worst",
        "",
        f"Best ablation by test MAE: `{best['ablation']}` ({_format_float(best['test_mae'])})",
        f"Worst ablation by test MAE: `{worst['ablation']}` ({_format_float(worst['test_mae'])})",
        "",
        "## Recommendations",
        "",
        *[f"- {recommendation}" for recommendation in summary["recommendations"]],
        "",
    ]
    return "\n".join(lines)


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ablation",
        "run_name",
        "dataset",
        "train_mae",
        "val_mae",
        "test_mae",
        "delta_vs_clean_test_mae",
        "effect_vs_clean",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recommendations(
    best: dict[str, Any],
    worst: dict[str, Any],
    clean_test_mae: float,
    label_equality_confirmed: bool,
) -> list[str]:
    notes = []
    if not label_equality_confirmed:
        notes.append("Do not over-interpret ablation differences until label equality is confirmed.")
    if best["test_mae"] < clean_test_mae:
        notes.append(f"{best['ablation']} improved over the clean baseline; consider scaling that control next.")
    else:
        notes.append("No render realism ablation beat the clean baseline in this run.")
    if worst["effect_vs_clean"] == "hurt":
        notes.append(f"{worst['ablation']} hurt the ridge baseline most; inspect its image features before using it at scale.")
    notes.append("Keep Phase 3L clean ridge as current best unless a same-body ablation beats its 6.5780 test MAE.")
    return notes


def _sample_counts(run: dict[str, Any]) -> dict[str, int]:
    counts = run["metrics"].get("sample_counts", {})
    return {split: int(counts.get(split, 0)) for split in SPLITS}


def _format_split(split: dict[str, int]) -> str:
    if not split:
        return "unknown"
    return ", ".join(f"{name}={split.get(name, 0)}" for name in SPLITS)


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
    parser = argparse.ArgumentParser(description="Analyze same-body render realism ablation runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Experiment directories containing metrics.json and config.json.")
    parser.add_argument("--output", required=True, help="Output directory for summary.json, report.md, and results.csv.")
    parser.add_argument("--clean-run-name", default="phase_3n_clean_baseline_ridge")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--label-equality-confirmed", action="store_true")
    args = parser.parse_args(argv)

    result = analyze_render_ablation(
        args.runs,
        args.output,
        clean_run_name=args.clean_run_name,
        label_equality_confirmed=args.label_equality_confirmed,
        sample_count=args.sample_count,
    )
    summary = result["summary"]
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(f"Results: {result['results_path']}")
    print(f"Best ablation: {summary['best_ablation']['ablation']} test MAE {summary['best_ablation']['test_mae']:.4f}")
    print(f"Worst ablation: {summary['worst_ablation']['ablation']} test MAE {summary['worst_ablation']['test_mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
