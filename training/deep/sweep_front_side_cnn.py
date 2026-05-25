from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from training.deep.dependencies import DeepLearningDependencyError
from training.deep.train_front_side_cnn import train_front_side_cnn

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
DEFAULT_IMAGE_SIZES = (96, 128)
DEFAULT_LEARNING_RATES = (0.001, 0.0005)
DEFAULT_BATCH_SIZES = (16, 32)
DEFAULT_WEIGHT_DECAYS = (0.0, 0.0001)


def sweep_front_side_cnn(
    dataset_root: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    patience: int = 3,
    max_runs: int = 4,
    device: str = "cpu",
    seed: int = 42,
    dry_run: bool = False,
    limit_samples: int | None = None,
) -> dict[str, Any]:
    if max_runs <= 0:
        raise ValueError("max_runs must be a positive integer.")
    planned_runs = generate_sweep_grid()[:max_runs]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    if not dry_run:
        for index, config in enumerate(planned_runs, start=1):
            run_name = run_directory_name(index, config)
            run_output = output_path / run_name
            try:
                result = train_front_side_cnn(
                    dataset_root,
                    run_output,
                    epochs=epochs,
                    patience=patience,
                    image_size=int(config["image_size"]),
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                    device=device,
                    seed=seed,
                    limit_samples=limit_samples,
                )
                metrics = result["metrics"]
                run_results.append(
                    {
                        "run_name": run_name,
                        "run_dir": str(run_output),
                        "status": "completed",
                        "config": config,
                        "train_mae": float(metrics["train"]["overall_mae"]),
                        "val_mae": float(metrics["val"]["overall_mae"]),
                        "test_mae": float(metrics["test"]["overall_mae"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_completed": int(metrics["epochs_completed"]),
                        "early_stopping_triggered": bool(metrics["early_stopping_triggered"]),
                    }
                )
            except (Exception, DeepLearningDependencyError) as error:
                run_results.append(
                    {
                        "run_name": run_name,
                        "run_dir": str(run_output),
                        "status": "failed",
                        "config": config,
                        "error": str(error),
                    }
                )
                break

    summary = build_sweep_summary(
        dataset_root=dataset_root,
        output_dir=output_path,
        planned_runs=planned_runs,
        run_results=run_results,
        epochs=epochs,
        patience=patience,
        max_runs=max_runs,
        device=device,
        seed=seed,
        dry_run=dry_run,
        limit_samples=limit_samples,
    )
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    _write_json(summary_path, summary)
    report_path.write_text(format_sweep_report(summary), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def generate_sweep_grid() -> list[dict[str, float | int]]:
    candidates = [
        {
            "image_size": image_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }
        for image_size in DEFAULT_IMAGE_SIZES
        for learning_rate in DEFAULT_LEARNING_RATES
        for batch_size in DEFAULT_BATCH_SIZES
        for weight_decay in DEFAULT_WEIGHT_DECAYS
    ]
    priority = [
        {"image_size": 128, "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0},
        {"image_size": 128, "learning_rate": 0.0005, "batch_size": 32, "weight_decay": 0.0},
        {"image_size": 96, "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0},
        {"image_size": 128, "learning_rate": 0.001, "batch_size": 16, "weight_decay": 0.0},
        {"image_size": 128, "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0001},
        {"image_size": 96, "learning_rate": 0.0005, "batch_size": 32, "weight_decay": 0.0},
    ]
    ordered = []
    for config in priority:
        if config in candidates and config not in ordered:
            ordered.append(config)
    for config in candidates:
        if config not in ordered:
            ordered.append(config)
    return ordered


def build_sweep_summary(
    dataset_root: str | Path,
    output_dir: Path,
    planned_runs: list[dict[str, float | int]],
    run_results: list[dict[str, Any]],
    epochs: int,
    patience: int,
    max_runs: int,
    device: str,
    seed: int,
    dry_run: bool,
    limit_samples: int | None,
) -> dict[str, Any]:
    completed_runs = [run for run in run_results if run["status"] == "completed"]
    failed_runs = [run for run in run_results if run["status"] == "failed"]
    ranked_runs = rank_completed_runs(completed_runs)
    best_run = ranked_runs[0] if ranked_runs else None
    return {
        "dataset": str(dataset_root),
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "epochs": epochs,
        "patience": patience,
        "max_runs": max_runs,
        "device": device,
        "seed": seed,
        "limit_samples": limit_samples,
        "planned_run_count": len(planned_runs),
        "completed_run_count": len(completed_runs),
        "failed_run_count": len(failed_runs),
        "planned_runs": [
            {"run_name": run_directory_name(index, config), "config": config}
            for index, config in enumerate(planned_runs, start=1)
        ],
        "runs": run_results,
        "ranked_runs": ranked_runs,
        "best_run_by_val_mae": best_run,
    }


def rank_completed_runs(run_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(run_results, key=lambda row: float(row["val_mae"]))
    return [{**row, "rank": index} for index, row in enumerate(ranked, start=1)]


def run_directory_name(index: int, config: dict[str, float | int]) -> str:
    lr = str(config["learning_rate"]).replace(".", "p")
    wd = str(config["weight_decay"]).replace(".", "p")
    return f"run_{index:03d}_img{config['image_size']}_lr{lr}_bs{config['batch_size']}_wd{wd}"


def format_sweep_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Front/Side CNN Sweep",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Dry run: {summary['dry_run']}",
        f"Planned runs: {summary['planned_run_count']}",
        f"Completed runs: {summary['completed_run_count']}",
        f"Failed runs: {summary['failed_run_count']}",
        "",
        "## Planned Runs",
        "",
        _markdown_table(
            ["Run", "Image", "LR", "Batch", "Weight Decay"],
            [
                [
                    row["run_name"],
                    str(row["config"]["image_size"]),
                    str(row["config"]["learning_rate"]),
                    str(row["config"]["batch_size"]),
                    str(row["config"]["weight_decay"]),
                ]
                for row in summary["planned_runs"]
            ],
        ),
    ]
    if summary["ranked_runs"]:
        lines.extend(
            [
                "",
                "## Results",
                "",
                _markdown_table(
                    ["Rank", "Run", "Train MAE", "Val MAE", "Test MAE", "Best Epoch"],
                    [
                        [
                            str(row["rank"]),
                            row["run_name"],
                            _format_float(row["train_mae"]),
                            _format_float(row["val_mae"]),
                            _format_float(row["test_mae"]),
                            str(row["best_epoch"]),
                        ]
                        for row in summary["ranked_runs"]
                    ],
                ),
                "",
                f"Best run by validation MAE: `{summary['best_run_by_val_mae']['run_name']}`",
                f"Best validation MAE: {_format_float(summary['best_run_by_val_mae']['val_mae'])}",
                f"Best run test MAE: {_format_float(summary['best_run_by_val_mae']['test_mae'])}",
            ]
        )
    if summary["failed_run_count"]:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                *[f"- `{run['run_name']}`: {run['error']}" for run in summary["runs"] if run["status"] == "failed"],
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _header in headers) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small controlled front/side CNN hyperparameter sweep.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Sweep output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = sweep_front_side_cnn(
        args.dataset,
        args.output,
        epochs=args.epochs,
        patience=args.patience,
        max_runs=args.max_runs,
        device=args.device,
        seed=args.seed,
        dry_run=args.dry_run,
        limit_samples=args.limit_samples,
    )
    summary = result["summary"]
    print(f"Planned runs: {summary['planned_run_count']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    if summary["dry_run"]:
        for planned in summary["planned_runs"]:
            print(f"- {planned['run_name']}: {planned['config']}")
    elif summary["best_run_by_val_mae"]:
        best = summary["best_run_by_val_mae"]
        print(f"Best run by val MAE: {best['run_name']} val={best['val_mae']:.4f} test={best['test_mae']:.4f}")
    if summary["failed_run_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
