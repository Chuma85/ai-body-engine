from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from training.experiments.run_image_feature_experiment import (
    DEFAULT_KNN_K,
    SUPPORTED_MODEL_TYPES,
    run_image_feature_experiment,
)

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
SPLITS = ("train", "val", "test")


def compare_image_feature_models(
    dataset_root: str | Path,
    output_dir: str | Path,
    model_types: list[str] | None = None,
    ridge_alpha: float = 10.0,
    knn_k: int = DEFAULT_KNN_K,
) -> dict[str, Any]:
    selected_models = model_types or list(SUPPORTED_MODEL_TYPES)
    for model_type in selected_models:
        _validate_model_type(model_type)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_results = []
    for model_type in selected_models:
        result = run_image_feature_experiment(
            dataset_root,
            output_path / model_type,
            model_type=model_type,
            ridge_alpha=ridge_alpha,
            knn_k=knn_k,
        )
        run_results.append(
            {
                "model_type": model_type,
                "output_dir": result["output_dir"],
                "metrics": result["metrics"],
            }
        )

    summary = build_model_comparison_summary(run_results)
    summary_path = output_path / SUMMARY_FILENAME
    report_path = output_path / REPORT_FILENAME
    _write_json(summary_path, summary)
    report_path.write_text(format_model_comparison_report(summary), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def build_model_comparison_summary(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_results:
        raise ValueError("At least one model run is required for comparison.")

    target_columns = list(run_results[0]["metrics"].get("target_columns", []))
    if not target_columns:
        raise ValueError("Model metrics are missing target_columns.")

    for result in run_results[1:]:
        targets = list(result["metrics"].get("target_columns", []))
        if targets != target_columns:
            raise ValueError(f"Model '{result['model_type']}' target_columns do not match the first run.")

    overall_mae = {
        result["model_type"]: {
            split: float(result["metrics"][split]["overall_mae"])
            for split in SPLITS
        }
        for result in run_results
    }
    per_target_test_mae = {
        target: {
            result["model_type"]: float(result["metrics"]["test"]["mae_by_target"][target])
            for result in run_results
        }
        for target in target_columns
    }
    best_model_overall = min(overall_mae, key=lambda model: overall_mae[model]["test"])
    best_model_per_target = {
        target: {
            "model_type": min(values, key=values.get),
            "mae": min(values.values()),
        }
        for target, values in per_target_test_mae.items()
    }
    worst_targets = sorted(
        (
            {
                "target": target,
                "best_model": best_model_per_target[target]["model_type"],
                "best_mae": best_model_per_target[target]["mae"],
                "worst_mae": max(values.values()),
            }
            for target, values in per_target_test_mae.items()
        ),
        key=lambda row: row["best_mae"],
        reverse=True,
    )

    return {
        "model_types": [result["model_type"] for result in run_results],
        "target_columns": target_columns,
        "feature_count": int(run_results[0]["metrics"]["feature_count"]),
        "sample_counts": run_results[0]["metrics"]["sample_counts"],
        "overall_mae": overall_mae,
        "per_target_test_mae": per_target_test_mae,
        "best_model_overall": {
            "model_type": best_model_overall,
            "test_mae": overall_mae[best_model_overall]["test"],
        },
        "best_model_per_target": best_model_per_target,
        "worst_targets": worst_targets,
        "run_dirs": {
            result["model_type"]: result["output_dir"]
            for result in run_results
        },
    }


def format_model_comparison_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Image Feature Model Family Comparison",
        "",
        f"Models: {', '.join(summary['model_types'])}",
        f"Feature count: {summary['feature_count']}",
        (
            "Samples: "
            f"train={summary['sample_counts']['train']} "
            f"val={summary['sample_counts']['val']} "
            f"test={summary['sample_counts']['test']}"
        ),
        "",
        "## Overall MAE",
        "",
        _markdown_table(
            ["Model", "Train MAE", "Val MAE", "Test MAE"],
            [
                [
                    model_type,
                    _format_float(summary["overall_mae"][model_type]["train"]),
                    _format_float(summary["overall_mae"][model_type]["val"]),
                    _format_float(summary["overall_mae"][model_type]["test"]),
                ]
                for model_type in summary["model_types"]
            ],
        ),
        "",
        (
            "Best overall model: "
            f"{summary['best_model_overall']['model_type']} "
            f"(test MAE {_format_float(summary['best_model_overall']['test_mae'])})"
        ),
        "",
        "## Per-Target Test MAE",
        "",
        _markdown_table(
            ["Target", *summary["model_types"], "Best Model"],
            [
                [
                    target,
                    *[_format_float(summary["per_target_test_mae"][target][model_type]) for model_type in summary["model_types"]],
                    summary["best_model_per_target"][target]["model_type"],
                ]
                for target in summary["target_columns"]
            ],
        ),
        "",
        "## Worst Targets",
        "",
        _markdown_table(
            ["Target", "Best Model", "Best Test MAE", "Worst Test MAE"],
            [
                [
                    row["target"],
                    row["best_model"],
                    _format_float(row["best_mae"]),
                    _format_float(row["worst_mae"]),
                ]
                for row in summary["worst_targets"][:5]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _validate_model_type(model_type: str) -> None:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type '{model_type}'. Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}."
        )


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
    parser = argparse.ArgumentParser(description="Compare lightweight image-feature model families.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing manifest.csv.")
    parser.add_argument("--output", required=True, help="Directory for model comparison outputs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODEL_TYPES),
        help=f"Model types to run. Supported: {', '.join(SUPPORTED_MODEL_TYPES)}.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    args = parser.parse_args(argv)

    result = compare_image_feature_models(
        args.dataset,
        args.output,
        model_types=args.models,
        ridge_alpha=args.ridge_alpha,
        knn_k=args.knn_k,
    )
    summary = result["summary"]
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(
        "Best overall model: "
        f"{summary['best_model_overall']['model_type']} "
        f"(test MAE {summary['best_model_overall']['test_mae']:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
