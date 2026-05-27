from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.validate_synthetic_dataset import validate_dataset
from training.train_baseline_measurements import TARGET_COLUMNS

DATASET_VALIDATION_JSON = "dataset_validation.json"
DATASET_VALIDATION_CSV = "dataset_validation.csv"
DATASET_VALIDATION_MD = "dataset_validation.md"
BENCHMARK_RESULTS_JSON = "benchmark_results.json"
BENCHMARK_RESULTS_CSV = "benchmark_results.csv"
BENCHMARK_RESULTS_MD = "benchmark_results.md"
PER_TARGET_RESULTS_CSV = "per_target_results.csv"
CNN_HISTORY_JSON = "cnn_training_history.json"
CNN_HISTORY_CSV = "cnn_training_history.csv"
PROMOTION_READINESS_MD = "promotion_readiness.md"

REFERENCE_BASELINES = [
    {
        "run_name": "phase_3l_clean_ridge",
        "source": "phase_3s_reference",
        "dataset": "data/synthetic/phase_3l_clean",
        "feature_group": "all_features",
        "model_type": "ridge",
        "test_mae": 6.5780,
        "notes": "Current best clean 1000-sample ridge anchor from Phase 3L/3M.",
    },
    {
        "run_name": "phase_3r_background_raw_scale_elasticnet",
        "source": "phase_3s_reference",
        "dataset": "data/synthetic/phase_3n_background_only",
        "feature_group": "raw_scale_camera",
        "model_type": "elasticnet",
        "test_mae": 7.0834,
        "notes": "Best Phase 3R regularized candidate.",
    },
    {
        "run_name": "phase_3r_camera_jitter_hybrid_without_area_random_forest",
        "source": "phase_3s_reference",
        "dataset": "data/synthetic/phase_3n_camera_jitter_only",
        "feature_group": "combined_hybrid_without_area_ratios",
        "model_type": "random_forest",
        "test_mae": 7.1666,
        "notes": "Best Phase 3R camera-jitter robust candidate.",
    },
    {
        "run_name": "phase_3r_combined_realism_raw_scale_gradient_boosting",
        "source": "phase_3s_reference",
        "dataset": "data/synthetic/phase_3n_combined_realism",
        "feature_group": "raw_scale_camera",
        "model_type": "gradient_boosting",
        "test_mae": 7.2811,
        "notes": "Phase 3S combined-realism candidate.",
    },
]


def build_phase3t_benchmark_report(
    dataset: str | Path,
    output_dir: str | Path,
    classical_dir: str | Path | None = None,
    cnn_dir: str | Path | None = None,
    audit_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset(dataset_path)
    manifest = read_manifest_summary(dataset_path)
    audit_summary = read_optional_json(Path(audit_dir) / "summary.json") if audit_dir else None
    dataset_summary = {
        "dataset": str(dataset_path),
        "validation": validation,
        "manifest": manifest,
        "audit": audit_summary,
    }

    benchmark_rows = reference_rows()
    per_target_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if classical_dir is not None:
        classical_path = Path(classical_dir)
        rows, targets, parse_warnings = parse_classical_results(classical_path)
        benchmark_rows.extend(rows)
        per_target_rows.extend(targets)
        warnings.extend(parse_warnings)
    if cnn_dir is not None:
        cnn_path = Path(cnn_dir)
        if cnn_path.exists():
            cnn_row, cnn_targets, history = parse_cnn_experiment(cnn_path)
            benchmark_rows.append(cnn_row)
            per_target_rows.extend(cnn_targets)
            write_cnn_history(output_path, history)
        else:
            warnings.append(f"CNN artifact directory is missing: {cnn_path}")

    benchmark_rows = sorted(
        benchmark_rows,
        key=lambda row: (
            float(row.get("test_mae", row.get("overall_test_mae", 10**9))),
            row.get("source", ""),
            row.get("run_name", ""),
        ),
    )
    best_model_per_target = best_per_target(per_target_rows)
    payload = {
        "dataset": str(dataset_path),
        "warnings": warnings,
        "reference_gate": "Phase 3T is experimental; Phase 3L clean ridge remains the promotion anchor unless beaten.",
        "benchmark_rows": benchmark_rows,
        "best_overall": benchmark_rows[0] if benchmark_rows else None,
        "best_model_per_target": best_model_per_target,
        "target_columns": TARGET_COLUMNS,
        "promotion": promotion_gate(float(benchmark_rows[0]["test_mae"])) if benchmark_rows else None,
    }

    paths = write_outputs(output_path, dataset_summary, payload, per_target_rows)
    return {
        "dataset_summary": dataset_summary,
        "benchmark": payload,
        "per_target_rows": per_target_rows,
        **{key: str(value) for key, value in paths.items()},
    }


def read_manifest_summary(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path / "manifest.csv"
    if not manifest_path.exists():
        return {"exists": False, "path": str(manifest_path), "row_count": 0, "split_counts": {}}
    split_counts: dict[str, int] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        split = row.get("dataset_split", "")
        split_counts[split] = split_counts.get(split, 0) + 1
    return {"exists": True, "path": str(manifest_path), "row_count": len(rows), "split_counts": split_counts}


def reference_rows() -> list[dict[str, Any]]:
    rows = []
    for candidate in REFERENCE_BASELINES:
        row = {**candidate}
        row["train_mae"] = ""
        row["val_mae"] = ""
        row["sample_count_test"] = ""
        row["promotion_gate"] = promotion_gate(float(row["test_mae"]))["gate"]
        rows.append(row)
    return rows


def parse_classical_results(classical_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    results_path = classical_dir / "results.csv"
    per_target_path = classical_dir / "per_target_results.csv"
    if not results_path.exists():
        return [], [], [f"Classical benchmark results are missing: {results_path}"]

    rows: list[dict[str, Any]] = []
    with results_path.open("r", newline="", encoding="utf-8") as csv_file:
        for raw in csv.DictReader(csv_file):
            run_name = f"phase_3t_{raw['feature_config']}__{raw['model_type']}"
            test_mae = float(raw["test_mae"])
            rows.append(
                {
                    "run_name": run_name,
                    "source": "phase_3t_classical",
                    "dataset": raw.get("dataset_path") or raw.get("dataset", ""),
                    "feature_group": raw["feature_config"],
                    "model_type": raw["model_type"],
                    "feature_count": raw.get("feature_count", ""),
                    "train_mae": float(raw["train_mae"]),
                    "val_mae": float(raw["val_mae"]),
                    "test_mae": test_mae,
                    "sample_count_test": "",
                    "promotion_gate": promotion_gate(test_mae)["gate"],
                    "notes": "Phase 3T classical benchmark on the realistic 1000-sample dataset.",
                }
            )

    per_target_rows: list[dict[str, Any]] = []
    if per_target_path.exists():
        with per_target_path.open("r", newline="", encoding="utf-8") as csv_file:
            for raw in csv.DictReader(csv_file):
                per_target_rows.append(
                    {
                        "run_name": f"phase_3t_{raw['feature_config']}__{raw['model_type']}",
                        "source": "phase_3t_classical",
                        "dataset": raw.get("dataset", ""),
                        "feature_group": raw["feature_config"],
                        "model_type": raw["model_type"],
                        "target": raw["target"],
                        "test_mae": float(raw["test_mae"]),
                    }
                )
    else:
        warnings.append(f"Classical per-target results are missing: {per_target_path}")
    return rows, per_target_rows, warnings


def parse_cnn_experiment(cnn_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_path = cnn_dir / "metrics.json"
    config_path = cnn_dir / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing CNN metrics.json: {metrics_path}")
    metrics = read_json(metrics_path)
    config = read_json(config_path) if config_path.exists() else {}
    test_mae = float(metrics["test"]["overall_mae"])
    row = {
        "run_name": cnn_dir.name,
        "source": "phase_3t_cnn",
        "dataset": config.get("dataset", ""),
        "feature_group": "front_side_images",
        "model_type": metrics.get("model_type", metrics.get("model_choice", config.get("model_type", "cnn"))),
        "feature_count": "",
        "train_mae": float(metrics["train"]["overall_mae"]),
        "val_mae": float(metrics["val"]["overall_mae"]),
        "test_mae": test_mae,
        "sample_count_test": metrics.get("sample_counts", {}).get("test", ""),
        "promotion_gate": promotion_gate(test_mae)["gate"],
        "notes": "Phase 3T front/side CNN benchmark.",
    }
    target_rows = [
        {
            "run_name": cnn_dir.name,
            "source": "phase_3t_cnn",
            "dataset": config.get("dataset", ""),
            "feature_group": "front_side_images",
            "model_type": row["model_type"],
            "target": target,
            "test_mae": float(mae),
        }
        for target, mae in metrics.get("test", {}).get("mae_by_target", {}).items()
    ]
    history = metrics.get("epoch_metrics", [])
    return row, target_rows, history


def best_per_target(per_target_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        rows = [row for row in per_target_rows if row.get("target") == target]
        if rows:
            best[target] = min(rows, key=lambda row: float(row["test_mae"]))
    return best


def promotion_gate(overall_mae: float) -> dict[str, str]:
    if overall_mae > 5.0:
        return {
            "gate": "research_only",
            "description": "Research only: overall MAE is above 5 cm and is not production-ready for tailoring.",
        }
    if overall_mae > 3.0:
        return {
            "gate": "assisted_sizing_manual_confirmation",
            "description": "Assisted sizing candidate: 3-5 cm MAE still needs manual confirmation.",
        }
    if overall_mae >= 1.0:
        return {
            "gate": "stronger_production_candidate",
            "description": "Stronger production candidate: 1-3 cm MAE on key targets still needs real-world validation.",
        }
    return {
        "gate": "below_defined_gate",
        "description": "Below the current gate range; verify evaluation quality before interpreting.",
    }


def write_outputs(
    output_path: Path,
    dataset_summary: dict[str, Any],
    benchmark: dict[str, Any],
    per_target_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "dataset_validation_json": output_path / DATASET_VALIDATION_JSON,
        "dataset_validation_csv": output_path / DATASET_VALIDATION_CSV,
        "dataset_validation_md": output_path / DATASET_VALIDATION_MD,
        "benchmark_results_json": output_path / BENCHMARK_RESULTS_JSON,
        "benchmark_results_csv": output_path / BENCHMARK_RESULTS_CSV,
        "benchmark_results_md": output_path / BENCHMARK_RESULTS_MD,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "promotion_readiness_md": output_path / PROMOTION_READINESS_MD,
    }
    write_json(paths["dataset_validation_json"], dataset_summary)
    write_dataset_validation_csv(paths["dataset_validation_csv"], dataset_summary)
    paths["dataset_validation_md"].write_text(format_dataset_validation_markdown(dataset_summary), encoding="utf-8")
    write_json(paths["benchmark_results_json"], benchmark)
    write_csv(paths["benchmark_results_csv"], benchmark["benchmark_rows"], benchmark_fieldnames())
    write_csv(paths["per_target_results_csv"], per_target_rows, per_target_fieldnames())
    paths["benchmark_results_md"].write_text(format_benchmark_markdown(benchmark), encoding="utf-8")
    paths["promotion_readiness_md"].write_text(format_promotion_markdown(benchmark), encoding="utf-8")
    return paths


def write_cnn_history(output_path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    write_json(output_path / CNN_HISTORY_JSON, history)
    write_csv(output_path / CNN_HISTORY_CSV, history, sorted({key for row in history for key in row}))


def write_dataset_validation_csv(path: Path, summary: dict[str, Any]) -> None:
    validation = summary["validation"]
    manifest = summary["manifest"]
    audit = summary.get("audit") or {}
    row = {
        "dataset": summary["dataset"],
        "valid": validation.get("valid", False),
        "sample_count": validation.get("sample_count", 0),
        "front_image_count": validation.get("front_image_count", 0),
        "side_image_count": validation.get("side_image_count", 0),
        "label_row_count": validation.get("label_row_count", 0),
        "manifest_exists": manifest.get("exists", False),
        "manifest_rows": manifest.get("row_count", 0),
        "audit_warning_count": len(audit.get("warnings", [])) if isinstance(audit, dict) else "",
    }
    write_csv(path, [row], list(row))


def format_dataset_validation_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    manifest = summary["manifest"]
    audit = summary.get("audit") or {}
    lines = [
        "# Phase 3T Dataset Validation",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Valid: `{validation.get('valid')}`",
        f"Samples complete: {validation.get('sample_count', 0)}",
        f"Front PNGs: {validation.get('front_image_count', 0)}",
        f"Side PNGs: {validation.get('side_image_count', 0)}",
        f"Label rows: {validation.get('label_row_count', 0)}",
        f"Manifest rows: {manifest.get('row_count', 0)}",
    ]
    split_counts = manifest.get("split_counts", {})
    if split_counts:
        lines.extend(["", "## Manifest Split", ""])
        lines.extend(f"- {split}: {split_counts.get(split, 0)}" for split in ("train", "val", "test"))
    if isinstance(audit, dict):
        lines.extend(["", "## Variation Audit", ""])
        lines.append(f"Audit warnings: {len(audit.get('warnings', []))}")
    if validation.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in validation["errors"])
    return "\n".join(lines) + "\n"


def format_benchmark_markdown(benchmark: dict[str, Any]) -> str:
    lines = [
        "# Phase 3T Benchmark Results",
        "",
        "| Rank | Run | Source | Model | Feature Group | Test MAE | Gate |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for index, row in enumerate(benchmark["benchmark_rows"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row.get("run_name", "")),
                    str(row.get("source", "")),
                    str(row.get("model_type", "")),
                    str(row.get("feature_group", "")),
                    format_float(row.get("test_mae", "")),
                    str(row.get("promotion_gate", "")),
                ]
            )
            + " |"
        )
    if benchmark.get("best_model_per_target"):
        lines.extend(["", "## Best Per Target", ""])
        lines.append("| Target | Run | Model | Test MAE |")
        lines.append("| --- | --- | --- | ---: |")
        for target, row in benchmark["best_model_per_target"].items():
            lines.append(
                f"| {target} | {row.get('run_name', '')} | {row.get('model_type', '')} | {format_float(row.get('test_mae', ''))} |"
            )
    return "\n".join(lines) + "\n"


def format_promotion_markdown(benchmark: dict[str, Any]) -> str:
    best = benchmark.get("best_overall") or {}
    gate = benchmark.get("promotion") or {}
    lines = [
        "# Phase 3T Promotion Readiness",
        "",
        f"Best observed Phase 3T/report row: `{best.get('run_name', '')}`",
        f"Best test MAE: {format_float(best.get('test_mae', ''))}",
        f"Gate: `{gate.get('gate', '')}`",
        "",
        gate.get("description", ""),
        "",
        "Promotion gates:",
        "- Research only: >5 cm MAE.",
        "- Assisted sizing/manual confirmation: 3-5 cm MAE.",
        "- Stronger production candidate: 1-3 cm MAE on key targets.",
        "",
        "A 6-8 cm benchmark is useful for research comparison, but it is not production-ready for final tailoring measurements.",
    ]
    return "\n".join(lines) + "\n"


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_fieldnames() -> list[str]:
    return [
        "run_name",
        "source",
        "dataset",
        "feature_group",
        "model_type",
        "feature_count",
        "train_mae",
        "val_mae",
        "test_mae",
        "sample_count_test",
        "promotion_gate",
        "notes",
    ]


def per_target_fieldnames() -> list[str]:
    return ["run_name", "source", "dataset", "feature_group", "model_type", "target", "test_mae"]


def format_float(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


def build_manifest_if_valid(dataset: str | Path) -> dict[str, Any]:
    return build_dataset_manifest(dataset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 3T validation, benchmark, and promotion reports.")
    parser.add_argument("--dataset", required=True, help="Phase 3T dataset root.")
    parser.add_argument("--output", required=True, help="Output report directory.")
    parser.add_argument("--classical", help="Classical benchmark artifact directory with results.csv.")
    parser.add_argument("--cnn", help="CNN experiment artifact directory with metrics.json.")
    parser.add_argument("--audit", help="Variation audit artifact directory with summary.json.")
    args = parser.parse_args(argv)

    result = build_phase3t_benchmark_report(
        args.dataset,
        args.output,
        classical_dir=args.classical,
        cnn_dir=args.cnn,
        audit_dir=args.audit,
    )
    print(f"Dataset validation: {result['dataset_validation_json']}")
    print(f"Benchmark results: {result['benchmark_results_json']}")
    print(f"Promotion readiness: {result['promotion_readiness_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
