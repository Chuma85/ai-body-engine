from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from training.train_baseline_measurements import TARGET_COLUMNS

TARGET_GROUPS_JSON = "target_groups.json"
GROUPED_RESULTS_JSON = "grouped_benchmark_results.json"
GROUPED_RESULTS_CSV = "grouped_benchmark_results.csv"
PER_TARGET_RECOMMENDATIONS_CSV = "per_target_recommendations.csv"
PROMOTION_GATES_MD = "promotion_gates.md"

SILHOUETTE_GATES = {
    "research_only": "Research-only: group MAE is above 5 cm.",
    "assisted_manual_confirmation": "Assisted/manual-confirmation candidate: group MAE is 3-5 cm.",
    "stronger_candidate": "Stronger candidate: group MAE is 1-3 cm, pending real-world validation.",
}


def benchmark_target_groups(
    target_config: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_target_strategy(target_config)
    rows_by_run, per_target_rows_by_run, warnings = load_artifact_sources(config)
    candidate_runs = config["candidate_runs"]
    target_groups = config["target_groups"]

    grouped_rows: list[dict[str, Any]] = []
    for candidate in candidate_runs:
        run_name = candidate["run_name"]
        benchmark_row = rows_by_run.get(run_name, {})
        target_rows = per_target_rows_by_run.get(run_name, {})
        if not benchmark_row:
            warnings.append(f"Missing benchmark row for candidate run: {run_name}")
        if not target_rows:
            warnings.append(f"Missing per-target metrics for candidate run: {run_name}")
        for group_name, targets in target_groups.items():
            grouped_rows.append(
                build_grouped_row(
                    run_name,
                    candidate.get("description", ""),
                    benchmark_row,
                    target_rows,
                    group_name,
                    targets,
                    warnings,
                )
            )

    best_per_target = select_best_model_per_target(per_target_rows_by_run, [candidate["run_name"] for candidate in candidate_runs])
    recommendations = build_per_target_recommendations(best_per_target, config)
    summary = {
        "strategy_version": config["strategy_version"],
        "target_groups": target_groups,
        "candidate_runs": candidate_runs,
        "warnings": warnings,
        "grouped_results": sorted(grouped_rows, key=lambda row: (row["group_name"], row["group_mae"], row["run_name"])),
        "best_by_group": best_by_group(grouped_rows),
        "best_per_target": best_per_target,
        "per_target_recommendations": recommendations,
        "interpretation": interpretation_text(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(output_path, config, summary, recommendations)
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def load_target_strategy(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Target strategy config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = {"strategy_version", "target_groups", "candidate_runs", "artifact_sources"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Target strategy config is missing keys: {', '.join(missing)}")
    for group_name, targets in config["target_groups"].items():
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"Target group '{group_name}' must contain at least one target.")
    return config


def load_artifact_sources(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], list[str]]:
    warnings: list[str] = []
    rows_by_run: dict[str, dict[str, Any]] = {}
    per_target_rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    sources = config.get("artifact_sources", {})

    for csv_path in sources.get("benchmark_csvs", []):
        path = Path(csv_path)
        if not path.exists():
            warnings.append(f"Optional benchmark CSV is missing: {path}")
            continue
        for row in read_csv(path):
            run_name = row.get("run_name", "")
            if not run_name:
                continue
            rows_by_run.setdefault(run_name, normalize_benchmark_row(row, source=str(path)))

    for csv_path in sources.get("per_target_csvs", []):
        path = Path(csv_path)
        if not path.exists():
            warnings.append(f"Optional per-target CSV is missing: {path}")
            continue
        for row in read_csv(path):
            run_name = row.get("run_name", "")
            target = row.get("target", "")
            if not run_name or not target:
                continue
            per_target_rows_by_run.setdefault(run_name, {})[target] = normalize_per_target_row(row, source=str(path))

    for metric_source in sources.get("metrics_jsons", []):
        path = Path(metric_source["path"])
        run_name = metric_source["run_name"]
        if not path.exists():
            warnings.append(f"Optional metrics JSON is missing for {run_name}: {path}")
            continue
        metrics = read_json(path)
        rows_by_run[run_name] = benchmark_row_from_metrics(run_name, metrics, metric_source)
        per_target_rows_by_run[run_name] = per_target_rows_from_metrics(run_name, metrics, metric_source)

    return rows_by_run, per_target_rows_by_run, warnings


def normalize_benchmark_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "run_name": row.get("run_name", ""),
        "source": row.get("source", source),
        "dataset": row.get("dataset", ""),
        "feature_group": row.get("feature_group", ""),
        "model_type": row.get("model_type", ""),
        "train_mae": parse_optional_float(row.get("train_mae")),
        "val_mae": parse_optional_float(row.get("val_mae")),
        "test_mae": parse_optional_float(row.get("test_mae")),
        "sample_count_test": parse_optional_int(row.get("sample_count_test")),
        "notes": row.get("notes", ""),
    }


def normalize_per_target_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "run_name": row.get("run_name", ""),
        "source": row.get("source", source),
        "dataset": row.get("dataset", ""),
        "feature_group": row.get("feature_group", ""),
        "model_type": row.get("model_type", ""),
        "target": row["target"],
        "test_mae": float(row["test_mae"]),
    }


def benchmark_row_from_metrics(run_name: str, metrics: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "source": source.get("source", str(source.get("path", ""))),
        "dataset": source.get("dataset", ""),
        "feature_group": source.get("feature_group", ""),
        "model_type": source.get("model_type", metrics.get("model_type", "")),
        "train_mae": float(metrics["train"]["overall_mae"]),
        "val_mae": float(metrics["val"]["overall_mae"]),
        "test_mae": float(metrics["test"]["overall_mae"]),
        "sample_count_test": metrics.get("sample_counts", {}).get("test"),
        "notes": "Loaded from standard metrics.json.",
    }


def per_target_rows_from_metrics(run_name: str, metrics: dict[str, Any], source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        target: {
            "run_name": run_name,
            "source": source.get("source", str(source.get("path", ""))),
            "dataset": source.get("dataset", ""),
            "feature_group": source.get("feature_group", ""),
            "model_type": source.get("model_type", metrics.get("model_type", "")),
            "target": target,
            "test_mae": float(mae),
        }
        for target, mae in metrics.get("test", {}).get("mae_by_target", {}).items()
    }


def build_grouped_row(
    run_name: str,
    description: str,
    benchmark_row: dict[str, Any],
    target_rows: dict[str, dict[str, Any]],
    group_name: str,
    targets: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    available = [target_rows[target] for target in targets if target in target_rows]
    missing = [target for target in targets if target not in target_rows]
    if missing:
        warnings.append(f"{run_name} is missing per-target metrics for group {group_name}: {', '.join(missing)}")
    if available:
        group_mae = grouped_mae([row["test_mae"] for row in available])
    elif group_name == "all_targets" and benchmark_row.get("test_mae") is not None:
        group_mae = float(benchmark_row["test_mae"])
    else:
        group_mae = float("inf")
    gate = promotion_gate_for_group(group_name, group_mae)
    return {
        "run_name": run_name,
        "description": description,
        "dataset": benchmark_row.get("dataset", ""),
        "feature_group": benchmark_row.get("feature_group", ""),
        "model_type": benchmark_row.get("model_type", ""),
        "group_name": group_name,
        "target_count": len(targets),
        "available_target_count": len(available),
        "missing_targets": ";".join(missing),
        "group_mae": group_mae,
        "promotion_gate": gate["gate"],
        "promotion_note": gate["description"],
        "overall_test_mae": benchmark_row.get("test_mae"),
    }


def grouped_mae(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate grouped MAE without values.")
    return float(sum(float(value) for value in values) / len(values))


def promotion_gate_for_group(group_name: str, group_mae: float) -> dict[str, str]:
    if group_name == "manual_or_user_input":
        return {
            "gate": "manual_or_user_input_required",
            "description": "Use explicit user input or calibrated metadata; do not infer this group from current silhouettes.",
        }
    if group_name == "landmark_or_proportion_required":
        return {
            "gate": "landmark_or_manual_required",
            "description": "Requires a landmark/proportion strategy or manual confirmation regardless of current MAE.",
        }
    if group_name == "mass_proxy_uncertain":
        return {
            "gate": "coarse_proxy_only",
            "description": "Treat as a coarse body-shape proxy, not a final tailoring measurement.",
        }
    if group_name == "all_targets":
        return {
            "gate": "global_mae_mixes_target_types",
            "description": "Global MAE mixes learnable and weak-signal targets; use group-specific gates.",
        }
    if group_mae > 5.0:
        return {"gate": "research_only", "description": SILHOUETTE_GATES["research_only"]}
    if group_mae > 3.0:
        return {"gate": "assisted_manual_confirmation", "description": SILHOUETTE_GATES["assisted_manual_confirmation"]}
    if group_mae >= 1.0:
        return {"gate": "stronger_candidate", "description": SILHOUETTE_GATES["stronger_candidate"]}
    return {"gate": "below_defined_gate", "description": "Verify evaluation quality before interpreting."}


def select_best_model_per_target(
    per_target_rows_by_run: dict[str, dict[str, dict[str, Any]]],
    candidate_run_names: list[str],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        rows = []
        for run_name in candidate_run_names:
            row = per_target_rows_by_run.get(run_name, {}).get(target)
            if row is not None:
                rows.append(row)
        if rows:
            best[target] = min(rows, key=lambda row: (float(row["test_mae"]), row["run_name"]))
    return best


def best_by_group(grouped_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for group_name in sorted({row["group_name"] for row in grouped_rows}):
        rows = [row for row in grouped_rows if row["group_name"] == group_name and row["group_mae"] != float("inf")]
        if rows:
            best[group_name] = min(rows, key=lambda row: (row["group_mae"], row["run_name"]))
    return best


def build_per_target_recommendations(
    best_per_target: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    target_to_groups: dict[str, list[str]] = {}
    for group_name, targets in config["target_groups"].items():
        if group_name == "all_targets":
            continue
        for target in targets:
            target_to_groups.setdefault(target, []).append(group_name)
    rows = []
    for target in TARGET_COLUMNS:
        groups = target_to_groups.get(target, [])
        best = best_per_target.get(target, {})
        primary_group = primary_group_for_target(groups)
        gate = promotion_gate_for_group(primary_group, float(best["test_mae"])) if best else {"gate": "missing_metrics", "description": "No candidate metrics found."}
        rows.append(
            {
                "target": target,
                "target_groups": ";".join(groups),
                "best_run_name": best.get("run_name", ""),
                "best_model_type": best.get("model_type", ""),
                "best_feature_group": best.get("feature_group", ""),
                "best_test_mae": best.get("test_mae", ""),
                "promotion_gate": gate["gate"],
                "recommendation": recommendation_for_target(target, groups, gate["gate"]),
            }
        )
    return rows


def primary_group_for_target(groups: list[str]) -> str:
    for group_name in ("manual_or_user_input", "landmark_or_proportion_required", "mass_proxy_uncertain", "silhouette_learnable"):
        if group_name in groups:
            return group_name
    return "all_targets"


def recommendation_for_target(target: str, groups: list[str], gate: str) -> str:
    if "manual_or_user_input" in groups:
        return "Use explicit user input or calibrated metadata before finalizing this measurement."
    if "landmark_or_proportion_required" in groups:
        return "Requires landmark/proportion strategy or manual confirmation; do not trust as final silhouette-only prediction."
    if "mass_proxy_uncertain" in groups:
        return "Use only as a coarse fit proxy; not a direct tailoring measurement."
    if "silhouette_learnable" in groups:
        if gate == "research_only":
            return "Visually learnable, but current MAE remains research-only."
        if gate == "assisted_manual_confirmation":
            return "Candidate for assisted sizing with manual confirmation."
        return "Promising silhouette target; validate on real data before promotion."
    return "No target strategy assigned."


def write_outputs(
    output_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "target_groups_json": output_path / TARGET_GROUPS_JSON,
        "grouped_results_json": output_path / GROUPED_RESULTS_JSON,
        "grouped_results_csv": output_path / GROUPED_RESULTS_CSV,
        "per_target_recommendations_csv": output_path / PER_TARGET_RECOMMENDATIONS_CSV,
        "promotion_gates_md": output_path / PROMOTION_GATES_MD,
    }
    write_json(paths["target_groups_json"], config["target_groups"])
    write_json(paths["grouped_results_json"], summary)
    write_csv(paths["grouped_results_csv"], summary["grouped_results"], grouped_result_fieldnames())
    write_csv(paths["per_target_recommendations_csv"], recommendations, recommendation_fieldnames())
    paths["promotion_gates_md"].write_text(format_promotion_gates(summary), encoding="utf-8")
    return paths


def format_promotion_gates(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3V Target-Specific Promotion Gates",
        "",
        "Global MAE is not a promotion gate because it mixes silhouette-learnable targets with weak-signal targets.",
        "",
        "## Best By Group",
        "",
        "| Group | Best Run | Model | Group MAE | Gate |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for group_name, row in summary["best_by_group"].items():
        lines.append(
            f"| {group_name} | {row['run_name']} | {row.get('model_type', '')} | {row['group_mae']:.4f} | {row['promotion_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Target Recommendations",
            "",
            "| Target | Groups | Best Run | Best MAE | Gate | Recommendation |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in summary["per_target_recommendations"]:
        mae = row["best_test_mae"]
        mae_text = f"{float(mae):.4f}" if mae != "" else ""
        lines.append(
            f"| {row['target']} | {row['target_groups']} | {row['best_run_name']} | {mae_text} | {row['promotion_gate']} | {row['recommendation']} |"
        )
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def interpretation_text() -> list[str]:
    return [
        "Global MAE hides the fact that some targets are visually learnable and others need landmarks, proportions, or user input.",
        "Silhouette-learnable targets can be benchmarked as a separate group.",
        "Height, inseam, neck, and sleeve should not be trusted as final silhouette-only outputs until evidence improves.",
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


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


def parse_optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def parse_optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(float(value))


def grouped_result_fieldnames() -> list[str]:
    return [
        "run_name",
        "description",
        "dataset",
        "feature_group",
        "model_type",
        "group_name",
        "target_count",
        "available_target_count",
        "missing_targets",
        "group_mae",
        "promotion_gate",
        "promotion_note",
        "overall_test_mae",
    ]


def recommendation_fieldnames() -> list[str]:
    return [
        "target",
        "target_groups",
        "best_run_name",
        "best_model_type",
        "best_feature_group",
        "best_test_mae",
        "promotion_gate",
        "recommendation",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark measurement targets by target-specific strategy groups.")
    parser.add_argument("--target-config", required=True, help="Target strategy config JSON.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    args = parser.parse_args(argv)

    result = benchmark_target_groups(args.target_config, args.output)
    print(f"Target groups: {result['target_groups_json']}")
    print(f"Grouped results: {result['grouped_results_json']}")
    print(f"Promotion gates: {result['promotion_gates_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
