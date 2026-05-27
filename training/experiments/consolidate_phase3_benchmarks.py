from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_CSV = "leaderboard.csv"
LEADERBOARD_MD = "leaderboard.md"
PER_TARGET_CSV = "per_target_leaderboard.csv"
CANDIDATES_JSON = "candidate_baselines.json"

TARGET_COLUMNS = [
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

DEFAULT_STANDARD_EXPERIMENTS = [
    "experiments/phase_3l_clean_ridge",
    "experiments/phase_3l_realism_ridge",
]
DEFAULT_ANALYSIS_RESULTS = [
    "analysis/phase_3n_render_ablation/results.csv",
    "analysis/phase_3o_render_ablation/results.csv",
    "analysis/phase_3p_render_ablation/results.csv",
    "analysis/phase_3q_render_ablation/results.csv",
    "analysis/phase_3r_hybrid_feature_selection/results.csv",
]


def consolidate_phase3_benchmarks(
    artifacts_root: str | Path = "artifacts",
    output_dir: str | Path = "artifacts/phase_3s_benchmark_leaderboard",
) -> dict[str, Any]:
    artifacts_path = Path(artifacts_root)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []

    for relative_path in DEFAULT_STANDARD_EXPERIMENTS:
        experiment_path = artifacts_path / relative_path
        if not experiment_path.exists():
            warnings.append(f"Missing standard experiment artifact: {experiment_path}")
            continue
        row, target_rows = parse_standard_experiment(experiment_path)
        rows.append(row)
        per_target_rows.extend(target_rows)

    for relative_path in DEFAULT_ANALYSIS_RESULTS:
        results_path = artifacts_path / relative_path
        if not results_path.exists():
            warnings.append(f"Missing analysis results artifact: {results_path}")
            continue
        parsed_rows, target_rows, parse_warnings = parse_analysis_results(results_path, artifacts_path)
        rows.extend(parsed_rows)
        per_target_rows.extend(target_rows)
        warnings.extend(parse_warnings)

    rows = dedupe_rows(rows)
    rows = sorted(rows, key=lambda row: (float(row["test_mae"]), row["phase"], row["run_name"]))
    per_target_rows = sorted(
        per_target_rows,
        key=lambda row: (
            row.get("target", ""),
            float(row.get("test_mae", 0.0)),
            row.get("phase", ""),
            row.get("run_name", ""),
        ),
    )
    candidates = select_candidate_baselines(rows, per_target_rows)
    leaderboard = {
        "row_count": len(rows),
        "per_target_row_count": len(per_target_rows),
        "warnings": warnings,
        "leaderboard": rows,
        "rankings": build_rankings(rows, per_target_rows),
        "candidate_baselines": candidates,
        "interpretation": benchmark_interpretation(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaderboard_json_path": output_path / LEADERBOARD_JSON,
        "leaderboard_csv_path": output_path / LEADERBOARD_CSV,
        "leaderboard_md_path": output_path / LEADERBOARD_MD,
        "per_target_leaderboard_path": output_path / PER_TARGET_CSV,
        "candidate_baselines_path": output_path / CANDIDATES_JSON,
    }
    write_json(paths["leaderboard_json_path"], leaderboard)
    write_json(paths["candidate_baselines_path"], candidates)
    write_csv(paths["leaderboard_csv_path"], rows, leaderboard_fieldnames())
    write_csv(paths["per_target_leaderboard_path"], per_target_rows, per_target_fieldnames())
    paths["leaderboard_md_path"].write_text(format_leaderboard_markdown(leaderboard), encoding="utf-8")

    return {key: str(path) for key, path in paths.items()} | {"leaderboard": leaderboard}


def parse_standard_experiment(experiment_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = experiment_path / "metrics.json"
    config_path = experiment_path / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for experiment: {experiment_path}")
    metrics = read_json(metrics_path)
    config = read_json(config_path) if config_path.exists() else {}
    run_name = experiment_path.name
    phase = phase_from_name(run_name)
    dataset = str(config.get("dataset", ""))
    ablation = ablation_from_name(run_name, dataset)
    model_type = nested_get(config, ["model", "type"], metrics.get("model_family", metrics.get("model_type", "")))
    feature_version = nested_get(config, ["feature_extractor", "version"], "")
    row = {
        "phase": phase,
        "run_name": run_name,
        "dataset": dataset,
        "ablation": ablation,
        "feature_version": feature_version,
        "feature_group": "all_features",
        "model_type": model_type,
        "train_mae": float(metrics["train"]["overall_mae"]),
        "val_mae": float(metrics["val"]["overall_mae"]),
        "test_mae": float(metrics["test"]["overall_mae"]),
        "sample_count_test": int(metrics.get("sample_counts", {}).get("test", 0)),
        "notes": notes_for_row(phase, ablation, "all_features", model_type),
        "source": str(experiment_path),
    }
    target_rows = per_target_from_metrics(row, metrics)
    return row, target_rows


def parse_analysis_results(results_path: Path, artifacts_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with results_path.open("r", newline="", encoding="utf-8") as csv_file:
        for raw in csv.DictReader(csv_file):
            if "run_name" in raw:
                row = row_from_ablation_result(raw, results_path, artifacts_root)
            else:
                row = row_from_phase3r_result(raw, results_path)
            rows.append(row)

    per_target_path = results_path.with_name("per_target_results.csv")
    if per_target_path.exists():
        target_rows.extend(parse_per_target_results(per_target_path, rows))
    else:
        warnings.append(f"No per-target results found for {results_path}")
    return rows, target_rows, warnings


def row_from_ablation_result(raw: dict[str, str], results_path: Path, artifacts_root: Path) -> dict[str, Any]:
    run_name = raw["run_name"]
    phase = phase_from_name(run_name)
    experiment_path = artifacts_root / "experiments" / run_name
    config = read_json(experiment_path / "config.json") if (experiment_path / "config.json").exists() else {}
    feature_version = nested_get(config, ["feature_extractor", "version"], phase_default_feature_version(phase))
    model_type = nested_get(config, ["model", "type"], "ridge")
    ablation = strip_phase_prefix(raw.get("ablation") or run_name)
    feature_group = "all_features"
    return {
        "phase": phase,
        "run_name": run_name,
        "dataset": raw.get("dataset", nested_get(config, ["dataset"], "")),
        "ablation": ablation,
        "feature_version": feature_version,
        "feature_group": feature_group,
        "model_type": model_type,
        "train_mae": float(raw["train_mae"]),
        "val_mae": float(raw["val_mae"]),
        "test_mae": float(raw["test_mae"]),
        "sample_count_test": sample_count_hint(phase),
        "notes": notes_for_row(phase, ablation, feature_group, model_type),
        "source": str(results_path),
    }


def row_from_phase3r_result(raw: dict[str, str], results_path: Path) -> dict[str, Any]:
    phase = "3R"
    ablation = strip_phase_prefix(raw["dataset"])
    feature_group = raw["feature_config"]
    model_type = raw["model_type"]
    return {
        "phase": phase,
        "run_name": f"{raw['dataset']}__{feature_group}__{model_type}",
        "dataset": raw.get("dataset_path", raw["dataset"]),
        "ablation": ablation,
        "feature_version": "silhouette_geometry_v5_hybrid",
        "feature_group": feature_group,
        "model_type": model_type,
        "train_mae": float(raw["train_mae"]),
        "val_mae": float(raw["val_mae"]),
        "test_mae": float(raw["test_mae"]),
        "sample_count_test": 30,
        "notes": notes_for_row(phase, ablation, feature_group, model_type),
        "source": str(results_path),
    }


def parse_per_target_results(per_target_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["phase"], row["ablation"], row["feature_group"], row["model_type"]): row for row in rows}
    by_run_name = {row["run_name"]: row for row in rows}
    target_rows: list[dict[str, Any]] = []
    with per_target_path.open("r", newline="", encoding="utf-8") as csv_file:
        for raw in csv.DictReader(csv_file):
            if "run_name" in raw:
                base = by_run_name.get(raw["run_name"])
                if base is None:
                    continue
            else:
                key = (
                    "3R",
                    strip_phase_prefix(raw["dataset"]),
                    raw["feature_config"],
                    raw["model_type"],
                )
                base = lookup.get(key)
                if base is None:
                    continue
            target_rows.append(
                {
                    "phase": base["phase"],
                    "run_name": base["run_name"],
                    "dataset": base["dataset"],
                    "ablation": base["ablation"],
                    "feature_version": base["feature_version"],
                    "feature_group": base["feature_group"],
                    "model_type": base["model_type"],
                    "target": raw["target"],
                    "test_mae": float(raw["test_mae"]),
                    "source": str(per_target_path),
                }
            )
    return target_rows


def per_target_from_metrics(row: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    mae_by_target = metrics.get("test", {}).get("mae_by_target", {})
    return [
        {
            "phase": row["phase"],
            "run_name": row["run_name"],
            "dataset": row["dataset"],
            "ablation": row["ablation"],
            "feature_version": row["feature_version"],
            "feature_group": row["feature_group"],
            "model_type": row["model_type"],
            "target": target,
            "test_mae": float(mae),
            "source": row["source"],
        }
        for target, mae in mae_by_target.items()
    ]


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["phase"], row["run_name"], row["feature_group"], row["model_type"])
        deduped[key] = row
    return list(deduped.values())


def select_candidate_baselines(rows: list[dict[str, Any]], per_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select candidate baselines without leaderboard rows.")
    rows = sorted(rows, key=lambda row: (float(row["test_mae"]), row["phase"], row["run_name"]))
    best_overall = rows[0]
    current_best_clean = first_or_best(
        rows,
        predicate=lambda row: row["phase"] == "3L" and row["ablation"] == "clean",
        fallback=lambda row: "clean" in row["ablation"],
    )
    phase3r_regularized = first_or_best(
        rows,
        predicate=lambda row: row["phase"] == "3R",
        fallback=lambda row: row["phase"] == "3R",
    )
    camera_jitter = first_or_best(
        rows,
        predicate=lambda row: "camera_jitter" in row["ablation"],
        fallback=lambda row: "camera" in row["notes"],
    )
    combined_realism = first_or_best(
        rows,
        predicate=lambda row: "combined_realism" in row["ablation"],
        fallback=lambda row: "realism" in row["ablation"],
    )
    return {
        "best_overall": candidate_record(best_overall),
        "current_best_clean_baseline": candidate_record(current_best_clean),
        "best_phase_3r_regularized": candidate_record(phase3r_regularized),
        "best_camera_jitter_robust": candidate_record(camera_jitter),
        "combined_realism_candidate": candidate_record(combined_realism),
        "best_per_target": best_per_target_records(per_target_rows),
    }


def build_rankings(rows: list[dict[str, Any]], per_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_overall_mae": [candidate_record(row) for row in rows[:10]],
        "best_clean_data_model": candidate_record(first_or_best(rows, lambda row: "clean" in row["ablation"])),
        "best_realism_tolerant_model": candidate_record(
            first_or_best(rows, lambda row: "realism" in row["ablation"] or "background" in row["ablation"] or "lighting" in row["ablation"] or "skin_tone" in row["ablation"])
        ),
        "best_camera_jitter_model": candidate_record(first_or_best(rows, lambda row: "camera_jitter" in row["ablation"])),
        "best_per_target": best_per_target_records(per_target_rows),
    }


def best_per_target_records(per_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        candidates = [row for row in per_target_rows if row.get("target") == target]
        if candidates:
            best[target] = target_candidate_record(min(candidates, key=lambda row: float(row["test_mae"])))
    return best


def first_or_best(rows: list[dict[str, Any]], predicate: Any, fallback: Any | None = None) -> dict[str, Any]:
    matches = [row for row in rows if predicate(row)]
    if not matches and fallback is not None:
        matches = [row for row in rows if fallback(row)]
    if not matches:
        return rows[0]
    return min(matches, key=lambda row: float(row["test_mae"]))


def candidate_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": row["phase"],
        "run_name": row["run_name"],
        "dataset": row["dataset"],
        "ablation": row["ablation"],
        "feature_version": row["feature_version"],
        "feature_group": row["feature_group"],
        "model_type": row["model_type"],
        "test_mae": float(row["test_mae"]),
        "notes": row.get("notes", ""),
    }


def target_candidate_record(row: dict[str, Any]) -> dict[str, Any]:
    record = candidate_record(row)
    record["target"] = row["target"]
    return record


def benchmark_interpretation() -> dict[str, list[str]]:
    return {
        "consistently_helped": [
            "Same-body controls made comparisons trustworthy.",
            "Phase 3L clean 1000-sample ridge remains strongest overall.",
            "Phase 3R regularization/tree models recovered signal on 300-sample ablations.",
            "Raw scale cues helped several targets and small ablation datasets.",
        ],
        "consistently_hurt": [
            "Pure canonical normalization removed useful scale signal.",
            "Direct raw pixel area/offset cues remained unstable under camera jitter.",
            "Combined realism remained fragile without camera metadata normalization.",
        ],
        "next_phase_recommendation": [
            "Compare Phase 3L clean ridge, Phase 3R regularized raw-scale candidate, and Phase 3R camera-jitter candidate on the next controlled dataset.",
            "Add or expose render/camera metadata before scaling realism-heavy data.",
            "Treat hand-engineered features as near plateau unless metadata-normalized scale cues improve.",
        ],
    }


def format_leaderboard_markdown(payload: dict[str, Any]) -> str:
    rows = payload["leaderboard"]
    candidates = payload["candidate_baselines"]
    lines = [
        "# Phase 3 Benchmark Leaderboard",
        "",
        f"Rows collected: {payload['row_count']}",
        f"Per-target rows collected: {payload['per_target_row_count']}",
        "",
        "## Top Overall",
        "",
        markdown_table(
            ["Rank", "Phase", "Ablation", "Feature Version", "Feature Group", "Model", "Test MAE"],
            [
                [
                    str(index + 1),
                    row["phase"],
                    row["ablation"],
                    row["feature_version"],
                    row["feature_group"],
                    row["model_type"],
                    format_float(row["test_mae"]),
                ]
                for index, row in enumerate(rows[:15])
            ],
        ),
        "",
        "## Candidate Baselines",
        "",
        markdown_table(
            ["Candidate", "Phase", "Ablation", "Feature Group", "Model", "Test MAE"],
            [
                [key, value["phase"], value["ablation"], value["feature_group"], value["model_type"], format_float(value["test_mae"])]
                for key, value in candidates.items()
                if key != "best_per_target"
            ],
        ),
        "",
        "## Interpretation",
        "",
    ]
    for title, notes in payload["interpretation"].items():
        lines.extend([f"### {title.replace('_', ' ').title()}", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    if payload["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
        lines.append("")
    return "\n".join(lines)


def phase_from_name(name: str) -> str:
    match = re.search(r"phase_3([a-z]+)", name.lower())
    if not match:
        match = re.search(r"phase_2([a-z]+)", name.lower())
        if match:
            return f"2{match.group(1).upper()}"
        return "unknown"
    return f"3{match.group(1).upper()}"


def strip_phase_prefix(name: str) -> str:
    cleaned = re.sub(r"^phase_3[a-z]+_", "", name)
    cleaned = re.sub(r"^phase_2[a-z]+_", "", cleaned)
    cleaned = re.sub(r"_ridge$", "", cleaned)
    return cleaned


def ablation_from_name(run_name: str, dataset: str) -> str:
    if "phase_3l_clean" in dataset or "phase_3l_clean" in run_name:
        return "clean"
    if "phase_3l_realism" in dataset or "phase_3l_realism" in run_name:
        return "realism"
    return strip_phase_prefix(run_name)


def phase_default_feature_version(phase: str) -> str:
    return {
        "3N": "silhouette_geometry_v2",
        "3O": "silhouette_geometry_v3",
        "3P": "silhouette_geometry_v4",
        "3Q": "silhouette_geometry_v5_hybrid",
        "3R": "silhouette_geometry_v5_hybrid",
    }.get(phase, "")


def sample_count_hint(phase: str) -> int:
    return 100 if phase == "3L" else 30


def notes_for_row(phase: str, ablation: str, feature_group: str, model_type: str) -> str:
    notes = []
    if phase == "3L":
        notes.append("1000-sample same-body clean/realism benchmark")
    if phase in {"3N", "3O", "3P", "3Q", "3R"}:
        notes.append("300-sample same-body ablation")
    if "camera_jitter" in ablation:
        notes.append("camera/framing jitter focused")
    if "realism" in ablation or ablation in {"background_only", "lighting_only", "skin_tone_only"}:
        notes.append("render realism tolerant")
    if feature_group not in {"all_features", ""}:
        notes.append(f"feature config {feature_group}")
    if model_type not in {"ridge", "image_silhouette_ridge_regressor"}:
        notes.append(f"regularized/model variant {model_type}")
    return "; ".join(notes)


def nested_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


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


def leaderboard_fieldnames() -> list[str]:
    return [
        "phase",
        "run_name",
        "dataset",
        "ablation",
        "feature_version",
        "feature_group",
        "model_type",
        "train_mae",
        "val_mae",
        "test_mae",
        "sample_count_test",
        "notes",
        "source",
    ]


def per_target_fieldnames() -> list[str]:
    return [
        "phase",
        "run_name",
        "dataset",
        "ablation",
        "feature_version",
        "feature_group",
        "model_type",
        "target",
        "test_mae",
        "source",
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _header in headers) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate Phase 3 benchmark artifacts into a leaderboard.")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts root directory.")
    parser.add_argument("--output", default="artifacts/phase_3s_benchmark_leaderboard", help="Output directory.")
    args = parser.parse_args(argv)

    result = consolidate_phase3_benchmarks(args.artifacts, args.output)
    print(f"Leaderboard JSON: {result['leaderboard_json_path']}")
    print(f"Leaderboard CSV: {result['leaderboard_csv_path']}")
    print(f"Leaderboard Markdown: {result['leaderboard_md_path']}")
    best = result["leaderboard"]["candidate_baselines"]["best_overall"]
    print(f"Best overall: {best['phase']} {best['ablation']} {best['model_type']} test MAE {best['test_mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
