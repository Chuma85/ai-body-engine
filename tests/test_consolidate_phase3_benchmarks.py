import csv
import json
from pathlib import Path

from training.experiments.consolidate_phase3_benchmarks import (
    consolidate_phase3_benchmarks,
    parse_standard_experiment,
    select_candidate_baselines,
)


def test_parse_standard_experiment_reads_metrics_and_config(tmp_path) -> None:
    experiment = tmp_path / "artifacts" / "experiments" / "phase_3l_clean_ridge"
    _write_standard_experiment(experiment, "data/synthetic/phase_3l_clean", 6.5)

    row, per_target_rows = parse_standard_experiment(experiment)

    assert row["phase"] == "3L"
    assert row["ablation"] == "clean"
    assert row["feature_version"] == "silhouette_geometry_v2"
    assert row["test_mae"] == 6.5
    assert per_target_rows[0]["target"] == "height_cm"


def test_consolidation_writes_outputs_and_candidates(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_standard_experiment(artifacts / "experiments" / "phase_3l_clean_ridge", "data/synthetic/phase_3l_clean", 6.5)
    _write_standard_experiment(artifacts / "experiments" / "phase_3l_realism_ridge", "data/synthetic/phase_3l_realism", 6.9)
    _write_analysis_results(
        artifacts / "analysis" / "phase_3r_hybrid_feature_selection",
        [
            {
                "dataset": "phase_3n_background_only",
                "dataset_path": "data/synthetic/phase_3n_background_only",
                "feature_config": "raw_scale_camera",
                "model_type": "elasticnet",
                "feature_count": "31",
                "train_mae": "6.8",
                "val_mae": "7.3",
                "test_mae": "7.1",
                "worst_target": "weight_kg",
                "worst_target_mae": "13.5",
            },
            {
                "dataset": "phase_3n_camera_jitter_only",
                "dataset_path": "data/synthetic/phase_3n_camera_jitter_only",
                "feature_config": "combined_hybrid_without_area_ratios",
                "model_type": "random_forest",
                "feature_count": "273",
                "train_mae": "4.5",
                "val_mae": "7.4",
                "test_mae": "7.2",
                "worst_target": "weight_kg",
                "worst_target_mae": "13.6",
            },
        ],
    )

    result = consolidate_phase3_benchmarks(artifacts, tmp_path / "leaderboard")

    assert Path(result["leaderboard_json_path"]).exists()
    assert Path(result["leaderboard_csv_path"]).exists()
    assert Path(result["leaderboard_md_path"]).exists()
    assert Path(result["per_target_leaderboard_path"]).exists()
    assert Path(result["candidate_baselines_path"]).exists()
    candidates = json.loads(Path(result["candidate_baselines_path"]).read_text(encoding="utf-8"))
    assert {
        "best_overall",
        "current_best_clean_baseline",
        "best_phase_3r_regularized",
        "best_camera_jitter_robust",
        "combined_realism_candidate",
    } <= set(candidates)
    assert candidates["current_best_clean_baseline"]["test_mae"] == 6.5
    assert result["leaderboard"]["warnings"]


def test_candidate_selection_is_deterministic() -> None:
    rows = [
        _row("3R", "phase_3n_background_only", "raw_scale_camera", "elasticnet", 7.1),
        _row("3L", "clean", "all_features", "ridge", 6.5),
        _row("3R", "phase_3n_camera_jitter_only", "combined_hybrid_without_area_ratios", "random_forest", 7.2),
    ]
    first = select_candidate_baselines(list(reversed(rows)), [])
    second = select_candidate_baselines(rows, [])

    assert first["best_overall"]["test_mae"] == second["best_overall"]["test_mae"] == 6.5
    assert first["best_camera_jitter_robust"]["model_type"] == "random_forest"
    assert second["best_phase_3r_regularized"]["model_type"] == "elasticnet"


def _write_standard_experiment(path: Path, dataset: str, test_mae: float) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "feature_extractor": {"version": "silhouette_geometry_v2"},
                "model": {"type": "ridge"},
            }
        ),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "sample_counts": {"train": 8, "val": 1, "test": 1},
                "train": {"overall_mae": 5.0, "mae_by_target": {"height_cm": 5.0}},
                "val": {"overall_mae": 6.0, "mae_by_target": {"height_cm": 6.0}},
                "test": {"overall_mae": test_mae, "mae_by_target": {"height_cm": test_mae}},
            }
        ),
        encoding="utf-8",
    )


def _write_analysis_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True)
    fieldnames = list(rows[0])
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (path / "per_target_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["dataset", "feature_config", "model_type", "target", "test_mae"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "feature_config": row["feature_config"],
                    "model_type": row["model_type"],
                    "target": "height_cm",
                    "test_mae": row["test_mae"],
                }
            )


def _row(phase: str, ablation: str, feature_group: str, model_type: str, test_mae: float) -> dict:
    return {
        "phase": phase,
        "run_name": f"{phase}_{ablation}_{model_type}",
        "dataset": ablation,
        "ablation": ablation,
        "feature_version": "test",
        "feature_group": feature_group,
        "model_type": model_type,
        "train_mae": test_mae,
        "val_mae": test_mae,
        "test_mae": test_mae,
        "sample_count_test": 1,
        "notes": "",
        "source": "test",
    }
