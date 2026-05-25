import json
from pathlib import Path

import pytest

from training.experiments.register_baseline_results import (
    build_registry_summary,
    load_baseline_run,
    register_baseline_results,
)


def test_registry_loads_metrics_and_config(tmp_path) -> None:
    run_dir = _write_run(tmp_path, "phase_2v", test_mae=9.4, feature_count=195)

    run = load_baseline_run(run_dir)

    assert run["run_name"] == "phase_2v"
    assert run["dataset"] == "data/synthetic/phase_2v"
    assert run["model_type"] == "ridge"
    assert run["feature_extractor_version"] == "phase_2p"
    assert run["metrics"]["test"]["overall_mae"] == 9.4


def test_registry_ranking_by_test_mae_and_current_best(tmp_path) -> None:
    run_a = load_baseline_run(_write_run(tmp_path, "phase_2v", test_mae=9.4))
    run_b = load_baseline_run(_write_run(tmp_path, "phase_2w", test_mae=9.5))
    run_c = load_baseline_run(_write_run(tmp_path, "phase_2y", test_mae=9.6))

    summary = build_registry_summary([run_b, run_c, run_a])

    assert [row["run_name"] for row in summary["ranked_runs"]] == ["phase_2v", "phase_2w", "phase_2y"]
    assert summary["current_best"]["run_name"] == "phase_2v"
    assert summary["ranked_runs"][0]["is_current_best"] is True


def test_registry_regression_delta_is_calculated(tmp_path) -> None:
    best = load_baseline_run(_write_run(tmp_path, "best", test_mae=8.0))
    candidate = load_baseline_run(_write_run(tmp_path, "candidate", test_mae=9.25))

    summary = build_registry_summary([candidate, best])
    candidate_row = next(row for row in summary["ranked_runs"] if row["run_name"] == "candidate")

    assert candidate_row["test_mae_delta_vs_best"] == pytest.approx(1.25)
    assert candidate_row["regression_vs_best"] is True


def test_registry_per_target_wins_are_counted(tmp_path) -> None:
    run_a = load_baseline_run(_write_run(tmp_path, "phase_2v", test_mae=9.4, chest_test=7.0, waist_test=12.0))
    run_b = load_baseline_run(_write_run(tmp_path, "phase_2w", test_mae=9.5, chest_test=8.0, waist_test=10.0))

    summary = build_registry_summary([run_a, run_b])

    assert summary["per_target_best"]["chest_cm"]["run_name"] == "phase_2v"
    assert summary["per_target_best"]["waist_cm"]["run_name"] == "phase_2w"
    assert summary["per_target_win_counts"] == {"phase_2v": 1, "phase_2w": 1}


def test_registry_writes_summary_json_and_report(tmp_path) -> None:
    run_a = _write_run(tmp_path, "phase_2v", test_mae=9.4)
    run_b = _write_run(tmp_path, "phase_2y", test_mae=9.6)
    output_dir = tmp_path / "registry"

    result = register_baseline_results([run_a, run_b], output_dir)

    summary_path = Path(result["summary_path"])
    report_path = Path(result["report_path"])
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert summary["current_best"]["run_name"] == "phase_2v"
    assert "# Baseline Registry" in report
    assert "Current best run" in report


def test_registry_missing_metrics_file_gives_clear_error(tmp_path) -> None:
    run_dir = tmp_path / "missing_metrics"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing metrics.json"):
        load_baseline_run(run_dir)


def _write_run(
    tmp_path: Path,
    run_name: str,
    *,
    test_mae: float,
    feature_count: int = 195,
    chest_test: float = 7.0,
    waist_test: float = 12.0,
) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    metrics = {
        "model_type": "image_silhouette_ridge_regressor",
        "model_family": "ridge",
        "feature_count": feature_count,
        "sample_counts": {"train": 8, "val": 1, "test": 1},
        "target_columns": ["chest_cm", "waist_cm"],
        "train": {
            "overall_mae": test_mae - 0.5,
            "mae_by_target": {"chest_cm": chest_test - 0.5, "waist_cm": waist_test - 0.5},
        },
        "val": {
            "overall_mae": test_mae + 0.25,
            "mae_by_target": {"chest_cm": chest_test + 0.25, "waist_cm": waist_test + 0.25},
        },
        "test": {
            "overall_mae": test_mae,
            "mae_by_target": {"chest_cm": chest_test, "waist_cm": waist_test},
        },
    }
    config = {
        "dataset": "data/synthetic/phase_2v",
        "feature_count": feature_count,
        "feature_extractor": {"name": "image_silhouette_features", "version": "phase_2p"},
        "model": {
            "type": "ridge",
            "artifact_type": "image_silhouette_ridge_regressor",
            "regression_method": "ridge_regression",
            "hyperparameters": {"ridge_alpha": 10.0},
        },
        "target_columns": ["chest_cm", "waist_cm"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_dir
