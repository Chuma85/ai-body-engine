import json
from pathlib import Path

import pytest

from training.analyze_baseline_errors import (
    analyze_baseline_errors,
    build_comparison_summary,
    format_markdown_report,
    load_metrics_run,
)


def test_load_metrics_file(tmp_path) -> None:
    run_dir = _write_run(tmp_path, "phase_2m", overall_test=10.0, chest_test=8.0, waist_test=12.0)

    run = load_metrics_run(run_dir)

    assert run["run_name"] == "phase_2m"
    assert run["metrics"]["test"]["overall_mae"] == 10.0


def test_compare_two_runs_and_best_per_target(tmp_path) -> None:
    run_a = load_metrics_run(_write_run(tmp_path, "phase_2m", overall_test=10.0, chest_test=8.0, waist_test=12.0))
    run_b = load_metrics_run(_write_run(tmp_path, "phase_2n", overall_test=7.0, chest_test=6.0, waist_test=14.0))

    summary = build_comparison_summary([run_a, run_b])

    assert summary["overall_mae"]["phase_2m"]["test"] == 10.0
    assert summary["overall_mae"]["phase_2n"]["test"] == 7.0
    assert summary["best_run_per_target"]["chest_cm"]["run_name"] == "phase_2n"
    assert summary["best_run_per_target"]["waist_cm"]["run_name"] == "phase_2m"


def test_calculates_improvement_and_regression(tmp_path) -> None:
    run_a = load_metrics_run(_write_run(tmp_path, "phase_2m", overall_test=10.0, chest_test=8.0, waist_test=12.0))
    run_b = load_metrics_run(_write_run(tmp_path, "phase_2n", overall_test=7.0, chest_test=6.0, waist_test=14.0))

    summary = build_comparison_summary([run_a, run_b])
    improvement = summary["pairwise_test_improvement"]

    assert improvement["overall_mae_delta"] == 3.0
    assert improvement["overall_percent_improvement"] == 30.0
    assert improvement["top_improved_targets"][0]["target"] == "chest_cm"
    assert improvement["top_regressed_targets"][0]["target"] == "waist_cm"


def test_writes_summary_json_and_markdown_report(tmp_path) -> None:
    run_a = _write_run(tmp_path, "phase_2m", overall_test=10.0, chest_test=8.0, waist_test=12.0)
    run_b = _write_run(tmp_path, "phase_2n", overall_test=7.0, chest_test=6.0, waist_test=14.0)
    output_dir = tmp_path / "analysis"

    result = analyze_baseline_errors([run_a, run_b], output_dir)

    summary_path = Path(result["summary_path"])
    report_path = Path(result["report_path"])
    assert summary_path.exists()
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert summary["run_names"] == ["phase_2m", "phase_2n"]
    assert "# Baseline Error Analysis" in report
    assert "Per-Target Test MAE" in report
    assert "Recommendations" in report


def test_markdown_report_contains_tables(tmp_path) -> None:
    run_a = load_metrics_run(_write_run(tmp_path, "phase_2m", overall_test=10.0, chest_test=8.0, waist_test=12.0))
    run_b = load_metrics_run(_write_run(tmp_path, "phase_2n", overall_test=7.0, chest_test=6.0, waist_test=14.0))

    report = format_markdown_report(build_comparison_summary([run_a, run_b]))

    assert "| Run | Train MAE | Val MAE | Test MAE |" in report
    assert "| Target | phase_2m | phase_2n | Best Run |" in report


def test_missing_metrics_raises_clear_error(tmp_path) -> None:
    missing_run = tmp_path / "missing_run"
    missing_run.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing metrics.json"):
        load_metrics_run(missing_run)


def _write_run(
    tmp_path: Path,
    run_name: str,
    *,
    overall_test: float,
    chest_test: float,
    waist_test: float,
) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    metrics = {
        "model_type": run_name,
        "sample_counts": {"train": 8, "val": 1, "test": 1},
        "target_columns": ["chest_cm", "waist_cm"],
        "train": {
            "overall_mae": overall_test + 1.0,
            "mae_by_target": {"chest_cm": chest_test + 1.0, "waist_cm": waist_test + 1.0},
        },
        "val": {
            "overall_mae": overall_test + 0.5,
            "mae_by_target": {"chest_cm": chest_test + 0.5, "waist_cm": waist_test + 0.5},
        },
        "test": {
            "overall_mae": overall_test,
            "mae_by_target": {"chest_cm": chest_test, "waist_cm": waist_test},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir
