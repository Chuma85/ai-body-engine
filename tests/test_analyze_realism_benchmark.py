import csv
import json
from pathlib import Path

import pytest

from training.experiments.analyze_realism_benchmark import (
    analyze_realism_benchmark,
    build_realism_summary,
    load_metrics_run,
    per_target_comparison_rows,
)


def test_per_target_improvement_calculation(tmp_path) -> None:
    phase_2v = load_metrics_run(_write_run(tmp_path, "phase_2v", overall_test=10.0, chest_test=12.0, waist_test=8.0), "phase_2v_ridge")
    phase_3h = load_metrics_run(_write_run(tmp_path, "phase_3h_ridge", overall_test=6.0, chest_test=5.0, waist_test=9.0), "phase_3h_ridge")
    cnn = load_metrics_run(_write_run(tmp_path, "phase_3h_cnn", overall_test=7.0, chest_test=7.0, waist_test=10.0), "phase_3h_cnn")

    rows = per_target_comparison_rows(phase_2v, phase_3h, cnn, ["chest_cm", "waist_cm"])

    chest = next(row for row in rows if row["target"] == "chest_cm")
    waist = next(row for row in rows if row["target"] == "waist_cm")
    assert chest["ridge_improvement_mae"] == 7.0
    assert chest["ridge_improvement_percent"] == pytest.approx(58.333333)
    assert waist["ridge_improvement_mae"] == -1.0
    assert waist["ridge_improved"] is False


def test_ranks_biggest_improved_targets(tmp_path) -> None:
    summary = _summary(tmp_path)

    assert [row["target"] for row in summary["top_improved_targets"]] == ["chest_cm"]
    assert summary["regressed_targets_after_realism"][0]["target"] == "waist_cm"


def test_ridge_vs_cnn_gap_calculation(tmp_path) -> None:
    summary = _summary(tmp_path)

    chest = next(row for row in summary["per_target_comparison"] if row["target"] == "chest_cm")
    assert chest["cnn_gap_vs_ridge_mae"] == 2.0
    assert summary["overall_gap_phase_3h_cnn_vs_ridge"] == 1.0
    assert summary["cnn_underperformance"]["targets_trailing_ridge_count"] == 2
    assert summary["cnn_underperformance"]["interpretation"] == "global_underperformance"


def test_analysis_writes_summary_report_and_csv(tmp_path) -> None:
    phase_2v = _write_run(tmp_path, "phase_2v", overall_test=10.0, chest_test=12.0, waist_test=8.0)
    phase_3h = _write_run(tmp_path, "phase_3h_ridge", overall_test=6.0, chest_test=5.0, waist_test=9.0)
    cnn = _write_run(tmp_path, "phase_3h_cnn", overall_test=7.0, chest_test=7.0, waist_test=10.0)

    result = analyze_realism_benchmark(phase_2v, phase_3h, cnn, tmp_path / "analysis")

    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["per_target_path"]).exists()
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Phase 3J Realism Benchmark Analysis" in report
    assert "Largest CNN Gaps" in report
    with Path(result["per_target_path"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["target"] == "chest_cm"
    assert "ridge_improvement_mae" in rows[0]


def test_missing_metrics_file_gives_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing metrics.json"):
        load_metrics_run(missing, "phase_2v_ridge")


def _summary(tmp_path: Path) -> dict:
    phase_2v = load_metrics_run(_write_run(tmp_path, "phase_2v", overall_test=10.0, chest_test=12.0, waist_test=8.0), "phase_2v_ridge")
    phase_3h = load_metrics_run(_write_run(tmp_path, "phase_3h_ridge", overall_test=6.0, chest_test=5.0, waist_test=9.0), "phase_3h_ridge")
    cnn = load_metrics_run(_write_run(tmp_path, "phase_3h_cnn", overall_test=7.0, chest_test=7.0, waist_test=10.0), "phase_3h_cnn")
    return build_realism_summary(phase_2v, phase_3h, cnn)


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
        "sample_counts": {"train": 8, "val": 1, "test": 1},
        "target_columns": ["chest_cm", "waist_cm"],
        "train": {
            "overall_mae": overall_test - 1.0,
            "mae_by_target": {"chest_cm": chest_test - 1.0, "waist_cm": waist_test - 1.0},
        },
        "val": {
            "overall_mae": overall_test - 0.5,
            "mae_by_target": {"chest_cm": chest_test - 0.5, "waist_cm": waist_test - 0.5},
        },
        "test": {
            "overall_mae": overall_test,
            "mae_by_target": {"chest_cm": chest_test, "waist_cm": waist_test},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir
