import json

import pytest

from training.experiments.analyze_render_ablation import (
    ablation_name,
    analyze_render_ablation,
    build_render_ablation_summary,
    effect_label,
)


def test_ablation_name_removes_phase_prefix_and_ridge_suffix() -> None:
    assert ablation_name("phase_3n_background_only_ridge") == "background_only"


def test_effect_label_classifies_delta() -> None:
    assert effect_label(-0.1) == "helped"
    assert effect_label(0.1) == "hurt"
    assert effect_label(0.0) == "matched"


def test_render_ablation_summary_ranks_best_and_worst() -> None:
    runs = [
        _run("phase_3n_clean_baseline_ridge", 5.0),
        _run("phase_3n_background_only_ridge", 4.5),
        _run("phase_3n_camera_jitter_only_ridge", 6.0),
    ]

    summary = build_render_ablation_summary(
        runs,
        clean_run_name="phase_3n_clean_baseline_ridge",
        label_equality_confirmed=True,
        sample_count=300,
    )

    assert summary["best_ablation"]["ablation"] == "background_only"
    assert summary["worst_ablation"]["ablation"] == "camera_jitter_only"
    assert summary["results"][1]["delta_vs_clean_test_mae"] == pytest.approx(-0.5)
    assert summary["results"][1]["effect_vs_clean"] == "helped"
    assert summary["label_equality_confirmed"] is True


def test_render_ablation_summary_requires_clean_baseline() -> None:
    with pytest.raises(ValueError, match="Clean baseline"):
        build_render_ablation_summary(
            [_run("phase_3n_background_only_ridge", 4.5)],
            clean_run_name="phase_3n_clean_baseline_ridge",
            label_equality_confirmed=True,
            sample_count=300,
        )


def test_analyze_render_ablation_writes_outputs(tmp_path) -> None:
    clean_dir = _write_run(tmp_path, "phase_3n_clean_baseline_ridge", 5.0)
    background_dir = _write_run(tmp_path, "phase_3n_background_only_ridge", 4.5)
    output_dir = tmp_path / "analysis"

    result = analyze_render_ablation(
        [clean_dir, background_dir],
        output_dir,
        label_equality_confirmed=True,
        sample_count=300,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "per_target_results.csv").exists()
    assert result["summary"]["best_ablation"]["ablation"] == "background_only"
    assert result["summary"]["per_target_results"][0]["target"] == "height_cm"


def test_analyze_render_ablation_missing_metrics_raises_clear_error(tmp_path) -> None:
    run_dir = tmp_path / "phase_3n_clean_baseline_ridge"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing metrics.json"):
        analyze_render_ablation([run_dir], tmp_path / "analysis")


def _run(name: str, test_mae: float) -> dict:
    return {
        "run_name": name,
        "run_dir": f"artifacts/experiments/{name}",
        "dataset": f"data/synthetic/{name.removesuffix('_ridge')}",
        "metrics": _metrics(test_mae),
        "config": {"dataset": f"data/synthetic/{name.removesuffix('_ridge')}"},
    }


def _write_run(tmp_path, name: str, test_mae: float):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(_metrics(test_mae)), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({"dataset": f"data/synthetic/{name}"}), encoding="utf-8")
    return run_dir


def _metrics(test_mae: float) -> dict:
    return {
        "sample_counts": {"train": 240, "val": 30, "test": 30},
        "target_columns": ["height_cm"],
        "train": {"overall_mae": test_mae - 0.2, "mae_by_target": {"height_cm": test_mae - 0.2}},
        "val": {"overall_mae": test_mae - 0.1, "mae_by_target": {"height_cm": test_mae - 0.1}},
        "test": {"overall_mae": test_mae, "mae_by_target": {"height_cm": test_mae}},
    }
