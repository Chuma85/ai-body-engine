from __future__ import annotations

import inspect
from pathlib import Path
import subprocess

import pytest

from scripts import run_phase_3h_m_view_ablation_benchmark as benchmark


def test_view_ablation_script_does_not_call_blender() -> None:
    source = inspect.getsource(benchmark).lower()

    assert "bpy" not in source
    assert "blender.exe" not in source
    assert "generate_blend_dataset" not in source
    assert "subprocess" not in source


def test_benchmark_rejects_archived_old_mannequin_dataset_paths() -> None:
    with pytest.raises(ValueError, match="archived old mannequin"):
        benchmark.ensure_not_archived_dataset(
            "data/synthetic/_archived_old_mannequin/phase_3h_j_mobile_realism_1000"
        )


def test_required_view_combinations_are_declared() -> None:
    assert benchmark.VIEW_COMBINATIONS == {
        "front": ("front",),
        "side": ("side",),
        "back": ("back",),
        "front_side": ("front", "side"),
        "front_back": ("front", "back"),
        "side_back": ("side", "back"),
        "front_side_back": ("front", "side", "back"),
    }


def test_compare_view_combinations_reports_schema_for_all_combinations() -> None:
    results = {
        "front": _combo_result(2.3),
        "side": _combo_result(2.1),
        "back": _combo_result(2.4),
        "front_side": _combo_result(1.9),
        "front_back": _combo_result(2.0),
        "side_back": _combo_result(1.8),
        "front_side_back": _combo_result(1.7),
    }

    comparison = benchmark.compare_view_combinations(results)

    assert comparison["best_overall_combination"] == "front_side_back"
    assert {row["view_combination"] for row in comparison["ranked_view_combinations"]} == set(
        benchmark.VIEW_COMBINATIONS
    )
    assert set(comparison["best_by_target"]) == set(benchmark.TARGET_COLUMNS)
    assert comparison["adding_back_to_front_side_delta"] == pytest.approx(-0.2)
    assert comparison["back_view_improves_front_side"] is True
    assert comparison["targets_improved_by_adding_back"] == list(benchmark.TARGET_COLUMNS)
    assert comparison["targets_worsened_by_adding_back"] == []
    assert comparison["back_only_rank"] == 7


def test_missing_view_folder_fails_clearly(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "front").mkdir(parents=True)
    (dataset / "images" / "side").mkdir(parents=True)
    (dataset / "labels.csv").write_text("sample_id,front_image,side_image,back_image\n", encoding="utf-8")
    (dataset / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="images.*back"):
        benchmark.validate_dataset(dataset, expected_samples=1)


def test_output_artifact_path_is_ignored_and_generated_outputs_are_not_staged_by_default() -> None:
    paths = [
        f"{benchmark.DEFAULT_OUTPUT}/metrics.json",
        f"{benchmark.DEFAULT_OUTPUT}/comparison.csv",
        f"{benchmark.DEFAULT_OUTPUT}/summary.json",
    ]
    ignored = subprocess.run(["git", "check-ignore", *paths], check=False, capture_output=True, text=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], check=False, capture_output=True, text=True)

    assert ignored.returncode == 0
    for path in paths:
        assert path in ignored.stdout
        assert path not in staged.stdout


def _combo_result(overall: float) -> dict[str, object]:
    return {
        "best_model": "ridge",
        "overall_mean_mae": overall,
        "mae_by_target": {
            target: overall + index / 10.0
            for index, target in enumerate(benchmark.TARGET_COLUMNS)
        },
    }
