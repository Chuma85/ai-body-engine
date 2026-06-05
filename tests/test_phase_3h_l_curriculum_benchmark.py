from __future__ import annotations

import inspect
from pathlib import Path
import subprocess

import pytest

from scripts import run_phase_3h_l_curriculum_benchmark as benchmark


def test_curriculum_benchmark_script_does_not_call_blender() -> None:
    source = inspect.getsource(benchmark).lower()

    assert "bpy" not in source
    assert "blender.exe" not in source
    assert "generate_blend_dataset" not in source
    assert "subprocess" not in source


def test_benchmark_rejects_archived_old_mannequin_manifest_paths(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for manifest_name in benchmark.REQUIRED_MANIFESTS:
        _write_manifest(
            manifest_dir / manifest_name,
            dataset_path="data/synthetic/_archived_old_mannequin/phase_3h_i_coupled_1000",
            image_path="missing.png",
        )

    with pytest.raises(ValueError, match="archived old mannequin"):
        benchmark.load_required_manifests(manifest_dir)


def test_missing_manifest_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required Phase 3H-K manifest"):
        benchmark.load_required_manifests(tmp_path)


def test_expected_result_schema_and_strategy_comparison() -> None:
    strategy_results = {
        "clean_only": _strategy_result(2.0, {"height_cm": 3.0, "chest_cm": 2.0, "waist_cm": 2.0, "hip_cm": 1.5, "shoulder_cm": 1.0, "inseam_cm": 2.5}),
        "mobile_realism_only": _strategy_result(1.8, {"height_cm": 2.7, "chest_cm": 1.8, "waist_cm": 1.9, "hip_cm": 1.4, "shoulder_cm": 0.9, "inseam_cm": 2.1}),
        "mixed_curriculum": _strategy_result(1.7, {"height_cm": 2.6, "chest_cm": 1.7, "waist_cm": 1.8, "hip_cm": 1.3, "shoulder_cm": 0.8, "inseam_cm": 2.0}),
    }

    comparison = benchmark.compare_strategies(strategy_results)

    assert comparison["best_strategy"] == "mixed_curriculum"
    assert comparison["mixed_improves_vs_clean"] is True
    assert comparison["mixed_improves_vs_mobile_only"] is True
    assert comparison["overall_deltas"]["mixed_minus_mobile"] == pytest.approx(-0.1)
    assert comparison["significantly_worse_targets_vs_mobile_only"] == []


def test_output_artifact_path_is_ignored_and_generated_artifacts_are_not_staged_by_default() -> None:
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


def _strategy_result(overall: float, target_mae: dict[str, float]) -> dict[str, object]:
    return {
        "best_model": "ridge",
        "overall_mean_mae": overall,
        "mae_by_target": target_mae,
    }


def _write_manifest(path: Path, *, dataset_path: str, image_path: str) -> None:
    path.write_text(
        ",".join(benchmark.MANIFEST_COLUMNS)
        + "\n"
        + ",".join(
            [
                "stage",
                "train",
                "source",
                dataset_path,
                "source:sample_000001",
                "sample_000001",
                image_path,
                image_path,
                image_path,
                "170",
                "95",
                "80",
                "98",
                "44",
                "78",
                "shape_key_coupled_synthetic",
                "test",
                "true",
                "false",
                "test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
