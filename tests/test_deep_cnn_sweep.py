import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.deep.dependencies import DeepLearningDependencyError
from training.deep.sweep_front_side_cnn import (
    generate_sweep_grid,
    rank_completed_runs,
    sweep_front_side_cnn,
)


def test_sweep_grid_generation_prioritizes_controlled_configs() -> None:
    grid = generate_sweep_grid()

    assert len(grid) == 16
    assert grid[0] == {"image_size": 128, "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0}
    assert grid[1] == {"image_size": 128, "learning_rate": 0.0005, "batch_size": 32, "weight_decay": 0.0}


def test_sweep_max_runs_and_dry_run(tmp_path) -> None:
    result = sweep_front_side_cnn(
        "data/synthetic/phase_2v",
        tmp_path / "sweep",
        max_runs=2,
        dry_run=True,
    )

    summary = result["summary"]
    assert summary["dry_run"] is True
    assert summary["planned_run_count"] == 2
    assert summary["completed_run_count"] == 0
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()


def test_sweep_summary_ranking_by_val_mae() -> None:
    ranked = rank_completed_runs(
        [
            {"run_name": "a", "status": "completed", "val_mae": 9.8},
            {"run_name": "b", "status": "completed", "val_mae": 9.1},
        ]
    )

    assert [row["run_name"] for row in ranked] == ["b", "a"]
    assert [row["rank"] for row in ranked] == [1, 2]


def test_sweep_records_failed_run(monkeypatch, tmp_path) -> None:
    def _raise_missing(*_args, **_kwargs):
        raise DeepLearningDependencyError("missing torch for sweep test")

    monkeypatch.setattr("training.deep.sweep_front_side_cnn.train_front_side_cnn", _raise_missing)

    result = sweep_front_side_cnn("dataset", tmp_path / "sweep", max_runs=2)

    summary = result["summary"]
    assert summary["failed_run_count"] == 1
    assert summary["completed_run_count"] == 0
    assert "missing torch" in summary["runs"][0]["error"]


def test_sweep_report_created_with_mocked_training(monkeypatch, tmp_path) -> None:
    def _fake_train(_dataset, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        val_mae = 9.0 + kwargs["learning_rate"]
        metrics = {
            "train": {"overall_mae": 8.0},
            "val": {"overall_mae": val_mae},
            "test": {"overall_mae": 9.5},
            "best_epoch": 1,
            "epochs_completed": 1,
            "early_stopping_triggered": False,
        }
        return {"metrics": metrics}

    monkeypatch.setattr("training.deep.sweep_front_side_cnn.train_front_side_cnn", _fake_train)

    result = sweep_front_side_cnn("dataset", tmp_path / "sweep", max_runs=2)

    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["summary"]["completed_run_count"] == 2
    assert result["summary"]["best_run_by_val_mae"]["val_mae"] == pytest.approx(9.0005)
    assert "Best run by validation MAE" in Path(result["report_path"]).read_text(encoding="utf-8")


def test_tiny_fixture_sweep_if_torch_available(tmp_path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed.")
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "deep" / "phase_3d_sweep_test"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.deep.sweep_front_side_cnn",
            "--dataset",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--patience",
            "1",
            "--max-runs",
            "1",
            "--device",
            "cpu",
            "--seed",
            "42",
            "--limit-samples",
            "8",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_run_count"] == 1
    assert summary["best_run_by_val_mae"]["status"] == "completed"
    assert (output_dir / summary["best_run_by_val_mae"]["run_name"] / "metrics.json").exists()


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_sweep_test"
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_dir = dataset_root / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    body_shapes = ["average", "athletic", "curvy", "broad"]
    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for index in range(1, count + 1):
            sample_id = f"sample_{index:06d}"
            front_width = 16 + (index % 8)
            side_width = 8 + (index % 5)
            _write_rect_image(front_dir / f"{sample_id}_front.png", rect=(20, 10, 20 + front_width, 54), size=(64, 64))
            _write_rect_image(side_dir / f"{sample_id}_side.png", rect=(24, 10, 24 + side_width, 54), size=(64, 64))
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(80 + front_width),
                    "waist_cm": str(70 + front_width),
                    "hip_cm": str(85 + front_width),
                    "shoulder_cm": str(38 + (index % 5)),
                    "inseam_cm": str(70 + (index % 8)),
                    "sleeve_cm": str(55 + (index % 7)),
                    "neck_cm": str(32 + (index % 4)),
                    "thigh_cm": str(45 + side_width),
                    "calf_cm": str(32 + (index % 6)),
                    "body_shape": body_shapes[index % len(body_shapes)],
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    result = build_dataset_manifest(dataset_root)
    assert result["valid"] is True
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (50, 50, 50))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)
