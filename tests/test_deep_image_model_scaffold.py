import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.deep.dependencies import DeepLearningDependencyError
from training.deep.synthetic_body_image_dataset import (
    TARGET_COLUMNS,
    SyntheticBodyImageDataset,
    load_normalized_image,
    target_vector,
)
from training.deep.train_front_side_cnn import (
    best_epoch_from_history,
    calculate_metrics_from_errors,
    calculate_per_target_errors,
    inverse_transform_targets,
    main,
    normalize_targets,
    should_stop_early,
    train_front_side_cnn,
)


def test_load_normalized_image_resizes_and_channels_first(tmp_path) -> None:
    image_path = tmp_path / "body.png"
    _write_rect_image(image_path, rect=(2, 2, 7, 8), size=(10, 12))

    array = load_normalized_image(image_path, image_size=16)

    assert array.shape == (3, 16, 16)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.min()) <= float(array.max()) <= 1.0


def test_deep_dataset_adapter_split_and_target_vector(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 20)

    dataset = SyntheticBodyImageDataset(dataset_root, split="train", image_size=32, as_tensors=False)
    sample = dataset[0]

    assert len(dataset) == 16
    assert sample["front_image"].shape == (3, 32, 32)
    assert sample["side_image"].shape == (3, 32, 32)
    assert sample["targets"].shape == (len(TARGET_COLUMNS),)
    assert sample["target_names"] == TARGET_COLUMNS
    assert sample["split"] == "train"


def test_target_vector_order_matches_existing_measurement_columns(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    sample = SyntheticBodyImageDataset(dataset_root, split="train", image_size=32)[0]
    base_sample = SyntheticBodyImageDataset(dataset_root, split="train", image_size=32).samples[0]

    expected = target_vector(base_sample, TARGET_COLUMNS)

    assert sample["targets"].tolist() == expected.tolist()
    assert TARGET_COLUMNS[:3] == ["height_cm", "weight_kg", "chest_cm"]


def test_model_output_shape_if_torch_available() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed.")

    code = (
        "import torch; "
        "from training.deep.models.simple_front_side_cnn import SimpleFrontSideCNN; "
        f"model = SimpleFrontSideCNN(target_count={len(TARGET_COLUMNS)}, embedding_dim=16); "
        "front = torch.zeros((2, 3, 32, 32), dtype=torch.float32); "
        "side = torch.zeros((2, 3, 32, 32), dtype=torch.float32); "
        "output = model(front, side); "
        "assert tuple(output.shape) == (2, 11)"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_cli_missing_dependency_behavior(monkeypatch, tmp_path, capsys) -> None:
    def _raise_missing() -> None:
        raise DeepLearningDependencyError("missing torch for test")

    monkeypatch.setattr("training.deep.train_front_side_cnn.import_torch", _raise_missing)

    exit_code = main(["--dataset", str(tmp_path), "--output", str(tmp_path / "artifacts")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "missing torch for test" in captured.err


def test_deep_per_target_metrics_are_calculated() -> None:
    errors = np.asarray([[1.0] * len(TARGET_COLUMNS), [3.0] * len(TARGET_COLUMNS)], dtype=np.float32)
    rows = [
        {f"abs_error_{target}": 1.0 for target in TARGET_COLUMNS},
        {f"abs_error_{target}": 3.0 for target in TARGET_COLUMNS},
    ]

    metrics = calculate_metrics_from_errors(errors)
    per_target = calculate_per_target_errors(rows)

    assert metrics["overall_mae"] == pytest.approx(2.0)
    assert metrics["mae_by_target"]["height_cm"] == pytest.approx(2.0)
    assert per_target["height_cm"]["count"] == 2
    assert per_target["height_cm"]["mae"] == pytest.approx(2.0)
    assert per_target["height_cm"]["max_abs_error"] == 3.0


def test_target_normalization_and_inverse_transform_are_consistent() -> None:
    targets = np.asarray([[10.0, 20.0], [14.0, 28.0]], dtype=np.float32)
    target_mean = np.asarray([12.0, 24.0], dtype=np.float32)
    target_std = np.asarray([2.0, 4.0], dtype=np.float32)

    normalized = normalize_targets(targets, target_mean, target_std)
    restored = inverse_transform_targets(normalized, target_mean, target_std)

    assert normalized.tolist() == [[-1.0, -1.0], [1.0, 1.0]]
    assert restored.tolist() == targets.tolist()


def test_early_stopping_helpers_select_best_epoch() -> None:
    history = [
        {"epoch": 1, "val_overall_mae": 9.5},
        {"epoch": 2, "val_overall_mae": 9.2},
        {"epoch": 3, "val_overall_mae": 9.3},
    ]

    assert best_epoch_from_history(history) == 2
    assert should_stop_early(epochs_without_improvement=2, patience=2) is True
    assert should_stop_early(epochs_without_improvement=2, patience=0) is False


def test_tiny_smoke_training_if_torch_available(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed.")
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "deep" / "phase_3a_smoke"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.deep.train_front_side_cnn",
            "--dataset",
            str(dataset_root),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--limit-samples",
            "8",
            "--image-size",
            "32",
            "--batch-size",
            "4",
            "--patience",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "per_target_errors.json").exists()
    assert (output_dir / "target_names.json").exists()
    assert (output_dir / "predictions_train.csv").exists()
    assert (output_dir / "predictions_val.csv").exists()
    assert (output_dir / "predictions_test.csv").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sample_counts"] == {"train": 8, "val": 2, "test": 2}
    assert "best_epoch" in metrics
    assert "best_val_overall_mae" in metrics
    assert "epochs_completed" in metrics
    assert "early_stopping_triggered" in metrics
    assert metrics["target_normalization_enabled"] is True
    assert "overall_mae" in metrics["val"]
    assert "overall_mae" in metrics["test"]
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["target_normalization"]["enabled"] is True
    assert config["patience"] == 2
    assert config["seed"] == 42
    assert config["best_epoch"] == metrics["best_epoch"]
    with (output_dir / "predictions_test.csv").open("r", newline="", encoding="utf-8") as predictions_file:
        rows = list(csv.DictReader(predictions_file))
    assert len(rows) == 2
    assert rows[0]["sample_id"].startswith("sample_")
    assert rows[0]["split"] == "test"
    assert f"true_{TARGET_COLUMNS[0]}" in rows[0]
    assert f"pred_{TARGET_COLUMNS[0]}" in rows[0]
    assert f"abs_error_{TARGET_COLUMNS[0]}" in rows[0]
    assert float(rows[0][f"true_{TARGET_COLUMNS[0]}"]) > 100.0


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_deep_test"
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
