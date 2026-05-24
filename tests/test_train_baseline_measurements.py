import csv
import json
from pathlib import Path
import zlib

import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.datasets.synthetic_body_dataset import MEASUREMENT_COLUMNS
from training.train_baseline_measurements import (
    TARGET_COLUMNS,
    evaluate_regressor,
    main,
    train_baseline,
    train_body_shape_mean_regressor,
)


def test_target_columns_match_available_measurements() -> None:
    assert TARGET_COLUMNS == MEASUREMENT_COLUMNS
    assert "height_cm" in TARGET_COLUMNS
    assert "calf_cm" in TARGET_COLUMNS


def test_train_val_test_samples_load_and_train(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    monkeypatch.chdir(tmp_path)

    result = train_baseline(dataset_root, tmp_path / "artifacts" / "baseline")

    assert result["metrics"]["sample_counts"] == {"train": 16, "val": 2, "test": 2}
    assert result["metrics"]["model_type"] == "body_shape_mean_regressor"


def test_metric_calculation_reports_mae() -> None:
    samples = [
        _sample("average", height_cm=170.0, weight_kg=70.0),
        _sample("average", height_cm=174.0, weight_kg=74.0),
    ]
    model = train_body_shape_mean_regressor(samples, ["height_cm", "weight_kg"])

    metrics = evaluate_regressor(model, samples, ["height_cm", "weight_kg"])

    assert metrics["mae_by_target"]["height_cm"] == 2.0
    assert metrics["mae_by_target"]["weight_kg"] == 2.0
    assert metrics["overall_mae"] == 2.0


def test_training_command_creates_metrics_file(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "phase_2m"
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dataset", str(dataset_root), "--output", str(output_dir)])

    assert exit_code == 0
    metrics_path = output_dir / "metrics.json"
    model_path = output_dir / "model.json"
    assert metrics_path.exists()
    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sample_counts"] == {"train": 16, "val": 2, "test": 2}
    assert "overall_mae" in metrics["val"]


def test_not_enough_samples_raises_helpful_error(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 2)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Not enough samples"):
        train_baseline(dataset_root, tmp_path / "artifacts" / "too_small")


def _sample(body_shape: str, **measurements: float) -> dict:
    return {
        "body_shape": body_shape,
        "measurements": measurements,
    }


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_test"
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
            _write_png(front_dir / f"{sample_id}_front.png")
            _write_png(side_dir / f"{sample_id}_side.png")
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(80 + index),
                    "waist_cm": str(70 + index),
                    "hip_cm": str(85 + index),
                    "shoulder_cm": str(40 + (index % 5)),
                    "inseam_cm": str(70 + (index % 8)),
                    "sleeve_cm": str(55 + (index % 7)),
                    "neck_cm": str(32 + (index % 4)),
                    "thigh_cm": str(45 + (index % 9)),
                    "calf_cm": str(32 + (index % 6)),
                    "body_shape": body_shapes[index % len(body_shapes)],
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    result = build_dataset_manifest(dataset_root)
    assert result["valid"] is True
    return dataset_root


def _write_png(path: Path) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")

    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    raw_scanline = b"\x00\x00\x00\x00\xff"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )
