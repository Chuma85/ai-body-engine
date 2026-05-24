import csv
from pathlib import Path
import zlib

import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.datasets.synthetic_body_dataset import SyntheticBodyDataset


def test_loads_all_samples(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    monkeypatch.chdir(tmp_path)

    dataset = SyntheticBodyDataset(dataset_root, split="all")

    assert len(dataset) == 20
    assert dataset.split_counts() == {"train": 16, "val": 2, "test": 2}


def test_loads_train_val_and_test_splits(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    monkeypatch.chdir(tmp_path)

    train_dataset = SyntheticBodyDataset(dataset_root, split="train")
    val_dataset = SyntheticBodyDataset(dataset_root, split="val")
    test_dataset = SyntheticBodyDataset(dataset_root, split="test")

    assert len(train_dataset) == 16
    assert len(val_dataset) == 2
    assert len(test_dataset) == 2
    assert {sample["dataset_split"] for sample in train_dataset} == {"train"}
    assert {sample["dataset_split"] for sample in val_dataset} == {"val"}
    assert {sample["dataset_split"] for sample in test_dataset} == {"test"}


def test_sample_fields_and_labels_are_present(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 5)
    monkeypatch.chdir(tmp_path)

    sample = SyntheticBodyDataset(dataset_root)[0]

    assert sample["sample_id"] == "sample_000001"
    assert sample["front_image_path"].exists()
    assert sample["side_image_path"].exists()
    assert sample["dataset_split"] in {"train", "val", "test"}
    assert sample["label_row_index"] == 0
    assert sample["labels"]["body_shape"] == "average"
    assert sample["measurements"]["height_cm"] == 171.0
    assert sample["measurements"]["weight_kg"] == 71.0


def test_load_images_is_optional(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 3)
    monkeypatch.chdir(tmp_path)

    path_only_sample = SyntheticBodyDataset(dataset_root, load_images=False)[0]
    image_sample = SyntheticBodyDataset(dataset_root, load_images=True)[0]

    assert "front_image_bytes" not in path_only_sample
    assert image_sample["front_image_bytes"].startswith(b"\x89PNG")
    assert image_sample["side_image_bytes"].startswith(b"\x89PNG")


def test_invalid_split_raises_clear_error(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 3)

    with pytest.raises(ValueError, match="Unknown split"):
        SyntheticBodyDataset(dataset_root, split="dev")


def test_missing_manifest_raises_clear_error(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 3)
    (dataset_root / "manifest.csv").unlink()

    with pytest.raises(FileNotFoundError, match="Missing manifest.csv"):
        SyntheticBodyDataset(dataset_root)


def test_missing_label_row_raises_clear_error(tmp_path) -> None:
    dataset_root = _write_dataset(tmp_path, 3)
    rows = list(csv.DictReader((dataset_root / "labels" / "labels.csv").open("r", newline="", encoding="utf-8")))
    with (dataset_root / "labels" / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="missing from labels.csv"):
        SyntheticBodyDataset(dataset_root)


def test_missing_image_path_raises_clear_error(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 3)
    monkeypatch.chdir(tmp_path)
    (dataset_root / "images" / "front" / "sample_000001_front.png").unlink()
    dataset = SyntheticBodyDataset(dataset_root)

    with pytest.raises(FileNotFoundError, match="Missing front image"):
        dataset[0]


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_test"
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_dir = dataset_root / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

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
                    "height_cm": str(170 + index),
                    "weight_kg": str(70 + index),
                    "chest_cm": "95.0",
                    "waist_cm": "80.0",
                    "hip_cm": "100.0",
                    "body_shape": "average",
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
