import csv
from pathlib import Path
import zlib

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.validate_synthetic_dataset import validate_dataset


def test_valid_dataset_passes(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001", "sample_000002"])

    result = validate_dataset(dataset)

    assert result["valid"] is True
    assert result["sample_count"] == 2
    assert result["errors"] == []


def test_headerless_labels_csv_is_supported(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"], include_header=False)

    result = validate_dataset(dataset)

    assert result["valid"] is True
    assert result["label_row_count"] == 1
    assert any("no header" in warning for warning in result["warnings"])


def test_missing_labels_csv_fails(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"])
    (dataset / "labels" / "labels.csv").unlink()

    result = validate_dataset(dataset)

    assert result["valid"] is False
    assert str(dataset / "labels" / "labels.csv") in result["missing_paths"]


def test_missing_front_image_fails(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"])
    (dataset / "images" / "front" / "sample_000001_front.png").unlink()

    result = validate_dataset(dataset)

    assert result["valid"] is False
    assert result["unpaired_side_samples"] == ["sample_000001"]
    assert result["label_rows_missing_image_pairs"] == ["sample_000001"]


def test_missing_side_image_fails(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"])
    (dataset / "images" / "side" / "sample_000001_side.png").unlink()

    result = validate_dataset(dataset)

    assert result["valid"] is False
    assert result["unpaired_front_samples"] == ["sample_000001"]
    assert result["label_rows_missing_image_pairs"] == ["sample_000001"]


def test_labels_row_without_image_pair_fails(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"], extra_label_samples=["sample_000002"])

    result = validate_dataset(dataset)

    assert result["valid"] is False
    assert result["label_rows_missing_image_pairs"] == ["sample_000002"]


def test_corrupt_png_fails(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, ["sample_000001"])
    (dataset / "images" / "front" / "sample_000001_front.png").write_bytes(b"not a png")

    result = validate_dataset(dataset)

    assert result["valid"] is False
    assert any("sample_000001_front.png" in image for image in result["unreadable_images"])


def _write_dataset(
    tmp_path: Path,
    sample_ids: list[str],
    *,
    include_header: bool = True,
    extra_label_samples: list[str] | None = None,
) -> Path:
    dataset = tmp_path / "phase_2g"
    front_dir = dataset / "images" / "front"
    side_dir = dataset / "images" / "side"
    labels_dir = dataset / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for sample_id in sample_ids:
        _write_png(front_dir / f"{sample_id}_front.png")
        _write_png(side_dir / f"{sample_id}_side.png")

    label_samples = [*sample_ids, *(extra_label_samples or [])]
    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if include_header:
            writer.writerow(LABEL_COLUMNS)
        for sample_id in label_samples:
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": str(front_dir / f"{sample_id}_front.png"),
                    "side_image_path": str(side_dir / f"{sample_id}_side.png"),
                    "height_cm": "170.0",
                    "weight_kg": "70.0",
                }
            )
            writer.writerow([row[column] for column in LABEL_COLUMNS])

    return dataset


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
