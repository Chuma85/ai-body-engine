import csv
from pathlib import Path
import zlib

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import MANIFEST_COLUMNS, build_dataset_manifest


def test_manifest_is_created_with_all_samples_once(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, 20)

    result = build_dataset_manifest(dataset)

    assert result["valid"] is True
    assert result["row_count"] == 20
    manifest_path = dataset / "manifest.csv"
    assert manifest_path.exists()

    rows = _read_manifest(manifest_path)
    sample_ids = [row["sample_id"] for row in rows]

    assert set(rows[0]) == set(MANIFEST_COLUMNS)
    assert len(sample_ids) == len(set(sample_ids)) == 20
    assert all(Path(row["front_image_path"]).exists() for row in rows)
    assert all(Path(row["side_image_path"]).exists() for row in rows)
    assert {row["dataset_split"] for row in rows} == {"train", "val", "test"}
    assert result["split_counts"] == {"train": 16, "val": 2, "test": 2}


def test_manifest_generation_is_deterministic(tmp_path) -> None:
    dataset = _write_dataset(tmp_path, 30)

    first_result = build_dataset_manifest(dataset)
    first_manifest = (dataset / "manifest.csv").read_text(encoding="utf-8")
    second_result = build_dataset_manifest(dataset)
    second_manifest = (dataset / "manifest.csv").read_text(encoding="utf-8")

    assert first_result["split_counts"] == {"train": 24, "val": 3, "test": 3}
    assert second_result["split_counts"] == first_result["split_counts"]
    assert second_manifest == first_manifest


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset = tmp_path / "phase_2k"
    front_dir = dataset / "images" / "front"
    side_dir = dataset / "images" / "side"
    labels_dir = dataset / "labels"
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
                    "height_cm": "170.0",
                    "weight_kg": "70.0",
                }
            )
            writer.writerow(row)

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
