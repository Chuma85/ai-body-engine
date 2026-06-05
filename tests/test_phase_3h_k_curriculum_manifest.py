from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import zlib

import pytest

from scripts.build_phase_3h_k_curriculum_manifest import (
    DEFAULT_OUTPUT,
    MANIFEST_COLUMNS,
    build_curriculum_manifests,
    ensure_not_archived_dataset,
)
from synthetic.blender.blend_dataset import BLEND_LABEL_COLUMNS, LABEL_GENERATION_MODE, PHASE_3H_J_LABEL_FORMULA_VERSION


def test_curriculum_manifest_rejects_archived_old_mannequin_paths() -> None:
    ensure_not_archived_dataset("data/synthetic/phase_3h_i_coupled_1000")

    with pytest.raises(ValueError, match="archived old mannequin"):
        ensure_not_archived_dataset("data/synthetic/_archived_old_mannequin/phase_3h_i_coupled_1000")


def test_curriculum_manifest_outputs_reference_existing_images_without_copying_pngs(tmp_path: Path) -> None:
    clean_dataset = _write_blend_dataset(tmp_path / "phase_3h_i_coupled_1000", "shape_key_coupled_synthetic_v2", 10)
    mobile_dataset = _write_blend_dataset(
        tmp_path / "phase_3h_j_mobile_realism_1000",
        "shape_keys_safe_range_plus_mobile_realism",
        10,
    )
    output = tmp_path / "manifest_out"

    summary = build_curriculum_manifests(
        clean_dataset=clean_dataset,
        mobile_realism_dataset=mobile_dataset,
        output_dir=output,
        expected_samples=10,
    )

    assert summary["row_counts"] == {
        "clean_train": 10,
        "mobile_realism_train": 8,
        "mixed_curriculum": 18,
        "evaluation": 2,
    }
    assert summary["copies_images"] is False
    assert list(output.rglob("*.png")) == []

    clean_rows = _read_manifest(output / "clean_train_manifest.csv")
    mobile_rows = _read_manifest(output / "mobile_realism_train_manifest.csv")
    mixed_rows = _read_manifest(output / "mixed_curriculum_manifest.csv")
    evaluation_rows = _read_manifest(output / "evaluation_manifest.csv")

    assert set(clean_rows[0]) == set(MANIFEST_COLUMNS)
    assert set(mobile_rows[0]) == set(MANIFEST_COLUMNS)
    assert {row["dataset_source"] for row in clean_rows} == {"phase_3h_i_clean"}
    assert {row["dataset_source"] for row in mobile_rows} == {"phase_3h_j_mobile_realism"}
    assert {row["dataset_source"] for row in mixed_rows} == {"phase_3h_i_clean", "phase_3h_j_mobile_realism"}
    assert {row["curriculum_split"] for row in evaluation_rows} == {"evaluation"}
    assert {row["sample_id"] for row in evaluation_rows} == {"sample_000005", "sample_000010"}
    assert all(Path(row["front_image"]).exists() for row in clean_rows)
    assert all(Path(row["side_image"]).exists() for row in clean_rows)
    assert all(Path(row["back_image"]).exists() for row in clean_rows)

    summary_json = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["datasets"]["clean"]["labels"] == 10
    assert summary_json["datasets"]["mobile_realism"]["pngs"] == 30
    assert summary_json["real_world_validated"] is False


def test_default_manifest_output_is_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", f"{DEFAULT_OUTPUT}/summary.json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"{DEFAULT_OUTPUT}/summary.json" in result.stdout


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _write_blend_dataset(dataset: Path, variation_source: str, count: int) -> Path:
    for view in ("front", "side", "back"):
        (dataset / "images" / view).mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, count + 1):
        sample_id = f"sample_{index:06d}"
        for view in ("front", "side", "back"):
            _write_png(dataset / "images" / view / f"{sample_id}_{view}.png")
        row = {column: "" for column in BLEND_LABEL_COLUMNS}
        row.update(
            {
                "sample_id": sample_id,
                "front_image": f"images/front/{sample_id}_front.png",
                "side_image": f"images/side/{sample_id}_side.png",
                "back_image": f"images/back/{sample_id}_back.png",
                "height_cm": "170.0",
                "chest_cm": "95.0",
                "waist_cm": "82.0",
                "hip_cm": "98.0",
                "shoulder_cm": "44.0",
                "inseam_cm": "78.0",
                "variation_source": variation_source,
                "synthetic_labels": "true",
                "real_world_validated": "false",
                "label_generation_mode": LABEL_GENERATION_MODE,
            }
        )
        rows.append(row)
    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (dataset / "metadata.json").write_text(
        json.dumps(
            {
                "sample_count": count,
                "variation_source": variation_source,
                "synthetic_labels": True,
                "real_world_validated": False,
                "label_generation_mode": LABEL_GENERATION_MODE,
                "label_formula_version": PHASE_3H_J_LABEL_FORMULA_VERSION,
            }
        ),
        encoding="utf-8",
    )
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
