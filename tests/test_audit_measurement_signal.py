import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.audit_measurement_signal import (
    TARGET_COLUMNS,
    audit_measurement_signal,
    build_correlation_rows,
    find_ambiguous_pairs,
    write_contact_sheets,
)


def test_correlation_output_schema_is_stable() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    targets = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)

    rows = build_correlation_rows(features, targets, ["front_raw_bbox_width_px", "side_waist_width_ratio"], ["waist_cm"])

    assert {"target", "feature", "correlation", "abs_correlation", "feature_group", "view_group"} <= set(rows[0])
    assert rows[0]["target"] == "waist_cm"
    assert rows[0]["abs_correlation"] > 0.99


def test_ambiguous_pair_detection_finds_similar_features_with_different_labels() -> None:
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    features = np.asarray([[1.0, 1.0], [1.01, 1.0], [5.0, 5.0]], dtype=np.float64)
    targets = np.asarray([[10.0], [30.0], [11.0]], dtype=np.float64)

    rows = find_ambiguous_pairs(sample_ids, features, targets, ["waist_cm"], ambiguous_pairs_per_target=1)

    assert rows[0]["target"] == "waist_cm"
    assert {rows[0]["sample_id_a"], rows[0]["sample_id_b"]} == {"sample_a", "sample_b"}
    assert rows[0]["label_diff"] == 20.0


def test_contact_sheet_generation_handles_missing_images_gracefully(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": "sample_000001",
            "front_image_path": tmp_path / "missing_front.png",
            "side_image_path": tmp_path / "missing_side.png",
            "measurements": {"waist_cm": 70.0},
        }
    ]

    warnings = write_contact_sheets(samples, tmp_path / "sheets", ["waist_cm"], contact_sheet_count=1)

    assert warnings
    assert "Could not add sample" in warnings[0]


def test_audit_handles_missing_optional_predictions_with_warning(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 12)
    output_dir = tmp_path / "audit"

    result = audit_measurement_signal(
        dataset_root,
        output_dir,
        prediction_csvs=[tmp_path / "missing_predictions.csv"],
        ambiguous_pairs_per_target=2,
        contact_sheet_count=2,
    )

    for key in (
        "signal_json",
        "signal_csv",
        "signal_md",
        "ambiguous_pairs_csv",
        "per_target_error_analysis_csv",
        "visual_audit_summary_md",
    ):
        assert Path(result[key]).exists()
    payload = json.loads(Path(result["signal_json"]).read_text(encoding="utf-8"))
    assert payload["warnings"]
    assert set(payload["target_columns"]) == set(TARGET_COLUMNS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3u_fixture"
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
            front_width = 16 + index
            side_width = 8 + index // 2
            _write_rect_image(front_dir / f"{sample_id}_front.png", (24, 8, 24 + front_width, 58), (96, 96))
            _write_rect_image(side_dir / f"{sample_id}_side.png", (32, 8, 32 + side_width, 58), (96, 96))
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
                    "hip_cm": str(88 + front_width),
                    "shoulder_cm": str(40 + index / 2),
                    "inseam_cm": str(70 + index / 3),
                    "sleeve_cm": str(55 + index / 4),
                    "neck_cm": str(33 + index / 8),
                    "thigh_cm": str(45 + side_width),
                    "calf_cm": str(32 + side_width / 2),
                    "body_shape": "average",
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    manifest = build_dataset_manifest(dataset_root)
    assert manifest["valid"] is True
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (40, 40, 40))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)
