import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments import build_geometry_calibrated_labels as calibration
from training.experiments import validate_geometry_calibrated_labels as validator


def test_delta_calculations_are_deterministic() -> None:
    rows = [
        {
            "sample_id": "a",
            "is_ambiguous_phase_3z": False,
            "chest_calibration_delta_cm": 2.0,
            "chest_abs_calibration_delta_cm": 2.0,
            "waist_calibration_delta_cm": -4.0,
            "waist_abs_calibration_delta_cm": 4.0,
            "hip_calibration_delta_cm": 6.0,
            "hip_abs_calibration_delta_cm": 6.0,
            "thigh_calibration_delta_cm": -8.0,
            "thigh_abs_calibration_delta_cm": 8.0,
        },
        {
            "sample_id": "b",
            "is_ambiguous_phase_3z": True,
            "chest_calibration_delta_cm": 4.0,
            "chest_abs_calibration_delta_cm": 4.0,
            "waist_calibration_delta_cm": -2.0,
            "waist_abs_calibration_delta_cm": 2.0,
            "hip_calibration_delta_cm": 8.0,
            "hip_abs_calibration_delta_cm": 8.0,
            "thigh_calibration_delta_cm": -6.0,
            "thigh_abs_calibration_delta_cm": 6.0,
        },
    ]

    first = validator.build_delta_rows(rows)
    second = validator.build_delta_rows(rows)

    assert first == second
    chest = next(row for row in first if row["target"] == "chest_cm")
    assert chest["mean_abs_delta_cm"] == 3.0
    assert chest["ambiguous_minus_clean_abs_delta_cm"] == 2.0


def test_outlier_detection_and_schema_are_stable() -> None:
    proxy_names = ["chest_band_00_y40_front_norm_width"]
    proxy_matrix = np.asarray([[1.0], [2.0], [3.0]], dtype=float)
    calibrated = np.asarray([[70.0, 70.0, 90.0, 45.0], [80.0, 75.0, 95.0, 50.0], [200.0, 80.0, 100.0, 55.0]], dtype=float)
    original = calibrated.copy()

    rows = validator.build_validation_rows(proxy_names, proxy_matrix, calibrated, original)

    chest = next(row for row in rows if row["target"] == "chest_cm")
    assert set(validator.validation_fieldnames()) <= set(chest)
    assert chest["outlier_count"] == 1
    assert chest["low_bucket_count"] == 1
    assert chest["mid_bucket_count"] == 1
    assert chest["high_bucket_count"] == 1


def test_promotion_gate_summary_is_deterministic() -> None:
    benchmark_summary = {
        "best_run": {"run_name": "calibrated", "test_group_mae": 1.8},
        "best_per_target": [{"label_variant": "calibrated_labels", "target": "chest_cm", "test_mae": 1.5}],
    }
    leakage = {"risk_level": "high"}

    first = validator.build_promotion_gate_summary(benchmark_summary, leakage)
    second = validator.build_promotion_gate_summary(benchmark_summary, leakage)

    assert first == second
    assert first["synthetic_gate"] == "synthetic_calibrated_strong_candidate"
    assert first["real_world_gate"] == "requires_real_world_calibration_before_production"


def test_missing_optional_benchmark_artifact_warns(tmp_path: Path) -> None:
    payload, warning = validator.load_optional_json(tmp_path / "missing.json")

    assert payload == {}
    assert warning is not None


def test_validate_geometry_calibrated_labels_fixture_writes_outputs(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 30)
    phase4a_dir = tmp_path / "artifacts" / "phase_4a"
    calibration.build_geometry_calibrated_labels(
        dataset_root,
        phase4a_dir,
        model_types=["ridge"],
        ambiguity_scores=tmp_path / "missing_ambiguity.csv",
        phase3y_artifacts=tmp_path / "missing_phase3y",
    )
    output_dir = tmp_path / "artifacts" / "phase_4b"

    result = validator.validate_geometry_calibrated_labels(
        dataset_root,
        phase4a_dir,
        output_dir,
        model_types=["ridge"],
    )

    for key in (
        "validation_json",
        "validation_csv",
        "validation_md",
        "delta_summary_csv",
        "proxy_leakage_md",
        "promotion_gate_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["validation_json"]).read_text(encoding="utf-8"))
    assert summary["targets"] == validator.TARGETS
    assert summary["proxy_leakage"]["results"]
    with Path(result["validation_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["target"] for row in rows} == set(validator.TARGETS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_4b_fixture"
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
            front_width = 14 + (index % 9)
            side_width = 7 + (index % 6)
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
                    "shoulder_cm": str(40 + index / 3),
                    "inseam_cm": str(70 + index / 4),
                    "sleeve_cm": str(55 + index / 5),
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
