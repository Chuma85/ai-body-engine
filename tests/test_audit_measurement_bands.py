import csv
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments import audit_measurement_bands as band_audit
from training.features.measurement_band_features import (
    candidate_band_definitions,
    extract_front_side_band_features,
    get_band_feature_names,
)


def test_band_feature_names_are_deterministic() -> None:
    first = get_band_feature_names()
    second = get_band_feature_names()

    assert first == second
    assert first[0] == "chest_band_00_y28_front_norm_width_ratio"
    assert "thigh_band_03_y80_raw_front_side_width_ratio" in first


def test_candidate_band_generation_is_stable() -> None:
    definitions = candidate_band_definitions()

    assert definitions[0] == {
        "target": "chest_cm",
        "band_index": 0,
        "band_name": "chest_band_00_y28",
        "center_y_ratio": 0.28,
    }
    assert definitions[-1]["band_name"] == "thigh_band_03_y80"


def test_band_correlations_are_computed_on_fixture_data() -> None:
    feature_names = ["chest_band_00_y28_front_norm_width_ratio", "chest_band_00_y28_side_norm_width_ratio"]
    matrix = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float64)
    targets = np.asarray([[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    rows = band_audit.build_band_correlation_rows(matrix, targets, feature_names)

    assert rows[0]["target"] == "chest_cm"
    assert rows[0]["correlation"] > 0.99
    assert rows[1]["correlation"] < -0.99


def test_band_benchmark_output_schema_is_stable(tmp_path: Path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 24)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(band_audit, "MODEL_TYPES", ["ridge"])
    monkeypatch.setattr(band_audit, "FEATURE_SETS", ["v6_bands"])

    result = band_audit.audit_measurement_bands(dataset_root, output_dir)

    for key in (
        "band_correlations_json",
        "band_correlations_csv",
        "band_correlations_md",
        "band_benchmark_json",
        "band_benchmark_csv",
        "per_target_band_results_csv",
        "worst_predictions_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()
    with Path(result["per_target_band_results_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["target"] for row in rows} == {"chest_cm", "waist_cm", "hip_cm", "thigh_cm"}


def test_missing_optional_images_and_predictions_warn_without_crash(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": "sample_000001",
            "front_image_path": tmp_path / "missing_front.png",
            "side_image_path": tmp_path / "missing_side.png",
            "measurements": {"chest_cm": 80.0, "waist_cm": 70.0, "hip_cm": 90.0, "thigh_cm": 50.0},
        }
    ]

    warnings = band_audit.write_measurement_band_contact_sheets(samples, tmp_path / "sheets", per_bucket=1)
    warnings.extend(band_audit.optional_prediction_warnings(tmp_path / "missing_predictions.csv"))

    assert warnings
    assert any("Could not add sample" in warning for warning in warnings)
    assert any("Optional prediction file is missing" in warning for warning in warnings)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3x_fixture"
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
            front_width = 16 + (index % 10)
            side_width = 8 + (index % 6)
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
