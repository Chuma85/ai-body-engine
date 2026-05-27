import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.build_geometry_calibrated_labels import build_geometry_calibrated_labels
from training.measurements import geometry_measurement_estimator as estimator


def test_ellipse_estimator_proxy_works_on_fixture_widths() -> None:
    components = estimator.target_geometry_components(
        "chest_cm",
        _band_features_for_widths(0.25, 0.12),
        {"front_image_width_px": 100.0, "side_image_width_px": 100.0},
        scale_factor=2.0,
    )

    assert components["front_width_px"] == 25.0
    assert components["side_depth_px"] == 12.0
    assert components["ellipse_proxy"] > 0.0


def test_train_only_calibration_is_deterministic() -> None:
    features = np.asarray([[1.0, 2.0, 3.0, 0.1], [2.0, 3.0, 4.0, 0.2], [3.0, 4.0, 5.0, 0.3]], dtype=np.float64)
    targets = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)

    first = estimator.fit_affine_calibration(features, targets, ridge_alpha=0.1)
    second = estimator.fit_affine_calibration(features, targets, ridge_alpha=0.1)

    np.testing.assert_allclose(first["coefficients"], second["coefficients"])
    assert first["intercept"] == second["intercept"]


def test_estimator_output_schema_and_quality_flags_are_stable(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 30)
    phase4a_dir = tmp_path / "phase4a"
    build_geometry_calibrated_labels(
        dataset_root,
        phase4a_dir,
        model_types=["ridge"],
        ambiguity_scores=tmp_path / "missing.csv",
        phase3y_artifacts=tmp_path / "missing_phase3y",
    )

    result = estimator.run_geometry_measurement_estimator(
        dataset_root,
        tmp_path / "phase4c",
        calibrated_labels=phase4a_dir / "calibrated_labels.csv",
        phase4a_results=phase4a_dir / "calibrated_benchmark_results.json",
    )

    for key in (
        "estimator_results_json",
        "estimator_results_csv",
        "per_target_results_csv",
        "calibration_coefficients_json",
        "summary_md",
        "failure_cases_csv",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["estimator_results_json"]).read_text(encoding="utf-8"))
    assert summary["targets"] == estimator.TARGETS
    assert summary["quality_flag_counts"].get("ok", 0) == 30
    with Path(result["estimator_results_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {"sample_id", "quality_flags", "chest_cm_estimated_cm", "chest_cm_ellipse_proxy"} <= set(rows[0])


def test_missing_front_or_side_geometry_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="Could not read image file"):
        estimator.extract_estimator_components(
            [
                {
                    "sample_id": "sample_000001",
                    "dataset_split": "train",
                    "front_image_path": missing,
                    "side_image_path": missing,
                    "measurements": {"height_cm": 170.0},
                }
            ]
        )


def test_quality_flags_are_deterministic_for_invalid_components() -> None:
    rows = [
        {"quality_flags": "ok"},
        {"quality_flags": "missing_height;unstable_scale_factor"},
        {"quality_flags": "missing_height;unstable_scale_factor"},
    ]

    assert estimator.quality_flag_counts(rows) == {"missing_height": 2, "ok": 1, "unstable_scale_factor": 2}


def _band_features_for_widths(front_width: float, side_width: float) -> dict[str, float]:
    features = {}
    for target, centers in estimator.BAND_CANDIDATES.items():
        for index, center in enumerate(centers):
            prefix = f"{target.removesuffix('_cm')}_band_{index:02d}_y{int(round(center * 100)):02d}"
            features[f"{prefix}_front_raw_width_ratio"] = front_width
            features[f"{prefix}_side_raw_width_ratio"] = side_width
            features[f"{prefix}_raw_width_depth_product"] = front_width * side_width
            features[f"{prefix}_raw_front_side_width_ratio"] = front_width / side_width
    return features


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_4c_fixture"
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
            _write_rect_image(front_dir / f"{sample_id}_front.png", (24, 8, 24 + front_width, 86), (96, 96))
            _write_rect_image(side_dir / f"{sample_id}_side.png", (32, 8, 32 + side_width, 86), (96, 96))
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
