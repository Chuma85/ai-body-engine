import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments import build_geometry_calibrated_labels as calibration


def test_geometry_calibration_is_deterministic() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float64)
    labels = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    first = calibration.fit_geometry_calibration(features, labels)
    second = calibration.fit_geometry_calibration(features, labels)

    np.testing.assert_allclose(first["coefficients"], second["coefficients"])
    np.testing.assert_allclose(
        calibration.predict_geometry_calibration(first, features),
        calibration.predict_geometry_calibration(second, features),
    )


def test_ellipse_proxy_import_remains_available() -> None:
    assert calibration.ellipse_circumference_proxy(0.5, 0.25) > 0.0
    assert calibration.ellipse_circumference_proxy(0.0, 0.25) == 0.0


def test_calibrated_label_rows_preserve_sample_ids_and_splits() -> None:
    samples = [
        {"sample_id": "sample_000001", "dataset_split": "train"},
        {"sample_id": "sample_000002", "dataset_split": "test"},
    ]
    original = np.asarray([[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]], dtype=np.float64)
    calibrated = original + 1.0
    blended = original + 0.3

    rows = calibration.build_calibrated_label_rows(samples, original, calibrated, blended, {"sample_000002": True})

    assert [row["sample_id"] for row in rows] == ["sample_000001", "sample_000002"]
    assert [row["dataset_split"] for row in rows] == ["train", "test"]
    assert rows[1]["is_ambiguous_phase_3z"] is True
    assert rows[0]["original_chest_cm"] == 10.0
    assert rows[0]["calibrated_chest_cm"] == 11.0
    assert rows[0]["blended_chest_cm"] == 10.3


def test_optional_artifacts_warn_without_crashing(tmp_path: Path) -> None:
    warnings = calibration.validate_optional_artifacts(tmp_path / "missing")
    flags, ambiguity_warning = calibration.load_ambiguity_flags(tmp_path / "missing_scores.csv")

    assert warnings
    assert "recomputed proxies" in warnings[0]
    assert flags == {}
    assert ambiguity_warning is not None


def test_geometry_calibrated_fixture_writes_outputs(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 30)
    output_dir = tmp_path / "artifacts" / "phase_4a"

    result = calibration.build_geometry_calibrated_labels(
        dataset_root,
        output_dir,
        model_types=["ridge"],
        ambiguity_scores=tmp_path / "missing_ambiguity.csv",
        phase3y_artifacts=tmp_path / "missing_phase3y",
    )

    for key in (
        "calibrated_labels_csv",
        "label_delta_summary_csv",
        "label_delta_summary_md",
        "benchmark_results_json",
        "benchmark_results_csv",
        "per_target_results_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()

    with Path(result["calibrated_labels_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {"sample_id", "dataset_split", "original_chest_cm", "calibrated_chest_cm", "blended_chest_cm"} <= set(rows[0])
    assert len(rows) == 30

    summary = json.loads(Path(result["benchmark_results_json"]).read_text(encoding="utf-8"))
    assert summary["targets"] == calibration.TARGETS
    assert summary["label_variants"] == calibration.LABEL_VARIANTS
    assert {"run_name", "label_variant", "model_type", "test_group_mae"} <= set(summary["benchmark_results"][0])
    with Path(result["per_target_results_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        per_target_rows = list(csv.DictReader(csv_file))
    assert {row["target"] for row in per_target_rows} == set(calibration.TARGETS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_4a_fixture"
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
