import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.build_geometry_calibrated_labels import build_geometry_calibrated_labels
from training.measurements import residual_correction as residuals


def test_residual_calculation_is_deterministic() -> None:
    geometry = np.asarray([[10.0, 20.0], [15.0, 25.0]], dtype=np.float64)
    labels = np.asarray([[12.0, 18.0], [14.0, 30.0]], dtype=np.float64)

    first = residuals.calculate_residuals(geometry, labels)
    second = residuals.calculate_residuals(geometry, labels)

    np.testing.assert_allclose(first, [[2.0, -2.0], [-1.0, 5.0]])
    np.testing.assert_allclose(first, second)


def test_final_estimate_is_geometry_plus_predicted_residual() -> None:
    geometry = np.asarray([[10.0, 20.0]], dtype=np.float64)
    predicted_residual = np.asarray([[1.5, -2.5]], dtype=np.float64)

    final = residuals.final_estimates(geometry, predicted_residual)

    np.testing.assert_allclose(final, [[11.5, 17.5]])


def test_split_indices_respect_manifest_splits() -> None:
    samples = [
        {"dataset_split": "train"},
        {"dataset_split": "val"},
        {"dataset_split": "test"},
        {"dataset_split": "train"},
    ]

    assert residuals.split_indices(samples) == {"train": [0, 3], "val": [1], "test": [2]}


def test_large_residual_and_out_of_range_flags_are_deterministic() -> None:
    flags = residuals.residual_quality_flags("chest_cm", "ok", predicted_residual=12.0, final_estimate=200.0)

    assert flags == "final_estimate_out_of_range;large_residual_correction"


def test_residual_correction_fixture_writes_outputs(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 30)
    phase4a_dir = tmp_path / "phase4a"
    build_geometry_calibrated_labels(
        dataset_root,
        phase4a_dir,
        model_types=["ridge"],
        ambiguity_scores=tmp_path / "missing.csv",
        phase3y_artifacts=tmp_path / "missing_phase3y",
    )

    result = residuals.run_residual_correction(
        dataset_root,
        tmp_path / "phase4d",
        calibrated_labels=phase4a_dir / "calibrated_labels.csv",
        phase4a_results=phase4a_dir / "calibrated_benchmark_results.json",
        model_types=["ridge"],
    )

    for key in (
        "residual_training_summary_json",
        "residual_training_summary_csv",
        "residual_benchmark_results_json",
        "residual_benchmark_results_csv",
        "per_target_results_csv",
        "residual_distribution_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["residual_benchmark_results_json"]).read_text(encoding="utf-8"))
    assert summary["targets"] == residuals.TARGETS
    assert summary["best_run"]["model_type"] == "ridge"
    with Path(result["residual_training_summary_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {"geometry_estimate_cm", "predicted_residual_cm", "final_estimate_cm", "confidence_flags"} <= set(rows[0])
    assert len(rows) == 30 * len(residuals.TARGETS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_4d_fixture"
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
