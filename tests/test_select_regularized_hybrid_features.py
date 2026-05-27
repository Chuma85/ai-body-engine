import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.select_regularized_hybrid_features import (
    is_area_ratio_feature,
    is_offset_feature,
    load_drift_scores,
    run_hybrid_feature_selection_benchmark,
    select_feature_names,
)
from training.features.image_silhouette_features import get_feature_names


def test_feature_group_filtering_selects_expected_groups() -> None:
    names = [
        "front_bbox_width_ratio",
        "front_raw_bbox_height_px",
        "front_crop_offset_x",
        "front_side_torso_volume_proxy",
        "front_to_side_area_ratio",
    ]

    assert select_feature_names(names, "normalized_shape") == ["front_bbox_width_ratio"]
    assert select_feature_names(names, "raw_scale_camera") == [
        "front_raw_bbox_height_px",
        "front_crop_offset_x",
        "front_to_side_area_ratio",
    ]
    assert select_feature_names(names, "combined_hybrid_without_offsets") == [
        "front_bbox_width_ratio",
        "front_raw_bbox_height_px",
        "front_side_torso_volume_proxy",
        "front_to_side_area_ratio",
    ]
    assert is_offset_feature("front_crop_offset_x")
    assert is_area_ratio_feature("front_to_side_area_ratio")


def test_selected_low_drift_features_are_deterministic() -> None:
    names = ["front_bbox_width_ratio", "front_raw_bbox_height_px", "front_crop_offset_x"]
    drift_scores = {
        "front_bbox_width_ratio": 0.1,
        "front_raw_bbox_height_px": 4.0,
        "front_crop_offset_x": 0.1,
    }

    first = select_feature_names(names, "selected_low_drift_features", drift_scores=drift_scores, drift_threshold=1.0)
    second = select_feature_names(list(reversed(names)), "selected_low_drift_features", drift_scores=drift_scores, drift_threshold=1.0)

    assert first == ["front_bbox_width_ratio"]
    assert second == ["front_bbox_width_ratio"]


def test_drift_scores_load_maximum_per_feature(tmp_path) -> None:
    drift_csv = tmp_path / "feature_drift.csv"
    drift_csv.write_text(
        "ablation,feature,mean_abs_drift\n"
        "a,front_raw_bbox_height_px,2.0\n"
        "b,front_raw_bbox_height_px,5.0\n",
        encoding="utf-8",
    )

    assert load_drift_scores(drift_csv)["front_raw_bbox_height_px"] == 5.0


def test_hybrid_selection_benchmark_writes_outputs(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 24)
    drift_csv = tmp_path / "feature_drift.csv"
    drift_csv.write_text(
        "ablation,feature,mean_abs_drift\n"
        "camera_jitter_only,front_raw_bbox_height_px,5.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts" / "phase_3r"
    monkeypatch.chdir(tmp_path)

    result = run_hybrid_feature_selection_benchmark(
        [dataset_root],
        output_dir,
        feature_configs=["normalized_shape", "combined_hybrid_without_offsets", "selected_low_drift_features"],
        model_types=["ridge"],
        drift_csv=drift_csv,
        drift_threshold=1.0,
    )

    for key in (
        "summary_path",
        "report_path",
        "results_path",
        "per_target_results_path",
        "feature_selection_path",
        "model_importance_path",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["run_count"] == 3
    assert dataset_root.name in summary["best_by_dataset"]
    with Path(result["per_target_results_path"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {"dataset", "feature_config", "model_type", "target", "test_mae"} <= set(rows[0])


def test_invalid_feature_config_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown feature config"):
        select_feature_names(get_feature_names(), "not_a_config")


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3r_test"
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
            front_width = 16 + (index % 8)
            side_width = 8 + (index % 5)
            _write_rect_image(front_dir / f"{sample_id}_front.png", rect=(20, 10, 20 + front_width, 54), size=(64, 64))
            _write_rect_image(side_dir / f"{sample_id}_side.png", rect=(24, 10, 24 + side_width, 54), size=(64, 64))
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
                    "hip_cm": str(85 + front_width),
                    "shoulder_cm": str(38 + (index % 5)),
                    "inseam_cm": str(70 + (index % 8)),
                    "sleeve_cm": str(55 + (index % 7)),
                    "neck_cm": str(32 + (index % 4)),
                    "thigh_cm": str(45 + side_width),
                    "calf_cm": str(32 + (index % 6)),
                    "body_shape": "average",
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    result = build_dataset_manifest(dataset_root)
    assert result["valid"] is True
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (50, 50, 50))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)
