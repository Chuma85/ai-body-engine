import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.features.image_silhouette_features import (
    CROSS_VIEW_GEOMETRY_FEATURES,
    FEATURE_EXTRACTOR_VERSION,
    CANONICAL_BODY_HEIGHT,
    CANONICAL_MASK_HEIGHT,
    CANONICAL_MASK_WIDTH,
    create_color_distance_foreground_mask,
    create_foreground_mask,
    extract_cross_view_geometry_features,
    extract_front_side_features,
    extract_image_features,
    extract_mask_features,
    feature_vector,
    foreground_bounding_box,
    get_feature_names,
    normalize_body_mask,
)
from training.train_image_feature_baseline import main, train_image_feature_baseline


def test_foreground_mask_creation_on_tiny_image() -> None:
    grayscale = np.full((10, 10), 50, dtype=np.float32)
    grayscale[3:7, 4:8] = 210

    mask = create_foreground_mask(grayscale)

    assert mask.sum() == 16
    assert mask[3, 4] is np.True_ or bool(mask[3, 4]) is True
    assert bool(mask[0, 0]) is False


def test_bounding_box_feature_extraction(tmp_path) -> None:
    image_path = tmp_path / "body.png"
    _write_rect_image(image_path, rect=(4, 3, 7, 6), size=(10, 10))

    features = extract_image_features(image_path, "front")

    assert foreground_bounding_box(create_foreground_mask(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32))) == (4, 3, 7, 6)
    assert features["front_image_width_px"] == CANONICAL_MASK_WIDTH
    assert features["front_image_height_px"] == CANONICAL_MASK_HEIGHT
    assert features["front_bbox_width_px"] == CANONICAL_BODY_HEIGHT
    assert features["front_bbox_height_px"] == CANONICAL_BODY_HEIGHT
    assert features["front_foreground_area_ratio"] == pytest.approx((CANONICAL_BODY_HEIGHT * CANONICAL_BODY_HEIGHT) / (CANONICAL_MASK_WIDTH * CANONICAL_MASK_HEIGHT))
    assert features["front_raw_bbox_width_px"] == 4.0
    assert features["front_raw_bbox_height_px"] == 4.0
    assert features["front_raw_mask_area_px"] == 16.0
    assert features["front_normalization_scale_factor"] == pytest.approx(CANONICAL_BODY_HEIGHT / 4)
    assert features["front_crop_offset_x"] == 4.0
    assert features["front_crop_offset_y"] == 3.0


def test_feature_vector_has_stable_names_and_order(tmp_path) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_rect_image(front_path, rect=(3, 2, 8, 9), size=(12, 12))
    _write_rect_image(side_path, rect=(5, 2, 7, 9), size=(12, 12))

    names = get_feature_names()
    features = extract_front_side_features(front_path, side_path)
    vector = feature_vector(features, names)

    assert names[:4] == [
        "front_image_width_px",
        "front_image_height_px",
        "front_foreground_area_ratio",
        "front_bbox_width_px",
    ]
    assert names[-len(CROSS_VIEW_GEOMETRY_FEATURES):] == CROSS_VIEW_GEOMETRY_FEATURES
    assert "front_body_top_y_ratio" in names
    assert "front_shoulder_to_waist_width_ratio" in names
    assert "front_thigh_to_height_ratio" in names
    assert "front_area_to_height_ratio" in names
    assert "front_waist_min_torso_width_ratio" in names
    assert "front_neck_to_shoulder_width_ratio" in names
    assert "front_calf_to_ankle_width_ratio" in names
    assert "front_raw_bbox_height_px" in names
    assert "front_normalization_scale_factor" in names
    assert "side_crop_offset_y_ratio" in names
    assert "side_hip_center_x_ratio" in names
    assert "front_to_side_bbox_height_ratio" in names
    assert "front_side_torso_volume_proxy" in names
    assert FEATURE_EXTRACTOR_VERSION == "silhouette_geometry_v5_hybrid"
    assert len(vector) == len(names)
    assert vector == feature_vector(features, names)
    assert features["front_to_side_bbox_width_ratio"] == pytest.approx(
        features["front_raw_bbox_width_px"] / features["side_raw_bbox_width_px"]
    )


def test_tiny_mask_profile_features_are_deterministic() -> None:
    mask = np.zeros((100, 50), dtype=bool)
    mask[10:91, 20:31] = True
    mask[24:27, 15:36] = True
    mask[48:51, 18:33] = True
    mask[74:77, 21:30] = True

    features = extract_mask_features(mask, "front")

    assert features["front_body_top_y_ratio"] == 0.10
    assert features["front_body_bottom_y_ratio"] == 0.90
    assert features["front_shoulder_width_ratio"] == 21 / 50
    assert features["front_waist_width_ratio"] == 15 / 50
    assert features["front_thigh_width_ratio"] == 11 / 50
    assert features["front_shoulder_to_waist_width_ratio"] == pytest.approx((21 / 50) / (15 / 50))
    assert features["front_thigh_to_height_ratio"] == pytest.approx((11 / 50) / (81 / 100))
    assert features["front_shoulder_peak_width_ratio"] == pytest.approx(21 / 50)
    assert features["front_neck_min_width_ratio"] == pytest.approx(11 / 50)
    assert features["front_calf_peak_width_ratio"] == pytest.approx(11 / 50)
    assert features["front_calf_to_ankle_width_ratio"] == 0.0


def test_geometry_volume_proxy_features_are_deterministic() -> None:
    front_mask = np.zeros((100, 50), dtype=bool)
    side_mask = np.zeros((100, 50), dtype=bool)
    front_mask[10:91, 20:31] = True
    side_mask[10:91, 22:28] = True

    features = {
        **extract_mask_features(front_mask, "front"),
        **extract_mask_features(side_mask, "side"),
    }
    cross_features = extract_cross_view_geometry_features(features)

    expected_front_width = 11 / 50
    expected_side_width = 6 / 50
    expected_proxy = expected_front_width * expected_side_width
    assert features["front_area_to_height_ratio"] == pytest.approx((891 / 5000) / 0.81)
    assert features["front_torso_integrated_width_ratio"] == pytest.approx(expected_front_width)
    assert cross_features["front_side_integrated_volume_proxy"] == pytest.approx(expected_proxy)
    assert cross_features["front_side_torso_volume_proxy"] == pytest.approx(expected_proxy)
    assert cross_features["front_side_lower_body_volume_proxy"] == pytest.approx(expected_proxy * 4 / 5)


def test_neck_shoulder_and_calf_geometry_features_on_simple_mask() -> None:
    mask = np.zeros((100, 50), dtype=bool)
    mask[10:15, 19:32] = True
    mask[18:22, 22:29] = True
    mask[24:29, 14:37] = True
    mask[48:51, 18:33] = True
    mask[60:64, 16:35] = True
    mask[74:77, 20:31] = True
    mask[86:89, 21:30] = True
    mask[94:96, 23:28] = True

    features = extract_mask_features(mask, "front")

    assert features["front_neck_min_width_ratio"] == pytest.approx(7 / 50)
    assert features["front_neck_to_head_width_ratio"] == pytest.approx((7 / 50) / (13 / 50))
    assert features["front_neck_to_shoulder_width_ratio"] == pytest.approx((7 / 50) / (23 / 50))
    assert features["front_shoulder_peak_width_ratio"] == pytest.approx(23 / 50)
    assert features["front_shoulder_to_hip_width_ratio"] == pytest.approx((23 / 50) / (19 / 50))
    assert features["front_calf_peak_width_ratio"] == pytest.approx(9 / 50)
    assert features["front_calf_to_ankle_width_ratio"] == pytest.approx((9 / 50) / (5 / 50))
    assert features["front_calf_to_thigh_width_ratio"] == pytest.approx((9 / 50) / (11 / 50))


def test_empty_foreground_mask_raises_clear_error() -> None:
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="Foreground mask is empty"):
        extract_mask_features(mask, "front")


def test_rgb_foreground_mask_handles_bright_and_dark_backgrounds() -> None:
    bright_background = np.full((48, 48, 3), 235, dtype=np.float32)
    bright_background[8:42, 18:30, :] = np.asarray([150, 120, 100], dtype=np.float32)
    dark_background = np.full((48, 48, 3), 35, dtype=np.float32)
    dark_background[8:42, 18:30, :] = np.asarray([160, 130, 110], dtype=np.float32)

    bright_mask = create_foreground_mask(bright_background)
    dark_mask = create_foreground_mask(dark_background)

    assert bright_mask.sum() == 34 * 12
    assert dark_mask.sum() == 34 * 12


def test_rgb_foreground_mask_is_stable_under_material_brightness_changes() -> None:
    base = np.full((64, 64, 3), 60, dtype=np.float32)
    darker_body = base.copy()
    brighter_body = base.copy()
    darker_body[10:56, 24:40, :] = np.asarray([130, 105, 90], dtype=np.float32)
    brighter_body[10:56, 24:40, :] = np.asarray([210, 175, 145], dtype=np.float32)

    darker_mask = create_color_distance_foreground_mask(darker_body)
    brighter_mask = create_color_distance_foreground_mask(brighter_body)

    assert foreground_bounding_box(darker_mask) == foreground_bounding_box(brighter_mask)
    assert darker_mask.sum() == brighter_mask.sum()


def test_foreground_mask_handles_small_camera_framing_jitter() -> None:
    base = np.full((64, 64, 3), 50, dtype=np.float32)
    shifted = np.full((64, 64, 3), 50, dtype=np.float32)
    base[8:58, 24:40, :] = np.asarray([205, 180, 155], dtype=np.float32)
    shifted[10:60, 27:43, :] = np.asarray([205, 180, 155], dtype=np.float32)

    base_mask = create_foreground_mask(base)
    shifted_mask = create_foreground_mask(shifted)

    assert foreground_bounding_box(base_mask) == (24, 8, 39, 57)
    assert foreground_bounding_box(shifted_mask) == (27, 10, 42, 59)
    assert base_mask.sum() == shifted_mask.sum()


def test_normalized_mask_is_stable_for_left_right_shift() -> None:
    left = np.zeros((80, 80), dtype=bool)
    right = np.zeros((80, 80), dtype=bool)
    left[10:70, 18:34] = True
    right[10:70, 38:54] = True

    assert np.array_equal(normalize_body_mask(left), normalize_body_mask(right))


def test_normalized_mask_is_stable_for_up_down_shift() -> None:
    upper = np.zeros((90, 80), dtype=bool)
    lower = np.zeros((90, 80), dtype=bool)
    upper[8:68, 30:46] = True
    lower[22:82, 30:46] = True

    assert np.array_equal(normalize_body_mask(upper), normalize_body_mask(lower))


def test_normalized_mask_is_stable_with_extra_padding() -> None:
    tight = np.zeros((80, 60), dtype=bool)
    padded = np.zeros((120, 100), dtype=bool)
    tight[10:70, 22:38] = True
    padded[30:90, 42:58] = True

    assert np.array_equal(normalize_body_mask(tight), normalize_body_mask(padded))


def test_normalized_mask_is_stable_for_same_aspect_scale_change() -> None:
    small = np.zeros((80, 80), dtype=bool)
    large = np.zeros((120, 120), dtype=bool)
    small[10:70, 30:46] = True
    large[15:105, 45:69] = True

    assert np.array_equal(normalize_body_mask(small), normalize_body_mask(large))


def test_normalized_features_are_near_identical_for_shifted_images(tmp_path) -> None:
    base_path = tmp_path / "base.png"
    shifted_path = tmp_path / "shifted.png"
    _write_rect_image(base_path, rect=(24, 8, 39, 57), size=(80, 80))
    _write_rect_image(shifted_path, rect=(34, 18, 49, 67), size=(100, 100))

    base_features = extract_image_features(base_path, "front")
    shifted_features = extract_image_features(shifted_path, "front")

    assert base_features["front_bbox_height_px"] == shifted_features["front_bbox_height_px"]
    assert base_features["front_bbox_width_px"] == shifted_features["front_bbox_width_px"]
    assert base_features["front_bbox_center_x_ratio"] == pytest.approx(shifted_features["front_bbox_center_x_ratio"])
    assert base_features["front_bbox_center_y_ratio"] == pytest.approx(shifted_features["front_bbox_center_y_ratio"])
    assert base_features["front_crop_offset_x"] != shifted_features["front_crop_offset_x"]
    assert base_features["front_crop_offset_y"] != shifted_features["front_crop_offset_y"]


def test_hybrid_raw_scale_features_change_when_body_scale_changes(tmp_path) -> None:
    small_path = tmp_path / "small.png"
    large_path = tmp_path / "large.png"
    _write_rect_image(small_path, rect=(30, 20, 45, 79), size=(100, 120))
    _write_rect_image(large_path, rect=(45, 25, 68, 114), size=(140, 160))

    small_features = extract_image_features(small_path, "front")
    large_features = extract_image_features(large_path, "front")

    assert small_features["front_bbox_height_px"] == large_features["front_bbox_height_px"]
    assert small_features["front_bbox_width_px"] == large_features["front_bbox_width_px"]
    assert small_features["front_raw_bbox_height_px"] == 60.0
    assert large_features["front_raw_bbox_height_px"] == 90.0
    assert small_features["front_raw_mask_area_px"] != large_features["front_raw_mask_area_px"]
    assert small_features["front_normalization_scale_factor"] != large_features["front_normalization_scale_factor"]


def test_normalize_body_mask_rejects_truncated_masks() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[0:60, 20:40] = True

    with pytest.raises(ValueError, match="truncated"):
        normalize_body_mask(mask)


def test_normalize_body_mask_rejects_empty_masks() -> None:
    mask = np.zeros((80, 80), dtype=bool)

    with pytest.raises(ValueError, match="No foreground pixels"):
        normalize_body_mask(mask)


def test_normalize_body_mask_rejects_tiny_masks() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:22, 20:22] = True

    with pytest.raises(ValueError, match="too small"):
        normalize_body_mask(mask)


def test_foreground_mask_rejects_over_thresholded_masks() -> None:
    image = np.full((50, 50), 255.0, dtype=np.float32)
    image[0, :] = 0.0
    image[-1, :] = 0.0
    image[:, 0] = 0.0
    image[:, -1] = 0.0

    with pytest.raises(ValueError, match="over-thresholded"):
        create_foreground_mask(image, min_contrast=1.0)


def test_image_feature_training_runs_and_creates_metrics(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "phase_2n"
    monkeypatch.chdir(tmp_path)

    result = train_image_feature_baseline(dataset_root, output_dir)

    assert Path(result["metrics_path"]).exists()
    assert Path(result["model_path"]).exists()
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["model_type"] == "image_silhouette_ridge_regressor"
    assert metrics["sample_counts"] == {"train": 16, "val": 2, "test": 2}
    assert "front_bbox_width_ratio" in metrics["feature_names"]
    assert "front_arm_span_to_torso_ratio" in metrics["feature_names"]
    assert "overall_mae" in metrics["test"]


def test_image_feature_training_cli_creates_metrics_file(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    output_dir = tmp_path / "artifacts" / "phase_2n_cli"
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dataset", str(dataset_root), "--output", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()


def test_unreadable_image_raises_helpful_error(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    monkeypatch.chdir(tmp_path)
    first_train_front = _first_manifest_path_for_split(dataset_root, "train", "front_image_path")
    Path(first_train_front).write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Could not read image file"):
        train_image_feature_baseline(dataset_root, tmp_path / "artifacts" / "bad_image")


def test_missing_image_raises_helpful_error(tmp_path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 20)
    monkeypatch.chdir(tmp_path)
    first_train_front = _first_manifest_path_for_split(dataset_root, "train", "front_image_path")
    Path(first_train_front).unlink()

    with pytest.raises(FileNotFoundError, match="Missing front image"):
        train_image_feature_baseline(dataset_root, tmp_path / "artifacts" / "missing_image")


def _first_manifest_path_for_split(dataset_root: Path, split: str, path_column: str) -> str:
    with (dataset_root / "manifest.csv").open("r", newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row["dataset_split"] == split:
                return row[path_column]
    raise AssertionError(f"No row found for split {split}")


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_image_test"
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_dir = dataset_root / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    body_shapes = ["average", "athletic", "curvy", "broad"]
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
                    "body_shape": body_shapes[index % len(body_shapes)],
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
