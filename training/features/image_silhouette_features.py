from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

FEATURE_EXTRACTOR_VERSION = "silhouette_geometry_v5_hybrid"
CANONICAL_MASK_WIDTH = 256
CANONICAL_MASK_HEIGHT = 256
CANONICAL_BODY_HEIGHT = 220
TORSO_REGION = (0.24, 0.62)
LOWER_LEG_REGION = (0.78, 0.94)
UPPER_BODY_REGION = (0.18, 0.38)
VOLUME_BANDS = ("neck", "shoulder", "chest", "waist", "hip", "thigh", "calf")
TORSO_VOLUME_BANDS = ("shoulder", "upper_chest", "chest", "mid_torso", "waist", "hip")
LOWER_BODY_VOLUME_BANDS = ("upper_thigh", "thigh", "knee", "calf", "ankle")
BAND_DEFINITIONS = [
    ("head", 0.10),
    ("neck", 0.18),
    ("shoulder", 0.24),
    ("upper_chest", 0.29),
    ("chest", 0.34),
    ("mid_torso", 0.42),
    ("waist", 0.48),
    ("hip", 0.60),
    ("upper_thigh", 0.68),
    ("thigh", 0.74),
    ("knee", 0.80),
    ("calf", 0.86),
    ("ankle", 0.94),
]
BASE_FEATURE_SUFFIXES = [
    "image_width_px",
    "image_height_px",
    "foreground_area_ratio",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_ratio",
    "bbox_center_y_ratio",
    "body_top_y_ratio",
    "body_bottom_y_ratio",
    "body_left_x_ratio",
    "body_right_x_ratio",
    "upper_body_area_ratio",
    "lower_body_area_ratio",
    "upper_to_lower_area_ratio",
    "max_row_width_ratio",
    "mean_row_width_ratio",
    "median_row_width_ratio",
    "arm_span_width_ratio",
    "arm_span_to_torso_ratio",
    "left_extension_ratio",
    "right_extension_ratio",
    "left_right_extension_delta",
]
GEOMETRY_FEATURE_SUFFIXES = [
    "area_to_height_ratio",
    "area_to_bbox_area_ratio",
    "torso_area_ratio",
    "torso_integrated_width_ratio",
    "lower_body_integrated_width_ratio",
    "upper_body_max_width_ratio",
    "upper_body_max_width_y_ratio",
    "min_torso_width_ratio",
    "min_torso_width_y_ratio",
    "waist_min_torso_width_ratio",
    "waist_to_chest_width_ratio",
    "waist_to_hip_width_ratio",
    "shoulder_peak_width_ratio",
    "shoulder_peak_y_ratio",
    "shoulder_to_hip_width_ratio",
    "shoulder_peak_to_waist_width_ratio",
    "shoulder_slope_proxy",
    "neck_min_width_ratio",
    "neck_min_width_y_ratio",
    "neck_to_head_width_ratio",
    "neck_to_shoulder_width_ratio",
    "neck_head_transition_delta",
    "lower_leg_max_width_ratio",
    "lower_leg_max_width_y_ratio",
    "lower_leg_min_width_ratio",
    "calf_peak_width_ratio",
    "calf_peak_y_ratio",
    "calf_to_ankle_width_ratio",
    "calf_to_thigh_width_ratio",
]
RAW_SCALE_CAMERA_FEATURE_SUFFIXES = [
    "raw_image_width_px",
    "raw_image_height_px",
    "raw_bbox_width_px",
    "raw_bbox_height_px",
    "raw_mask_area_px",
    "raw_bbox_aspect_ratio",
    "raw_bbox_width_ratio",
    "raw_bbox_height_ratio",
    "raw_mask_area_ratio",
    "normalization_scale_factor",
    "crop_offset_x",
    "crop_offset_y",
    "crop_offset_x_ratio",
    "crop_offset_y_ratio",
]
CROSS_VIEW_GEOMETRY_FEATURES = [
    "front_side_area_product_proxy",
    "front_side_integrated_volume_proxy",
    "front_side_torso_volume_proxy",
    "front_side_lower_body_volume_proxy",
    "front_side_area_to_height_proxy",
    "front_side_waist_width_depth_proxy",
    "front_side_chest_width_depth_proxy",
    "front_side_hip_width_depth_proxy",
    "front_side_shoulder_width_depth_proxy",
    "front_side_neck_width_depth_proxy",
    "front_side_calf_width_depth_proxy",
    "front_side_min_torso_width_depth_proxy",
    "front_side_lower_leg_width_depth_proxy",
]


def get_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("front", "side"):
        names.extend(f"{prefix}_{suffix}" for suffix in BASE_FEATURE_SUFFIXES)
        names.extend(f"{prefix}_{band_name}_width_ratio" for band_name, _center in BAND_DEFINITIONS)
        names.extend(f"{prefix}_{band_name}_left_extent_ratio" for band_name, _center in BAND_DEFINITIONS)
        names.extend(f"{prefix}_{band_name}_right_extent_ratio" for band_name, _center in BAND_DEFINITIONS)
        names.extend(f"{prefix}_{band_name}_center_x_ratio" for band_name, _center in BAND_DEFINITIONS)
        names.extend(f"{prefix}_{band_name}_asymmetry_ratio" for band_name, _center in BAND_DEFINITIONS)
        names.extend(
            [
                f"{prefix}_shoulder_to_waist_width_ratio",
                f"{prefix}_chest_to_waist_width_ratio",
                f"{prefix}_hip_to_waist_width_ratio",
                f"{prefix}_thigh_to_height_ratio",
                f"{prefix}_calf_to_height_ratio",
                f"{prefix}_upper_thigh_to_waist_width_ratio",
            ]
        )
        names.extend(f"{prefix}_{suffix}" for suffix in GEOMETRY_FEATURE_SUFFIXES)
        names.extend(f"{prefix}_{suffix}" for suffix in RAW_SCALE_CAMERA_FEATURE_SUFFIXES)
    names.extend(
        [
            "front_to_side_bbox_height_ratio",
            "front_to_side_bbox_width_ratio",
            "front_to_side_area_ratio",
        ]
    )
    names.extend(CROSS_VIEW_GEOMETRY_FEATURES)
    return names


def extract_front_side_features(front_image_path: str | Path, side_image_path: str | Path, normalize: bool = True) -> dict[str, float]:
    front_features = extract_image_features(front_image_path, "front", normalize=normalize)
    side_features = extract_image_features(side_image_path, "side", normalize=normalize)
    features = {**front_features, **side_features}
    features["front_to_side_bbox_height_ratio"] = _safe_ratio(
        features["front_raw_bbox_height_px"],
        features["side_raw_bbox_height_px"],
    )
    features["front_to_side_bbox_width_ratio"] = _safe_ratio(
        features["front_raw_bbox_width_px"],
        features["side_raw_bbox_width_px"],
    )
    features["front_to_side_area_ratio"] = _safe_ratio(
        features["front_raw_mask_area_ratio"],
        features["side_raw_mask_area_ratio"],
    )
    features.update(extract_cross_view_geometry_features(features))
    return {name: float(features[name]) for name in get_feature_names()}


def extract_image_features(image_path: str | Path, prefix: str, normalize: bool = True) -> dict[str, float]:
    image = load_rgb_image(image_path)
    raw_mask = create_foreground_mask(image)
    if normalize:
        mask, normalization = _normalize_body_mask_and_metadata(raw_mask)
    else:
        mask = raw_mask
        _normalized_mask, normalization = _normalize_body_mask_and_metadata(raw_mask)
    features = extract_mask_features(mask, prefix)
    features.update(extract_raw_scale_camera_features(raw_mask, prefix, normalization))
    return features


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {path}") from error


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {path}") from error


def create_foreground_mask(image: np.ndarray, min_contrast: float = 12.0) -> np.ndarray:
    if image.ndim == 3:
        return create_color_distance_foreground_mask(image, min_distance=min_contrast)
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image array or 3D RGB image array.")

    grayscale = image.astype(np.float32)
    border_pixels = np.concatenate(
        [
            grayscale[0, :],
            grayscale[-1, :],
            grayscale[:, 0],
            grayscale[:, -1],
        ]
    )
    background_level = float(np.median(border_pixels))
    lighter_mask = grayscale > background_level + min_contrast
    darker_mask = grayscale < background_level - min_contrast
    mask = lighter_mask if int(lighter_mask.sum()) >= int(darker_mask.sum()) else darker_mask
    return validate_foreground_mask(mask)


def create_color_distance_foreground_mask(rgb: np.ndarray, min_distance: float = 12.0) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape (height, width, 3).")

    image = rgb.astype(np.float32)
    border_pixels = np.concatenate(
        [
            image[0, :, :],
            image[-1, :, :],
            image[:, 0, :],
            image[:, -1, :],
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    distances = np.linalg.norm(image - background_color, axis=2)
    border_distances = np.concatenate(
        [
            distances[0, :],
            distances[-1, :],
            distances[:, 0],
            distances[:, -1],
        ]
    )
    adaptive_threshold = float(np.percentile(border_distances, 98) + 6.0)
    threshold = max(float(min_distance), adaptive_threshold)
    mask = distances > threshold

    try:
        return validate_foreground_mask(mask)
    except ValueError:
        grayscale = image.mean(axis=2)
        return create_foreground_mask(grayscale, min_contrast=max(8.0, min_distance))


def validate_foreground_mask(mask: np.ndarray, min_area_ratio: float = 0.002, max_area_ratio: float = 0.80) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Expected a 2D foreground mask.")
    if not bool(mask.any()):
        raise ValueError("No foreground pixels found with the current contrast threshold.")

    area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
    if area_ratio < min_area_ratio:
        raise ValueError(f"Foreground mask is too small or unstable: area_ratio={area_ratio:.6f}.")
    if area_ratio > max_area_ratio:
        raise ValueError(f"Foreground mask is too large or over-thresholded: area_ratio={area_ratio:.6f}.")

    _x_min, y_min, _x_max, y_max = foreground_bounding_box(mask)
    bbox_height_ratio = (y_max - y_min + 1) / float(mask.shape[0])
    if bbox_height_ratio < 0.20:
        raise ValueError(f"Foreground mask is too short or partial: bbox_height_ratio={bbox_height_ratio:.6f}.")
    return mask


def normalize_body_mask(
    mask: np.ndarray,
    canvas_width: int = CANONICAL_MASK_WIDTH,
    canvas_height: int = CANONICAL_MASK_HEIGHT,
    target_body_height: int = CANONICAL_BODY_HEIGHT,
    edge_margin: int = 1,
) -> np.ndarray:
    normalized, _metadata = _normalize_body_mask_and_metadata(
        mask,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        target_body_height=target_body_height,
        edge_margin=edge_margin,
    )
    return normalized


def _normalize_body_mask_and_metadata(
    mask: np.ndarray,
    canvas_width: int = CANONICAL_MASK_WIDTH,
    canvas_height: int = CANONICAL_MASK_HEIGHT,
    target_body_height: int = CANONICAL_BODY_HEIGHT,
    edge_margin: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    validate_foreground_mask(mask)
    if canvas_width <= 0 or canvas_height <= 0 or target_body_height <= 0:
        raise ValueError("Canonical mask dimensions and target body height must be positive.")
    if target_body_height > canvas_height:
        raise ValueError("Target body height must fit inside the canonical canvas.")

    x_min, y_min, x_max, y_max = foreground_bounding_box(mask)
    if (
        x_min <= edge_margin
        or y_min <= edge_margin
        or x_max >= mask.shape[1] - 1 - edge_margin
        or y_max >= mask.shape[0] - 1 - edge_margin
    ):
        raise ValueError("Foreground mask appears truncated at the image boundary.")

    cropped = mask[y_min : y_max + 1, x_min : x_max + 1]
    bbox_height, bbox_width = cropped.shape
    scale = target_body_height / float(bbox_height)
    resized_width = max(1, int(round(bbox_width * scale)))
    resized_height = target_body_height
    if resized_width > canvas_width:
        scale = canvas_width / float(bbox_width)
        resized_width = canvas_width
        resized_height = max(1, int(round(bbox_height * scale)))
        if resized_height > canvas_height:
            raise ValueError("Normalized foreground mask does not fit the canonical canvas.")

    resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize((resized_width, resized_height), resample=resample)
    resized_mask = np.asarray(resized, dtype=np.uint8) > 0

    normalized = np.zeros((canvas_height, canvas_width), dtype=bool)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    normalized[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized_mask
    metadata = {
        "normalization_scale_factor": float(scale),
        "crop_offset_x": float(x_min),
        "crop_offset_y": float(y_min),
        "crop_offset_x_ratio": float(x_min) / float(mask.shape[1]),
        "crop_offset_y_ratio": float(y_min) / float(mask.shape[0]),
    }
    return validate_foreground_mask(normalized), metadata


def extract_raw_scale_camera_features(mask: np.ndarray, prefix: str, normalization: dict[str, float]) -> dict[str, float]:
    validate_foreground_mask(mask)
    image_height, image_width = mask.shape
    x_min, y_min, x_max, y_max = foreground_bounding_box(mask)
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    mask_area = float(mask.sum())
    return {
        f"{prefix}_raw_image_width_px": float(image_width),
        f"{prefix}_raw_image_height_px": float(image_height),
        f"{prefix}_raw_bbox_width_px": float(bbox_width),
        f"{prefix}_raw_bbox_height_px": float(bbox_height),
        f"{prefix}_raw_mask_area_px": mask_area,
        f"{prefix}_raw_bbox_aspect_ratio": _safe_ratio(bbox_width, bbox_height),
        f"{prefix}_raw_bbox_width_ratio": _safe_ratio(bbox_width, image_width),
        f"{prefix}_raw_bbox_height_ratio": _safe_ratio(bbox_height, image_height),
        f"{prefix}_raw_mask_area_ratio": mask_area / float(image_width * image_height),
        f"{prefix}_normalization_scale_factor": float(normalization["normalization_scale_factor"]),
        f"{prefix}_crop_offset_x": float(normalization["crop_offset_x"]),
        f"{prefix}_crop_offset_y": float(normalization["crop_offset_y"]),
        f"{prefix}_crop_offset_x_ratio": float(normalization["crop_offset_x_ratio"]),
        f"{prefix}_crop_offset_y_ratio": float(normalization["crop_offset_y_ratio"]),
    }


def extract_mask_features(mask: np.ndarray, prefix: str) -> dict[str, float]:
    if mask.ndim != 2:
        raise ValueError("Expected a 2D foreground mask.")

    image_height, image_width = mask.shape
    x_min, y_min, x_max, y_max = foreground_bounding_box(mask)
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    foreground_area = float(mask.sum())
    bbox_center_x = x_min + bbox_width / 2
    bbox_center_y = y_min + bbox_height / 2
    row_widths = _row_widths(mask)
    upper_area = float(mask[: y_min + int(round(bbox_height * 0.50)), :].sum())
    lower_area = float(mask[y_min + int(round(bbox_height * 0.50)) :, :].sum())
    shoulder_width = _band_width_ratio(mask, _band_center("shoulder"))
    waist_width = _band_width_ratio(mask, _band_center("waist"))
    chest_width = _band_width_ratio(mask, _band_center("chest"))
    hip_width = _band_width_ratio(mask, _band_center("hip"))
    thigh_width = _band_width_ratio(mask, _band_center("thigh"))
    calf_width = _band_width_ratio(mask, _band_center("calf"))
    upper_thigh_width = _band_width_ratio(mask, _band_center("upper_thigh"))
    arm_span_width = _body_band_width_ratio(mask, 0.18, 0.58)
    torso_area = _region_area(mask, *TORSO_REGION)
    torso_integrated_width = _region_mean_width_ratio(mask, *TORSO_REGION)
    lower_body_integrated_width = _region_mean_width_ratio(mask, 0.62, 0.96)
    upper_body_max_width, upper_body_max_y = _region_extreme_width(mask, *UPPER_BODY_REGION, mode="max")
    min_torso_width, min_torso_y = _region_extreme_width(mask, *TORSO_REGION, mode="min")
    lower_leg_max_width, lower_leg_max_y = _region_extreme_width(mask, *LOWER_LEG_REGION, mode="max")
    lower_leg_min_width, _lower_leg_min_y = _region_extreme_width(mask, *LOWER_LEG_REGION, mode="min")
    calf_peak_width, calf_peak_y = _region_extreme_width(mask, 0.82, 0.91, mode="max")
    shoulder_peak_width, shoulder_peak_y = _region_extreme_width(mask, 0.20, 0.30, mode="max")
    neck_min_width, neck_min_y = _region_extreme_width(mask, 0.13, 0.23, mode="min")
    head_width = _band_width_ratio(mask, _band_center("head"))
    neck_width = _band_width_ratio(mask, _band_center("neck"))
    ankle_width = _band_width_ratio(mask, _band_center("ankle"))

    features = {
        f"{prefix}_image_width_px": float(image_width),
        f"{prefix}_image_height_px": float(image_height),
        f"{prefix}_foreground_area_ratio": foreground_area / float(image_width * image_height),
        f"{prefix}_bbox_width_px": float(bbox_width),
        f"{prefix}_bbox_height_px": float(bbox_height),
        f"{prefix}_bbox_width_ratio": bbox_width / float(image_width),
        f"{prefix}_bbox_height_ratio": bbox_height / float(image_height),
        f"{prefix}_bbox_aspect_ratio": bbox_width / float(max(bbox_height, 1)),
        f"{prefix}_bbox_center_x_ratio": bbox_center_x / float(image_width),
        f"{prefix}_bbox_center_y_ratio": bbox_center_y / float(image_height),
        f"{prefix}_body_top_y_ratio": y_min / float(image_height),
        f"{prefix}_body_bottom_y_ratio": y_max / float(image_height),
        f"{prefix}_body_left_x_ratio": x_min / float(image_width),
        f"{prefix}_body_right_x_ratio": x_max / float(image_width),
        f"{prefix}_upper_body_area_ratio": upper_area / float(max(foreground_area, 1.0)),
        f"{prefix}_lower_body_area_ratio": lower_area / float(max(foreground_area, 1.0)),
        f"{prefix}_upper_to_lower_area_ratio": _safe_ratio(upper_area, lower_area),
        f"{prefix}_max_row_width_ratio": float(row_widths.max()) / float(image_width),
        f"{prefix}_mean_row_width_ratio": float(row_widths[row_widths > 0].mean()) / float(image_width),
        f"{prefix}_median_row_width_ratio": float(np.median(row_widths[row_widths > 0])) / float(image_width),
        f"{prefix}_arm_span_width_ratio": arm_span_width,
        f"{prefix}_arm_span_to_torso_ratio": _safe_ratio(arm_span_width, max(chest_width, waist_width)),
        f"{prefix}_left_extension_ratio": max(0.0, (arm_span_width - chest_width) / 2),
        f"{prefix}_right_extension_ratio": max(0.0, (arm_span_width - chest_width) / 2),
        f"{prefix}_left_right_extension_delta": 0.0,
        f"{prefix}_area_to_height_ratio": _safe_ratio(foreground_area / float(image_width * image_height), bbox_height / float(image_height)),
        f"{prefix}_area_to_bbox_area_ratio": _safe_ratio(foreground_area, bbox_width * bbox_height),
        f"{prefix}_torso_area_ratio": torso_area / float(image_width * image_height),
        f"{prefix}_torso_integrated_width_ratio": torso_integrated_width,
        f"{prefix}_lower_body_integrated_width_ratio": lower_body_integrated_width,
        f"{prefix}_upper_body_max_width_ratio": upper_body_max_width,
        f"{prefix}_upper_body_max_width_y_ratio": upper_body_max_y,
        f"{prefix}_min_torso_width_ratio": min_torso_width,
        f"{prefix}_min_torso_width_y_ratio": min_torso_y,
        f"{prefix}_waist_min_torso_width_ratio": _safe_ratio(waist_width, min_torso_width),
        f"{prefix}_waist_to_chest_width_ratio": _safe_ratio(waist_width, chest_width),
        f"{prefix}_waist_to_hip_width_ratio": _safe_ratio(waist_width, hip_width),
        f"{prefix}_shoulder_peak_width_ratio": shoulder_peak_width,
        f"{prefix}_shoulder_peak_y_ratio": shoulder_peak_y,
        f"{prefix}_shoulder_to_hip_width_ratio": _safe_ratio(shoulder_width, hip_width),
        f"{prefix}_shoulder_peak_to_waist_width_ratio": _safe_ratio(shoulder_peak_width, waist_width),
        f"{prefix}_shoulder_slope_proxy": abs(shoulder_width - upper_body_max_width),
        f"{prefix}_neck_min_width_ratio": neck_min_width,
        f"{prefix}_neck_min_width_y_ratio": neck_min_y,
        f"{prefix}_neck_to_head_width_ratio": _safe_ratio(neck_width, head_width),
        f"{prefix}_neck_to_shoulder_width_ratio": _safe_ratio(neck_width, shoulder_width),
        f"{prefix}_neck_head_transition_delta": head_width - neck_width,
        f"{prefix}_lower_leg_max_width_ratio": lower_leg_max_width,
        f"{prefix}_lower_leg_max_width_y_ratio": lower_leg_max_y,
        f"{prefix}_lower_leg_min_width_ratio": lower_leg_min_width,
        f"{prefix}_calf_peak_width_ratio": calf_peak_width,
        f"{prefix}_calf_peak_y_ratio": calf_peak_y,
        f"{prefix}_calf_to_ankle_width_ratio": _safe_ratio(calf_width, ankle_width),
        f"{prefix}_calf_to_thigh_width_ratio": _safe_ratio(calf_width, thigh_width),
    }
    for band_name, center in BAND_DEFINITIONS:
        features[f"{prefix}_{band_name}_width_ratio"] = _band_width_ratio(mask, center)
        left_extent, right_extent, center_x, asymmetry = _band_extent_features(mask, center)
        features[f"{prefix}_{band_name}_left_extent_ratio"] = left_extent
        features[f"{prefix}_{band_name}_right_extent_ratio"] = right_extent
        features[f"{prefix}_{band_name}_center_x_ratio"] = center_x
        features[f"{prefix}_{band_name}_asymmetry_ratio"] = asymmetry
    features[f"{prefix}_shoulder_to_waist_width_ratio"] = _safe_ratio(shoulder_width, waist_width)
    features[f"{prefix}_chest_to_waist_width_ratio"] = _safe_ratio(chest_width, waist_width)
    features[f"{prefix}_hip_to_waist_width_ratio"] = _safe_ratio(hip_width, waist_width)
    features[f"{prefix}_thigh_to_height_ratio"] = _safe_ratio(thigh_width, bbox_height / float(image_height))
    features[f"{prefix}_calf_to_height_ratio"] = _safe_ratio(calf_width, bbox_height / float(image_height))
    features[f"{prefix}_upper_thigh_to_waist_width_ratio"] = _safe_ratio(upper_thigh_width, waist_width)
    return features


def foreground_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Foreground mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_cross_view_geometry_features(features: dict[str, float]) -> dict[str, float]:
    cross_features = {
        "front_side_area_product_proxy": features["front_foreground_area_ratio"] * features["side_foreground_area_ratio"],
        "front_side_integrated_volume_proxy": _band_volume_proxy(features, VOLUME_BANDS),
        "front_side_torso_volume_proxy": _band_volume_proxy(features, TORSO_VOLUME_BANDS),
        "front_side_lower_body_volume_proxy": _band_volume_proxy(features, LOWER_BODY_VOLUME_BANDS),
        "front_side_area_to_height_proxy": (
            features["front_area_to_height_ratio"] * features["side_area_to_height_ratio"]
        ),
        "front_side_waist_width_depth_proxy": features["front_waist_width_ratio"] * features["side_waist_width_ratio"],
        "front_side_chest_width_depth_proxy": features["front_chest_width_ratio"] * features["side_chest_width_ratio"],
        "front_side_hip_width_depth_proxy": features["front_hip_width_ratio"] * features["side_hip_width_ratio"],
        "front_side_shoulder_width_depth_proxy": features["front_shoulder_width_ratio"] * features["side_shoulder_width_ratio"],
        "front_side_neck_width_depth_proxy": features["front_neck_width_ratio"] * features["side_neck_width_ratio"],
        "front_side_calf_width_depth_proxy": features["front_calf_width_ratio"] * features["side_calf_width_ratio"],
        "front_side_min_torso_width_depth_proxy": features["front_min_torso_width_ratio"] * features["side_min_torso_width_ratio"],
        "front_side_lower_leg_width_depth_proxy": features["front_lower_leg_max_width_ratio"] * features["side_lower_leg_max_width_ratio"],
    }
    return cross_features


def feature_vector(features: dict[str, Any], feature_names: list[str] | None = None) -> list[float]:
    names = feature_names or get_feature_names()
    return [float(features[name]) for name in names]


def _band_width_ratio(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float = 0.025) -> float:
    occupied_columns = _band_occupied_columns(mask, center_y_ratio, half_window_ratio)
    if len(occupied_columns) == 0:
        return 0.0
    return (int(occupied_columns.max()) - int(occupied_columns.min()) + 1) / float(mask.shape[1])


def _band_extent_features(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float = 0.025) -> tuple[float, float, float, float]:
    occupied_columns = _band_occupied_columns(mask, center_y_ratio, half_window_ratio)
    image_width = mask.shape[1]
    if len(occupied_columns) == 0:
        return 0.0, 0.0, 0.0, 0.0

    left = int(occupied_columns.min()) / float(image_width)
    right = int(occupied_columns.max()) / float(image_width)
    center = (left + right) / 2
    asymmetry = abs((0.5 - left) - (right - 0.5))
    return left, right, center, asymmetry


def _band_occupied_columns(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float) -> np.ndarray:
    image_height, image_width = mask.shape
    y_start = max(0, int(round((center_y_ratio - half_window_ratio) * image_height)))
    y_end = min(image_height, int(round((center_y_ratio + half_window_ratio) * image_height)))
    if y_end <= y_start:
        y_end = min(image_height, y_start + 1)

    band = mask[y_start:y_end, :]
    return np.where(band.any(axis=0))[0]


def _body_band_width_ratio(mask: np.ndarray, start_ratio: float, end_ratio: float) -> float:
    image_height, image_width = mask.shape
    y_start = max(0, int(round(start_ratio * image_height)))
    y_end = min(image_height, int(round(end_ratio * image_height)))
    occupied_columns = np.where(mask[y_start:y_end, :].any(axis=0))[0]
    if len(occupied_columns) == 0:
        return 0.0
    return (int(occupied_columns.max()) - int(occupied_columns.min()) + 1) / float(image_width)


def _region_area(mask: np.ndarray, start_ratio: float, end_ratio: float) -> float:
    image_height = mask.shape[0]
    y_start, y_end = _region_bounds(image_height, start_ratio, end_ratio)
    return float(mask[y_start:y_end, :].sum())


def _region_mean_width_ratio(mask: np.ndarray, start_ratio: float, end_ratio: float) -> float:
    image_height, image_width = mask.shape
    y_start, y_end = _region_bounds(image_height, start_ratio, end_ratio)
    row_widths = _row_widths(mask[y_start:y_end, :])
    nonzero_widths = row_widths[row_widths > 0]
    if len(nonzero_widths) == 0:
        return 0.0
    return float(nonzero_widths.mean()) / float(image_width)


def _region_extreme_width(mask: np.ndarray, start_ratio: float, end_ratio: float, mode: str) -> tuple[float, float]:
    image_height, image_width = mask.shape
    y_start, y_end = _region_bounds(image_height, start_ratio, end_ratio)
    row_widths = _row_widths(mask)
    region_widths = row_widths[y_start:y_end]
    occupied_indices = np.where(region_widths > 0)[0]
    if len(occupied_indices) == 0:
        return 0.0, 0.0

    occupied_widths = region_widths[occupied_indices]
    if mode == "max":
        local_index = int(occupied_indices[int(np.argmax(occupied_widths))])
    elif mode == "min":
        local_index = int(occupied_indices[int(np.argmin(occupied_widths))])
    else:
        raise ValueError(f"Unsupported width mode: {mode}")
    y_index = y_start + local_index
    return float(row_widths[y_index]) / float(image_width), y_index / float(image_height)


def _region_bounds(image_height: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    y_start = max(0, int(round(start_ratio * image_height)))
    y_end = min(image_height, int(round(end_ratio * image_height)))
    if y_end <= y_start:
        y_end = min(image_height, y_start + 1)
    return y_start, y_end


def _band_volume_proxy(features: dict[str, float], band_names: tuple[str, ...]) -> float:
    values = [
        features[f"front_{band_name}_width_ratio"] * features[f"side_{band_name}_width_ratio"]
        for band_name in band_names
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _row_widths(mask: np.ndarray) -> np.ndarray:
    return mask.sum(axis=1).astype(np.float64)


def _band_center(band_name: str) -> float:
    return dict(BAND_DEFINITIONS)[band_name]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
