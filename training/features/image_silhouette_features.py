from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

BAND_DEFINITIONS = [
    ("shoulder", 0.24),
    ("chest", 0.34),
    ("waist", 0.48),
    ("hip", 0.60),
    ("thigh", 0.74),
    ("calf", 0.86),
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
]


def get_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("front", "side"):
        names.extend(f"{prefix}_{suffix}" for suffix in BASE_FEATURE_SUFFIXES)
        names.extend(f"{prefix}_{band_name}_width_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(
        [
            "front_to_side_bbox_width_ratio",
            "front_to_side_area_ratio",
        ]
    )
    return names


def extract_front_side_features(front_image_path: str | Path, side_image_path: str | Path) -> dict[str, float]:
    front_features = extract_image_features(front_image_path, "front")
    side_features = extract_image_features(side_image_path, "side")
    features = {**front_features, **side_features}
    features["front_to_side_bbox_width_ratio"] = _safe_ratio(
        features["front_bbox_width_px"],
        features["side_bbox_width_px"],
    )
    features["front_to_side_area_ratio"] = _safe_ratio(
        features["front_foreground_area_ratio"],
        features["side_foreground_area_ratio"],
    )
    return {name: float(features[name]) for name in get_feature_names()}


def extract_image_features(image_path: str | Path, prefix: str) -> dict[str, float]:
    grayscale = load_grayscale_image(image_path)
    mask = create_foreground_mask(grayscale)
    return extract_mask_features(mask, prefix)


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {path}") from error


def create_foreground_mask(grayscale: np.ndarray, min_contrast: float = 20.0) -> np.ndarray:
    if grayscale.ndim != 2:
        raise ValueError("Expected a 2D grayscale image array.")

    # Current synthetic renders use a mostly uniform gray background and a lighter body.
    # Estimate the background from image borders so the threshold stays deterministic
    # while remaining tolerant of future light-on-dark or dark-on-light renders.
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

    if not bool(mask.any()):
        raise ValueError("No foreground pixels found with the current contrast threshold.")

    return mask


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
    }
    for band_name, center in BAND_DEFINITIONS:
        features[f"{prefix}_{band_name}_width_ratio"] = _band_width_ratio(mask, center)
    return features


def foreground_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Foreground mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def feature_vector(features: dict[str, Any], feature_names: list[str] | None = None) -> list[float]:
    names = feature_names or get_feature_names()
    return [float(features[name]) for name in names]


def _band_width_ratio(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float = 0.025) -> float:
    image_height, image_width = mask.shape
    y_start = max(0, int(round((center_y_ratio - half_window_ratio) * image_height)))
    y_end = min(image_height, int(round((center_y_ratio + half_window_ratio) * image_height)))
    if y_end <= y_start:
        y_end = min(image_height, y_start + 1)

    band = mask[y_start:y_end, :]
    occupied_columns = np.where(band.any(axis=0))[0]
    if len(occupied_columns) == 0:
        return 0.0
    return (int(occupied_columns.max()) - int(occupied_columns.min()) + 1) / float(image_width)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
