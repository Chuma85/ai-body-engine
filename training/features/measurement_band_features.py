from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.features.image_silhouette_features import (
    create_foreground_mask,
    foreground_bounding_box,
    load_rgb_image,
    normalize_body_mask,
)

FEATURE_EXTRACTOR_VERSION = "silhouette_geometry_v6_bands"
MEASUREMENT_BAND_TARGETS = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm"]
BAND_HALF_WINDOW = 0.025
SLOPE_DELTA = 0.045
BAND_CANDIDATES = {
    "chest_cm": [0.28, 0.32, 0.36, 0.40],
    "waist_cm": [0.42, 0.46, 0.50, 0.54],
    "hip_cm": [0.56, 0.60, 0.64, 0.68],
    "thigh_cm": [0.68, 0.72, 0.76, 0.80],
}
VIEW_FEATURE_SUFFIXES = [
    "norm_width_ratio",
    "norm_local_area_ratio",
    "norm_contour_slope",
    "raw_width_ratio",
    "raw_local_area_ratio",
    "raw_contour_slope",
]
CROSS_VIEW_SUFFIXES = [
    "norm_width_depth_product",
    "raw_width_depth_product",
    "norm_front_side_width_ratio",
    "raw_front_side_width_ratio",
    "band_center_y_ratio",
]


def candidate_band_definitions() -> list[dict[str, Any]]:
    definitions = []
    for target in MEASUREMENT_BAND_TARGETS:
        for index, center_y_ratio in enumerate(BAND_CANDIDATES[target]):
            definitions.append(
                {
                    "target": target,
                    "band_index": index,
                    "band_name": band_name(target, index, center_y_ratio),
                    "center_y_ratio": center_y_ratio,
                }
            )
    return definitions


def get_band_feature_names() -> list[str]:
    names: list[str] = []
    for definition in candidate_band_definitions():
        prefix = definition["band_name"]
        for view in ("front", "side"):
            names.extend(f"{prefix}_{view}_{suffix}" for suffix in VIEW_FEATURE_SUFFIXES)
        names.extend(f"{prefix}_{suffix}" for suffix in CROSS_VIEW_SUFFIXES)
    return names


def extract_front_side_band_features(front_image_path: str | Path, side_image_path: str | Path) -> dict[str, float]:
    front = extract_view_band_features(front_image_path)
    side = extract_view_band_features(side_image_path)
    features: dict[str, float] = {}
    for definition in candidate_band_definitions():
        prefix = definition["band_name"]
        for suffix in VIEW_FEATURE_SUFFIXES:
            features[f"{prefix}_front_{suffix}"] = front[prefix][suffix]
            features[f"{prefix}_side_{suffix}"] = side[prefix][suffix]
        front_norm_width = front[prefix]["norm_width_ratio"]
        side_norm_width = side[prefix]["norm_width_ratio"]
        front_raw_width = front[prefix]["raw_width_ratio"]
        side_raw_width = side[prefix]["raw_width_ratio"]
        features[f"{prefix}_norm_width_depth_product"] = front_norm_width * side_norm_width
        features[f"{prefix}_raw_width_depth_product"] = front_raw_width * side_raw_width
        features[f"{prefix}_norm_front_side_width_ratio"] = safe_ratio(front_norm_width, side_norm_width)
        features[f"{prefix}_raw_front_side_width_ratio"] = safe_ratio(front_raw_width, side_raw_width)
        features[f"{prefix}_band_center_y_ratio"] = float(definition["center_y_ratio"])
    return {name: float(features[name]) for name in get_band_feature_names()}


def extract_view_band_features(image_path: str | Path) -> dict[str, dict[str, float]]:
    image = load_rgb_image(image_path)
    raw_mask = create_foreground_mask(image)
    normalized_mask = normalize_body_mask(raw_mask)
    rows: dict[str, dict[str, float]] = {}
    for definition in candidate_band_definitions():
        prefix = definition["band_name"]
        center = float(definition["center_y_ratio"])
        rows[prefix] = {
            "norm_width_ratio": band_width_ratio(normalized_mask, center),
            "norm_local_area_ratio": local_area_ratio(normalized_mask, center),
            "norm_contour_slope": contour_slope(normalized_mask, center),
            "raw_width_ratio": band_width_ratio(raw_mask, center),
            "raw_local_area_ratio": local_area_ratio(raw_mask, center),
            "raw_contour_slope": contour_slope(raw_mask, center),
        }
    return rows


def band_width_ratio(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float = BAND_HALF_WINDOW) -> float:
    columns = band_occupied_columns(mask, center_y_ratio, half_window_ratio)
    if len(columns) == 0:
        return 0.0
    return float(int(columns.max()) - int(columns.min()) + 1) / float(mask.shape[1])


def local_area_ratio(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float = BAND_HALF_WINDOW) -> float:
    y_start, y_end = band_row_bounds(mask.shape[0], center_y_ratio, half_window_ratio)
    band_area = float(mask[y_start:y_end, :].sum())
    total_area = float(mask.sum())
    return safe_ratio(band_area, total_area)


def contour_slope(mask: np.ndarray, center_y_ratio: float, delta: float = SLOPE_DELTA) -> float:
    upper = band_width_ratio(mask, max(0.0, center_y_ratio - delta), half_window_ratio=BAND_HALF_WINDOW)
    lower = band_width_ratio(mask, min(1.0, center_y_ratio + delta), half_window_ratio=BAND_HALF_WINDOW)
    return lower - upper


def band_occupied_columns(mask: np.ndarray, center_y_ratio: float, half_window_ratio: float) -> np.ndarray:
    y_start, y_end = band_row_bounds(mask.shape[0], center_y_ratio, half_window_ratio)
    band = mask[y_start:y_end, :]
    return np.where(band.any(axis=0))[0]


def band_row_bounds(image_height: int, center_y_ratio: float, half_window_ratio: float) -> tuple[int, int]:
    y_start = max(0, int(round((center_y_ratio - half_window_ratio) * image_height)))
    y_end = min(image_height, int(round((center_y_ratio + half_window_ratio) * image_height)))
    if y_end <= y_start:
        y_end = min(image_height, y_start + 1)
    return y_start, y_end


def band_name(target: str, band_index: int, center_y_ratio: float) -> str:
    target_name = target.removesuffix("_cm")
    center_tag = f"{int(round(center_y_ratio * 100)):02d}"
    return f"{target_name}_band_{band_index:02d}_y{center_tag}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)
