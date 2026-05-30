from __future__ import annotations

from pathlib import Path
from typing import Any

from training.features.image_silhouette_features import (
    BAND_DEFINITIONS,
    BASE_FEATURE_SUFFIXES,
    GEOMETRY_FEATURE_SUFFIXES,
    RAW_SCALE_CAMERA_FEATURE_SUFFIXES,
    extract_front_side_features,
    extract_image_features,
    _safe_ratio,
)

FEATURE_EXTRACTOR_VERSION = "back_view_silhouette_geometry_v1"
BACK_VIEW_PREFIX = "back"
BACK_VIEW_FEATURE_SUFFIXES = [
    "back_shoulder_width_proxy",
    "back_across_back_width_proxy",
    "back_upper_back_width_proxy",
    "back_upper_back_area_proxy",
    "back_torso_width_band_24",
    "back_torso_width_band_29",
    "back_torso_width_band_34",
    "back_torso_width_band_42",
    "back_torso_width_band_48",
    "back_torso_width_band_60",
]
FRONT_BACK_COMPARISON_FEATURES = [
    "front_back_shoulder_width_ratio",
    "front_back_shoulder_width_delta",
    "front_back_across_upper_ratio",
    "front_back_upper_area_ratio",
    "front_side_back_shoulder_volume_proxy",
    "front_side_back_upper_torso_volume_proxy",
    "front_side_back_torso_balance_proxy",
]


def get_back_view_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"back_{suffix}" for suffix in BASE_FEATURE_SUFFIXES)
    names.extend(f"back_{band_name}_width_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(f"back_{band_name}_left_extent_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(f"back_{band_name}_right_extent_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(f"back_{band_name}_center_x_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(f"back_{band_name}_asymmetry_ratio" for band_name, _center in BAND_DEFINITIONS)
    names.extend(
        [
            "back_shoulder_to_waist_width_ratio",
            "back_chest_to_waist_width_ratio",
            "back_hip_to_waist_width_ratio",
            "back_thigh_to_height_ratio",
            "back_calf_to_height_ratio",
            "back_upper_thigh_to_waist_width_ratio",
        ]
    )
    names.extend(f"back_{suffix}" for suffix in GEOMETRY_FEATURE_SUFFIXES)
    names.extend(f"back_{suffix}" for suffix in RAW_SCALE_CAMERA_FEATURE_SUFFIXES)
    names.extend(BACK_VIEW_FEATURE_SUFFIXES)
    return names


def get_front_side_back_feature_names(front_side_names: list[str] | None = None) -> list[str]:
    from training.features.image_silhouette_features import get_feature_names

    return [*(front_side_names or get_feature_names()), *get_back_view_feature_names(), *FRONT_BACK_COMPARISON_FEATURES]


def extract_back_view_features(back_image_path: str | Path | None, normalize: bool = True) -> dict[str, float]:
    path = _require_back_image_path(back_image_path)
    features = extract_image_features(path, BACK_VIEW_PREFIX, normalize=normalize)
    proxies = {
        "back_shoulder_width_proxy": features["back_shoulder_peak_width_ratio"],
        "back_across_back_width_proxy": features["back_upper_body_max_width_ratio"],
        "back_upper_back_width_proxy": features["back_upper_chest_width_ratio"],
        "back_upper_back_area_proxy": features["back_torso_area_ratio"],
        "back_torso_width_band_24": features["back_shoulder_width_ratio"],
        "back_torso_width_band_29": features["back_upper_chest_width_ratio"],
        "back_torso_width_band_34": features["back_chest_width_ratio"],
        "back_torso_width_band_42": features["back_mid_torso_width_ratio"],
        "back_torso_width_band_48": features["back_waist_width_ratio"],
        "back_torso_width_band_60": features["back_hip_width_ratio"],
    }
    features.update({name: float(value) for name, value in proxies.items()})
    return {name: float(features[name]) for name in get_back_view_feature_names()}


def extract_front_side_back_features(
    front_image_path: str | Path,
    side_image_path: str | Path,
    back_image_path: str | Path | None,
    normalize: bool = True,
) -> dict[str, float]:
    front_side_features = extract_front_side_features(front_image_path, side_image_path, normalize=normalize)
    back_features = extract_back_view_features(back_image_path, normalize=normalize)
    features = {**front_side_features, **back_features}
    features.update(extract_front_back_comparison_features(features))
    return {name: float(features[name]) for name in get_front_side_back_feature_names(list(front_side_features))}


def extract_front_back_comparison_features(features: dict[str, Any]) -> dict[str, float]:
    front_shoulder = float(features["front_shoulder_peak_width_ratio"])
    side_shoulder = float(features["side_shoulder_peak_width_ratio"])
    back_shoulder = float(features["back_shoulder_width_proxy"])
    back_upper = float(features["back_upper_back_width_proxy"])
    return {
        "front_back_shoulder_width_ratio": _safe_ratio(front_shoulder, back_shoulder),
        "front_back_shoulder_width_delta": front_shoulder - back_shoulder,
        "front_back_across_upper_ratio": _safe_ratio(float(features["front_upper_chest_width_ratio"]), back_upper),
        "front_back_upper_area_ratio": _safe_ratio(float(features["front_torso_area_ratio"]), float(features["back_upper_back_area_proxy"])),
        "front_side_back_shoulder_volume_proxy": front_shoulder * side_shoulder * back_shoulder,
        "front_side_back_upper_torso_volume_proxy": (
            float(features["front_upper_chest_width_ratio"])
            * float(features["side_upper_chest_width_ratio"])
            * back_upper
        ),
        "front_side_back_torso_balance_proxy": _safe_ratio(
            back_upper,
            (float(features["front_upper_chest_width_ratio"]) + float(features["side_upper_chest_width_ratio"])) / 2,
        ),
    }


def _require_back_image_path(back_image_path: str | Path | None) -> Path:
    if back_image_path in ("", None):
        raise ValueError("Missing back image path; back-view features require a back capture.")
    path = Path(back_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing back image: {path}")
    return path
