from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REQUIRED_BODY_PARAMETER_KEYS = [
    "height_cm",
    "weight_kg",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
    "sleeve_cm",
    "neck_cm",
    "thigh_cm",
    "calf_cm",
]


@dataclass(frozen=True)
class RenderConfig:
    generator_version: str
    output_dir: str
    sample_count: int
    image_width: int
    image_height: int
    camera_distance: float
    camera_focal_length: float
    render_engine: str
    random_seed: int
    views: list[str]
    body_parameter_ranges: dict[str, list[float]]
    lighting: dict[str, Any]
    background: dict[str, Any]


def load_render_config(path: str) -> RenderConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    config = RenderConfig(**raw_config)
    validate_render_config(config)
    return config


def validate_render_config(config: RenderConfig) -> None:
    errors: list[str] = []

    if config.sample_count <= 0:
        errors.append("sample_count must be greater than 0")
    if config.image_width <= 0:
        errors.append("image_width must be greater than 0")
    if config.image_height <= 0:
        errors.append("image_height must be greater than 0")
    if not config.output_dir:
        errors.append("output_dir must not be empty")
    if "front" not in config.views or "side" not in config.views:
        errors.append("views must include both front and side")

    missing_keys = [key for key in REQUIRED_BODY_PARAMETER_KEYS if key not in config.body_parameter_ranges]
    if missing_keys:
        errors.append(f"body_parameter_ranges missing keys: {', '.join(missing_keys)}")

    for key, bounds in config.body_parameter_ranges.items():
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            errors.append(f"body_parameter_ranges.{key} must contain [min, max] with min < max")

    if errors:
        raise ValueError("; ".join(errors))
