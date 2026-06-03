from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


GENERATOR_VERSION = "phase_3h_blend_dataset_v1"
DEFAULT_SOURCE_MODE = "blend"
DEFAULT_BLEND_FILE = "assets/body_meshes/base_body_scene.blend"
CANONICAL_BLEND_FILE = "assets/body-ai/blender/base_body_scene.blend"
DEFAULT_OUTPUT_DIR = "data/synthetic/phase_3h_blend"
DEFAULT_BLENDER_SCRIPT = "synthetic/blender/scripts/render_blend_dataset.py"
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SEED = 42
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 896
DEFAULT_SHAPE_KEY_RANGE = 0.15
DEFAULT_POSE_VARIATION_DEGREES = 0.0
DEFAULT_CAMERA_NAMES = {
    "front": "FrontCam",
    "side": "SideCam",
    "back": "BackCam",
}
CAMERA_VIEWS = ("front", "side", "back")
STATIC_BLEND_WARNING = (
    "TODO: true body shape variation requires shape keys, parametric mesh controls, "
    "or multiple body meshes; this .blend currently renders as a static blend mesh."
)
SYNTHETIC_LABEL_SOURCE = "existing_synthetic_label_generator"
BLEND_LABEL_COLUMNS = [
    "sample_id",
    "front_image",
    "side_image",
    "back_image",
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
    "source_blend_file",
    "variation_source",
    "camera_set",
    "seed",
    "label_source",
    "synthetic_labels",
    "real_world_validated",
]
BODY_PARAMETER_RANGES = {
    "height_cm": [150, 205],
    "weight_kg": [45, 130],
    "chest_cm": [75, 130],
    "waist_cm": [55, 125],
    "hip_cm": [75, 135],
    "shoulder_cm": [35, 60],
    "inseam_cm": [65, 95],
    "sleeve_cm": [50, 75],
    "neck_cm": [30, 50],
    "thigh_cm": [40, 80],
    "calf_cm": [28, 55],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_value: str | Path) -> Path:
    raw_value = str(path_value)
    path = Path(raw_value).expanduser()
    if path.is_absolute() or PureWindowsPath(raw_value).is_absolute() or PurePosixPath(raw_value).is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def repo_relative_path(path: str | Path) -> str:
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(resolved_path)


def camera_set_name(camera_names: dict[str, str] | None = None) -> str:
    names = camera_names or DEFAULT_CAMERA_NAMES
    return ",".join(names[view] for view in CAMERA_VIEWS)


def blend_generation_config(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    samples: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "output_dir": str(output_dir),
        "sample_count": int(samples),
        "random_seed": int(seed),
        "body_seed": int(seed),
        "render_seed": int(seed),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "body_parameter_ranges": BODY_PARAMETER_RANGES,
        "materials": {
            "skin_tones": [
                [0.75, 0.56, 0.42, 1.0],
                [0.58, 0.38, 0.26, 1.0],
                [0.88, 0.68, 0.52, 1.0],
            ],
        },
        "variation_controls": {"enabled": False},
        "anatomy": {"enable_pose_variation": False, "pose_variation_degrees": 0.0},
    }


def build_blend_blender_command(
    *,
    blender_executable: str,
    script_path: str,
    blend_file: str,
    output_dir: str,
    samples: int,
    seed: int,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    shape_key_range: float = DEFAULT_SHAPE_KEY_RANGE,
    pose_variation_degrees: float = DEFAULT_POSE_VARIATION_DEGREES,
    camera_names: dict[str, str] | None = None,
) -> list[str]:
    names = camera_names or DEFAULT_CAMERA_NAMES
    return [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python",
        script_path,
        "--",
        "--source",
        DEFAULT_SOURCE_MODE,
        "--blend-file",
        blend_file,
        "--out",
        output_dir,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--image-width",
        str(image_width),
        "--image-height",
        str(image_height),
        "--shape-key-range",
        str(shape_key_range),
        "--pose-variation-degrees",
        str(pose_variation_degrees),
        "--front-camera",
        names["front"],
        "--side-camera",
        names["side"],
        "--back-camera",
        names["back"],
    ]


def validate_blend_file_exists(blend_file: str | Path) -> Path:
    blend_path = resolve_repo_path(blend_file)
    if not blend_path.exists():
        raise FileNotFoundError(
            f"Missing Blender .blend file: {blend_file}. Place the master scene at "
            f"{DEFAULT_BLEND_FILE} or pass --blend-file {CANONICAL_BLEND_FILE}."
        )
    if blend_path.suffix.lower() != ".blend":
        raise ValueError(f"Expected a .blend file, got: {blend_path}")
    return blend_path


def validate_generated_blend_dataset(dataset_root: str | Path, expected_samples: int | None = None) -> dict[str, Any]:
    root = Path(dataset_root)
    labels_path = root / "labels.csv"
    metadata_path = root / "metadata.json"
    images_dir = root / "images"
    result: dict[str, Any] = {
        "valid": False,
        "dataset": str(root),
        "expected_samples": expected_samples,
        "label_row_count": 0,
        "errors": [],
        "warnings": [],
        "missing_paths": [],
        "missing_image_paths": [],
    }

    for path in (root, images_dir, labels_path, metadata_path):
        if not path.exists():
            result["missing_paths"].append(str(path))
            result["errors"].append(f"missing path: {path}")

    if result["errors"]:
        return result

    try:
        with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
            reader = csv.DictReader(labels_file)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
    except csv.Error as error:
        result["errors"].append(f"labels.csv parse error: {error}")
        return result

    missing_columns = [column for column in BLEND_LABEL_COLUMNS if column not in fieldnames]
    if missing_columns:
        result["errors"].append(f"labels.csv missing columns: {', '.join(missing_columns)}")

    result["label_row_count"] = len(rows)
    if expected_samples is not None and len(rows) != expected_samples:
        result["errors"].append(f"labels.csv row count {len(rows)} does not match expected samples {expected_samples}")

    for row in rows:
        sample_id = row.get("sample_id", "<missing sample_id>")
        for column in ("front_image", "side_image", "back_image"):
            image_value = row.get(column, "")
            image_path = root / image_value
            if not image_value or not image_path.exists():
                missing = f"{sample_id}:{column}:{image_value}"
                result["missing_image_paths"].append(missing)
                result["errors"].append(f"missing image for {missing}")
        if row.get("synthetic_labels") != "true":
            result["errors"].append(f"{sample_id}: synthetic_labels must be true")
        if row.get("real_world_validated") != "false":
            result["errors"].append(f"{sample_id}: real_world_validated must be false")

    result["valid"] = not result["errors"] and result["label_row_count"] > 0
    return result
