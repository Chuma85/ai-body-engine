from __future__ import annotations

import csv
import hashlib
import json
import math
import random
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
SHAPE_KEY_COUPLED_LABEL_SOURCE = "shape_key_coupled_synthetic_formula"
LABEL_GENERATION_MODE = "shape_key_coupled_synthetic"
LABEL_FORMULA_VERSION = "shape_key_coupled_synthetic_v1"
DEFAULT_LABEL_NOISE_CM = 0.15
LEGACY_BLEND_LABEL_COLUMNS = [
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
TRACEABILITY_LABEL_COLUMNS = [
    "label_generation_mode",
    "height_factor",
    "chest_factor",
    "waist_factor",
    "hip_factor",
    "shoulder_factor",
    "inseam_factor",
    "torso_width_factor",
    "leg_length_factor",
    "shape_key_values_json",
    "body_shape_profile_id",
]
BODY_FACTOR_COLUMNS = [
    "height_factor",
    "chest_factor",
    "waist_factor",
    "hip_factor",
    "shoulder_factor",
    "inseam_factor",
    "torso_width_factor",
    "leg_length_factor",
]
BLEND_LABEL_COLUMNS = [*LEGACY_BLEND_LABEL_COLUMNS, *TRACEABILITY_LABEL_COLUMNS]
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
BASE_MEASUREMENT_PROFILE = {
    "height_cm": 178.0,
    "chest_cm": 102.0,
    "waist_cm": 88.0,
    "hip_cm": 104.0,
    "shoulder_cm": 46.0,
    "inseam_cm": 80.0,
}
BODY_FACTOR_DEFINITIONS = {
    "height_factor": "Stature-oriented factor derived from shape-key variation and shared body-size signal.",
    "chest_factor": "Upper torso breadth/fullness factor derived from shape-key variation.",
    "waist_factor": "Mid-torso width factor derived from shape-key variation.",
    "hip_factor": "Lower torso/pelvis width factor derived from shape-key variation.",
    "shoulder_factor": "Shoulder breadth factor derived from shape-key variation.",
    "inseam_factor": "Leg-length factor coupled to height and leg shape-key variation.",
    "torso_width_factor": "Shared torso-width signal used to keep chest, waist, hip, and shoulder positively co-varying.",
    "leg_length_factor": "Shared leg-length signal used to couple height and inseam.",
}
MEASUREMENT_FORMULA_SUMMARY = {
    "height_cm": "base + 10.0*height_factor + 4.0*leg_length_factor + deterministic noise",
    "chest_cm": "base + 8.0*chest_factor + 5.0*torso_width_factor + deterministic noise",
    "waist_cm": "base + 8.5*waist_factor + 4.5*torso_width_factor + deterministic noise",
    "hip_cm": "base + 8.0*hip_factor + 5.0*torso_width_factor + deterministic noise",
    "shoulder_cm": "base + 4.2*shoulder_factor + 2.5*torso_width_factor + deterministic noise",
    "inseam_cm": "base + 5.5*inseam_factor + 3.0*height_factor + deterministic noise",
}
SAFE_MEASUREMENT_RANGES = {
    "height_cm": (140.0, 220.0),
    "chest_cm": (60.0, 160.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (60.0, 170.0),
    "shoulder_cm": (25.0, 80.0),
    "inseam_cm": (50.0, 110.0),
}
FACTOR_KEYS = (
    "height",
    "torso_width",
    "chest",
    "waist",
    "hip",
    "shoulder",
    "leg_length",
)


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
        "label_generation": shape_key_label_metadata(seed),
    }


def shape_key_label_metadata(seed: int) -> dict[str, Any]:
    return {
        "label_generation_mode": LABEL_GENERATION_MODE,
        "label_formula_version": LABEL_FORMULA_VERSION,
        "synthetic_labels": True,
        "real_world_validated": False,
        "body_factor_definitions": BODY_FACTOR_DEFINITIONS,
        "base_measurement_profile": BASE_MEASUREMENT_PROFILE,
        "shape_key_to_factor_mapping": {
            "strategy": "deterministic_name_hash_weights_plus_shared_body_size_signal",
            "factor_keys": list(FACTOR_KEYS),
            "note": (
                "Each shape key contributes deterministic signed weights to interpretable body factors; "
                "labels are synthetic and derived from these same rendered shape-key values."
            ),
        },
        "measurement_formula_summary": MEASUREMENT_FORMULA_SUMMARY,
        "deterministic_seed": int(seed),
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
    label_noise_cm: float = DEFAULT_LABEL_NOISE_CM,
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
        "--label-noise-cm",
        str(label_noise_cm),
        "--front-camera",
        names["front"],
        "--side-camera",
        names["side"],
        "--back-camera",
        names["back"],
    ]


def shape_key_factor_weights(shape_key_name: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for factor in FACTOR_KEYS:
        digest = hashlib.sha256(f"{shape_key_name}:{factor}".encode("utf-8")).digest()
        unit_value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        weights[factor] = round(unit_value * 2.0 - 1.0, 6)
    lowered = shape_key_name.lower()
    if "$ma" in lowered:
        weights["shoulder"] += 0.45
        weights["chest"] += 0.25
        weights["hip"] -= 0.15
    if "$fe" in lowered:
        weights["hip"] += 0.35
        weights["waist"] += 0.15
    if "$af" in lowered:
        weights["height"] += 0.25
        weights["leg_length"] += 0.25
    if "$as" in lowered:
        weights["height"] -= 0.20
    if "$ca" in lowered:
        weights["torso_width"] += 0.20
    return {factor: _clamp(value, -1.0, 1.0) for factor, value in weights.items()}


def centered_shape_key_values(shape_key_values: dict[str, float], shape_key_range: float) -> dict[str, float]:
    if not shape_key_values:
        return {}
    safe_range = max(float(shape_key_range), 1e-6)
    values = [float(value) for value in shape_key_values.values()]
    min_value = min(values)
    max_value = max(values)
    all_nonnegative = min_value >= -1e-9 and max_value <= safe_range + 1e-9
    centered: dict[str, float] = {}
    for name, value in shape_key_values.items():
        numeric_value = float(value)
        if all_nonnegative:
            centered_value = (numeric_value / safe_range) * 2.0 - 1.0
        else:
            centered_value = numeric_value / safe_range
        centered[name] = _clamp(centered_value, -1.0, 1.0)
    return centered


def derive_body_factors_from_shape_keys(
    shape_key_values: dict[str, float],
    *,
    shape_key_range: float = DEFAULT_SHAPE_KEY_RANGE,
) -> dict[str, float]:
    centered = centered_shape_key_values(shape_key_values, shape_key_range)
    if not centered:
        return {column: 0.0 for column in BODY_FACTOR_COLUMNS}

    weighted_sums = {factor: 0.0 for factor in FACTOR_KEYS}
    weight_totals = {factor: 0.0 for factor in FACTOR_KEYS}
    for name, centered_value in sorted(centered.items()):
        weights = shape_key_factor_weights(name)
        for factor, weight in weights.items():
            weighted_sums[factor] += centered_value * weight
            weight_totals[factor] += abs(weight)

    raw = {
        factor: _clamp(weighted_sums[factor] / max(weight_totals[factor], 1e-6), -1.0, 1.0)
        for factor in FACTOR_KEYS
    }
    body_size_signal = _clamp(sum(centered.values()) / len(centered), -1.0, 1.0)
    torso_width_factor = _clamp(0.55 * raw["torso_width"] + 0.45 * body_size_signal, -1.0, 1.0)
    leg_length_factor = _clamp(0.60 * raw["leg_length"] + 0.40 * raw["height"], -1.0, 1.0)
    height_factor = _clamp(0.65 * raw["height"] + 0.20 * leg_length_factor + 0.15 * body_size_signal, -1.0, 1.0)
    chest_factor = _clamp(0.50 * raw["chest"] + 0.35 * torso_width_factor + 0.15 * body_size_signal, -1.0, 1.0)
    waist_factor = _clamp(0.50 * raw["waist"] + 0.35 * torso_width_factor + 0.15 * body_size_signal, -1.0, 1.0)
    hip_factor = _clamp(0.50 * raw["hip"] + 0.35 * torso_width_factor + 0.15 * body_size_signal, -1.0, 1.0)
    shoulder_factor = _clamp(0.55 * raw["shoulder"] + 0.35 * torso_width_factor + 0.10 * body_size_signal, -1.0, 1.0)
    inseam_factor = _clamp(0.65 * leg_length_factor + 0.35 * height_factor, -1.0, 1.0)
    return {
        "height_factor": round(height_factor, 6),
        "chest_factor": round(chest_factor, 6),
        "waist_factor": round(waist_factor, 6),
        "hip_factor": round(hip_factor, 6),
        "shoulder_factor": round(shoulder_factor, 6),
        "inseam_factor": round(inseam_factor, 6),
        "torso_width_factor": round(torso_width_factor, 6),
        "leg_length_factor": round(leg_length_factor, 6),
    }


def generate_shape_key_coupled_measurements(
    *,
    sample_id: str,
    seed: int,
    shape_key_values: dict[str, float],
    shape_key_range: float = DEFAULT_SHAPE_KEY_RANGE,
    label_noise_cm: float = DEFAULT_LABEL_NOISE_CM,
) -> dict[str, Any]:
    factors = derive_body_factors_from_shape_keys(shape_key_values, shape_key_range=shape_key_range)
    noise_rng = random.Random(f"{seed}:{sample_id}:{LABEL_FORMULA_VERSION}")
    noise = {
        target: noise_rng.uniform(-abs(float(label_noise_cm)), abs(float(label_noise_cm)))
        for target in BASE_MEASUREMENT_PROFILE
    }
    labels = {
        "height_cm": BASE_MEASUREMENT_PROFILE["height_cm"]
        + 10.0 * factors["height_factor"]
        + 4.0 * factors["leg_length_factor"]
        + noise["height_cm"],
        "chest_cm": BASE_MEASUREMENT_PROFILE["chest_cm"]
        + 8.0 * factors["chest_factor"]
        + 5.0 * factors["torso_width_factor"]
        + noise["chest_cm"],
        "waist_cm": BASE_MEASUREMENT_PROFILE["waist_cm"]
        + 8.5 * factors["waist_factor"]
        + 4.5 * factors["torso_width_factor"]
        + noise["waist_cm"],
        "hip_cm": BASE_MEASUREMENT_PROFILE["hip_cm"]
        + 8.0 * factors["hip_factor"]
        + 5.0 * factors["torso_width_factor"]
        + noise["hip_cm"],
        "shoulder_cm": BASE_MEASUREMENT_PROFILE["shoulder_cm"]
        + 4.2 * factors["shoulder_factor"]
        + 2.5 * factors["torso_width_factor"]
        + noise["shoulder_cm"],
        "inseam_cm": BASE_MEASUREMENT_PROFILE["inseam_cm"]
        + 5.5 * factors["inseam_factor"]
        + 3.0 * factors["height_factor"]
        + noise["inseam_cm"],
    }
    rounded_labels = {
        target: round(_clamp(value, *SAFE_MEASUREMENT_RANGES[target]), 1)
        for target, value in labels.items()
    }
    return {
        "measurements": rounded_labels,
        "factors": factors,
        "label_generation_mode": LABEL_GENERATION_MODE,
        "label_formula_version": LABEL_FORMULA_VERSION,
        "body_shape_profile_id": body_shape_profile_id(factors),
    }


def body_shape_profile_id(factors: dict[str, float]) -> str:
    height = "tall" if factors["height_factor"] > 0.25 else "short" if factors["height_factor"] < -0.25 else "midheight"
    torso = "wide_torso" if factors["torso_width_factor"] > 0.25 else "narrow_torso" if factors["torso_width_factor"] < -0.25 else "mid_torso"
    hip = "hip_dominant" if factors["hip_factor"] - factors["shoulder_factor"] > 0.25 else "balanced"
    return f"{height}_{torso}_{hip}"


def shape_key_traceability_json(shape_key_values: dict[str, float]) -> str:
    return json.dumps({name: round(float(value), 6) for name, value in sorted(shape_key_values.items())}, sort_keys=True, separators=(",", ":"))


def validate_shape_key_coupled_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["labels.csv has no rows to validate shape-key label coupling"]
    for row in rows:
        sample_id = row.get("sample_id", "<missing sample_id>")
        if row.get("label_generation_mode") != LABEL_GENERATION_MODE:
            errors.append(f"{sample_id}: label_generation_mode must be {LABEL_GENERATION_MODE}")
        if row.get("synthetic_labels") != "true":
            errors.append(f"{sample_id}: synthetic_labels must be true")
        if row.get("real_world_validated") != "false":
            errors.append(f"{sample_id}: real_world_validated must be false")
        if not row.get("shape_key_values_json"):
            errors.append(f"{sample_id}: shape_key_values_json is required")
        else:
            try:
                json.loads(row["shape_key_values_json"])
            except json.JSONDecodeError:
                errors.append(f"{sample_id}: shape_key_values_json is not valid JSON")
        for column in BODY_FACTOR_COLUMNS:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError):
                errors.append(f"{sample_id}: {column} must be numeric")
                continue
            if not -1.000001 <= value <= 1.000001:
                errors.append(f"{sample_id}: {column} out of expected [-1, 1] range")
        for target, (lower, upper) in SAFE_MEASUREMENT_RANGES.items():
            try:
                value = float(row.get(target, ""))
            except (TypeError, ValueError):
                errors.append(f"{sample_id}: {target} must be numeric")
                continue
            if value < lower or value > upper:
                errors.append(f"{sample_id}: {target}={value} outside safe range {lower}-{upper}")
    return errors


def validate_factor_label_correlations(rows: list[dict[str, str]], min_abs_correlation: float = 0.70) -> dict[str, Any]:
    target_factor_pairs = {
        "height_cm": "height_factor",
        "chest_cm": "chest_factor",
        "waist_cm": "waist_factor",
        "hip_cm": "hip_factor",
        "shoulder_cm": "shoulder_factor",
        "inseam_cm": "inseam_factor",
    }
    correlations: dict[str, float | None] = {}
    weak_targets: list[str] = []
    for target, factor in target_factor_pairs.items():
        target_values = _column_floats(rows, target)
        factor_values = _column_floats(rows, factor)
        correlation = _pearson(target_values, factor_values)
        correlations[target] = correlation
        if correlation is None or abs(correlation) < min_abs_correlation:
            weak_targets.append(target)
    return {
        "valid": not weak_targets,
        "min_abs_correlation": min_abs_correlation,
        "correlations": correlations,
        "weak_targets": weak_targets,
    }


def _column_floats(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row.get(column) not in (None, "")]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_delta))
    right_norm = math.sqrt(sum(value * value for value in right_delta))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / (left_norm * right_norm)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


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

    if any(row.get("label_generation_mode") for row in rows):
        result["errors"].extend(validate_shape_key_coupled_rows(rows))
        factor_correlation = validate_factor_label_correlations(rows, min_abs_correlation=0.60)
        result["factor_label_correlation"] = factor_correlation
        if not factor_correlation["valid"]:
            result["errors"].append(
                "weak factor-to-label coupling for targets: "
                + ", ".join(factor_correlation["weak_targets"])
            )

    result["valid"] = not result["errors"] and result["label_row_count"] > 0
    return result
