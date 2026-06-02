"""Procedural Blender body renderer.

This module remains safe to import in a regular Python environment because bpy
is imported only inside the runtime path. Phase 2C creates a simple mannequin
from Blender primitives; SMPL/SMPL-X, garments, and high-quality try-on
rendering are intentionally deferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.blender.utils.deformation_math import compute_shape_key_targets

GENERATOR_VERSION = "phase_2c_blender_procedural_body_v1"
BODY_SHAPES = ("slim", "average", "athletic", "curvy", "broad", "plus")
CANONICAL_FRONT_AXIS = "-Y"
CANONICAL_SIDE_VIEW_AXIS = "-X"
MINIMUM_SCAN_VIEWS = "front,side"
ENHANCED_SCAN_VIEWS = "front,side,back"
DEFAULT_RENDER_REALISM_CONTROLS = {
    "enabled": False,
    "background": {
        "brightness_range": [1.0, 1.0],
        "color_jitter": 0.0,
    },
    "lighting": {
        "strength_multiplier_range": [1.0, 1.0],
    },
    "camera": {
        "distance_jitter_range": [0.0, 0.0],
        "orthographic_scale_jitter_range": [1.0, 1.0],
        "lateral_offset_range": [0.0, 0.0],
        "vertical_offset_range": [0.0, 0.0],
    },
    "render_resolution": {
        "enabled": False,
        "image_width": None,
        "image_height": None,
    },
    "materials": {
        "skin_tone_brightness_range": [1.0, 1.0],
    },
}
HORIZONTAL_AXIS_ANGLES = {
    "+X": 0.0,
    "+Y": math.pi / 2,
    "-X": math.pi,
    "-Y": -math.pi / 2,
}
OPTIONAL_METADATA_COLUMNS = [
    "has_front",
    "has_side",
    "has_back",
    "capture_views",
    "minimum_scan_views",
    "enhanced_scan_views",
    "render_source",
    "is_smoke_dataset",
    "is_training_candidate",
    "quality_tier",
    "skin_tone_id",
    "pose_variation_degrees",
    "camera_distance",
    "camera_focal_length",
    "render_width",
    "render_height",
    "anatomy_version",
    "renderer_mode",
    "base_mesh_asset",
    "mesh_deformation_enabled",
    "fallback_used",
    "deformation_mode",
    "deformation_strength",
    "preserve_height",
    "auto_frame_body",
    "camera_margin",
    "rigging_enabled",
    "armature_detected",
    "shape_keys_detected",
    "deformation_applied",
    "shape_key_matches",
    "body_seed",
    "render_seed",
    "render_realism_enabled",
    "render_realism_version",
]
LABEL_COLUMNS = [
    "sample_id",
    "front_image_path",
    "side_image_path",
    "back_image_path",
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
    "body_shape",
    "generator_version",
    *OPTIONAL_METADATA_COLUMNS,
]


def main() -> None:
    args = parse_args()

    try:
        import bpy
    except ImportError:
        print("Blender bpy module is not available. Run this script with Blender.")
        return

    config = load_config(args.config)
    apply_cli_overrides(config, args)
    apply_render_realism_resolution_override(config)
    rng_seeds = resolved_rng_seeds(config)
    resolved_output_dir = resolve_output_dir(config["output_dir"])
    output_dirs = ensure_output_dirs(resolved_output_dir)
    body_rng = random.Random(rng_seeds["body_seed"])
    render_rng = random.Random(rng_seeds["render_seed"])
    start_index = args.start_index
    end_index = start_index + config["sample_count"]
    labels_path = output_dirs["labels"] / "labels.csv"
    reset_labels = not args.resume and not args.append_labels
    ensure_labels_csv(labels_path, reset=reset_labels)
    labeled_sample_ids = read_labeled_sample_ids(labels_path)
    completed_count = 0
    skipped_count = 0

    print(f"Resolved output dir: {resolved_output_dir}")
    print(f"Using body_seed={rng_seeds['body_seed']} render_seed={rng_seeds['render_seed']}")

    for skipped_index in range(1, start_index):
        generate_body_parameters(skipped_index, body_rng, config)

    for index in range(start_index, end_index):
        params = generate_body_parameters(index, body_rng, config)
        if config.get("anatomy", {}).get("enable_body_shape_adjustments", False):
            params = apply_body_shape_adjustments(params)
        view_paths = {
            view: output_dirs[view] / f"{params['sample_id']}_{view}.png"
            for view in configured_views(config)
        }
        front_path = view_paths["front"]
        side_path = view_paths["side"]
        back_path = view_paths.get("back")

        resume_action = resume_action_for_sample(params["sample_id"], front_path, side_path, labeled_sample_ids, back_path)
        if args.resume and resume_action in {"skip", "checkpoint_existing_pair"}:
            row = label_row_for_sample(params, config, front_path, side_path, resume_render_metadata(config), back_path=back_path)
            if resume_action == "checkpoint_existing_pair":
                append_label_row(labels_path, row)
                labeled_sample_ids.add(params["sample_id"])
                completed_count += 1
                print(f"Checkpointed existing rendered pair: {params['sample_id']}")
            else:
                skipped_count += 1
                print(f"Skipping completed sample: {params['sample_id']}")
            continue

        clear_scene(bpy)
        setup_world_background(bpy, config, render_rng)
        setup_render_quality(bpy, config)
        material = create_material(bpy, f"{params['sample_id']}_skin", adjusted_skin_tone(params["skin_tone"], config, render_rng))
        render_metadata = create_body_from_config(bpy, params, config, material)
        setup_lighting(bpy, config, render_rng)

        for view in configured_views(config):
            view_camera_jitter = camera_jitter(config, render_rng)
            camera = setup_camera(bpy, view, config["camera_distance"], config["camera_focal_length"], view_camera_jitter)
            if render_metadata.get("objects") and (config.get("camera") or {}).get("auto_frame_body", False):
                auto_frame_camera_to_objects(
                    bpy,
                    camera,
                    render_metadata["objects"],
                    view,
                    (config.get("camera") or {}).get("margin", 0.15),
                    config["camera_focal_length"],
                    config["image_width"],
                    config["image_height"],
                    config["camera_distance"],
                    view_camera_jitter,
                )
            render_view(bpy, view_paths[view], config["image_width"], config["image_height"])

        row = label_row_for_sample(params, config, front_path, side_path, render_metadata, back_path=back_path)
        if params["sample_id"] not in labeled_sample_ids:
            append_label_row(labels_path, row)
            labeled_sample_ids.add(params["sample_id"])
        completed_count += 1

    print(
        f"Rendered or checkpointed {completed_count} samples to {resolved_output_dir}; "
        f"skipped {skipped_count} completed samples."
    )


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if getattr(args, "output", None):
        config["output_dir"] = args.output
    if getattr(args, "num_samples", None) is not None:
        config["sample_count"] = args.num_samples
    if getattr(args, "body_seed", None) is not None:
        config["body_seed"] = args.body_seed
    if getattr(args, "render_seed", None) is not None:
        config["render_seed"] = args.render_seed


def resolved_rng_seeds(config: dict) -> dict[str, int]:
    legacy_seed = int(config.get("random_seed", 42))
    return {
        "body_seed": int(config.get("body_seed", legacy_seed)),
        "render_seed": int(config.get("render_seed", legacy_seed)),
        "legacy_random_seed": legacy_seed,
    }


def render_realism_controls(config: dict) -> dict:
    controls = _deep_copy_dict(DEFAULT_RENDER_REALISM_CONTROLS)
    user_controls = config.get("render_realism") or {}
    _deep_update(controls, user_controls)
    controls["enabled"] = bool(controls.get("enabled", False))
    validate_render_realism_controls(controls)
    return controls


def apply_render_realism_resolution_override(config: dict) -> None:
    controls = render_realism_controls(config)
    resolution = controls.get("render_resolution") or {}
    if not controls["enabled"] or not resolution.get("enabled", False):
        return
    width = resolution.get("image_width")
    height = resolution.get("image_height")
    if width is not None:
        config["image_width"] = int(width)
    if height is not None:
        config["image_height"] = int(height)


def validate_render_realism_controls(controls: dict) -> None:
    _validate_range(controls["background"]["brightness_range"], "render_realism.background.brightness_range", 0.1, 2.0)
    color_jitter = float(controls["background"].get("color_jitter", 0.0))
    if not 0.0 <= color_jitter <= 0.25:
        raise ValueError("render_realism.background.color_jitter must be between 0.0 and 0.25.")
    _validate_range(controls["lighting"]["strength_multiplier_range"], "render_realism.lighting.strength_multiplier_range", 0.25, 2.0)
    _validate_range(controls["camera"]["distance_jitter_range"], "render_realism.camera.distance_jitter_range", -0.5, 0.5)
    _validate_range(controls["camera"]["orthographic_scale_jitter_range"], "render_realism.camera.orthographic_scale_jitter_range", 0.9, 1.15)
    _validate_range(controls["camera"]["lateral_offset_range"], "render_realism.camera.lateral_offset_range", -0.12, 0.12)
    _validate_range(controls["camera"]["vertical_offset_range"], "render_realism.camera.vertical_offset_range", -0.12, 0.12)
    resolution = controls["render_resolution"]
    if resolution.get("enabled", False):
        if resolution.get("image_width") is not None and int(resolution["image_width"]) <= 0:
            raise ValueError("render_realism.render_resolution.image_width must be positive.")
        if resolution.get("image_height") is not None and int(resolution["image_height"]) <= 0:
            raise ValueError("render_realism.render_resolution.image_height must be positive.")
    _validate_range(controls["materials"]["skin_tone_brightness_range"], "render_realism.materials.skin_tone_brightness_range", 0.5, 1.5)


def _validate_range(bounds: list[float], label: str, minimum_allowed: float, maximum_allowed: float) -> None:
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(f"{label} must contain [min, max] with min <= max.")
    if bounds[0] < minimum_allowed or bounds[1] > maximum_allowed:
        raise ValueError(f"{label} must stay within safe bounds [{minimum_allowed}, {maximum_allowed}].")


def _deep_copy_dict(value: dict) -> dict:
    return json.loads(json.dumps(value))


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def configured_views(config: dict) -> list[str]:
    views = list(config.get("views") or ["front", "side"])
    if "front" not in views or "side" not in views:
        raise ValueError("Configured views must include front and side.")
    unsupported = sorted(set(views) - {"front", "side", "back"})
    if unsupported:
        raise ValueError(f"Unsupported configured views: {', '.join(unsupported)}")
    return views


def label_row_for_sample(
    params: dict,
    config: dict,
    front_path: Path,
    side_path: Path,
    render_metadata: dict,
    back_path: Path | None = None,
) -> dict:
    rng_seeds = resolved_rng_seeds(config)
    realism_controls = render_realism_controls(config)
    render_realism_config = config.get("render_realism") or {}
    capture_views = configured_views(config)
    has_back = back_path is not None
    return {
        "sample_id": params["sample_id"],
        "front_image_path": repo_relative_path(front_path),
        "side_image_path": repo_relative_path(side_path),
        "back_image_path": repo_relative_path(back_path) if back_path is not None else "",
        "has_front": True,
        "has_side": True,
        "has_back": has_back,
        "capture_views": ",".join(capture_views),
        "minimum_scan_views": MINIMUM_SCAN_VIEWS,
        "enhanced_scan_views": ENHANCED_SCAN_VIEWS,
        "render_source": "blender_body_mesh",
        "is_smoke_dataset": False,
        "is_training_candidate": True,
        "quality_tier": "training_candidate",
        "height_cm": params["height_cm"],
        "weight_kg": params["weight_kg"],
        "chest_cm": params["chest_cm"],
        "waist_cm": params["waist_cm"],
        "hip_cm": params["hip_cm"],
        "shoulder_cm": params["shoulder_cm"],
        "inseam_cm": params["inseam_cm"],
        "sleeve_cm": params["sleeve_cm"],
        "neck_cm": params["neck_cm"],
        "thigh_cm": params["thigh_cm"],
        "calf_cm": params["calf_cm"],
        "body_shape": params["body_shape"],
        "generator_version": config["generator_version"],
        "skin_tone_id": params["skin_tone_id"],
        "pose_variation_degrees": params["pose_variation_degrees"],
        "camera_distance": config["camera_distance"],
        "camera_focal_length": config["camera_focal_length"],
        "render_width": config["image_width"],
        "render_height": config["image_height"],
        "anatomy_version": config.get("generator_version", GENERATOR_VERSION),
        "renderer_mode": render_metadata["renderer_mode"],
        "base_mesh_asset": render_metadata["base_mesh_asset"],
        "mesh_deformation_enabled": render_metadata["mesh_deformation_enabled"],
        "fallback_used": render_metadata["fallback_used"],
        "deformation_mode": render_metadata["deformation_mode"],
        "deformation_strength": render_metadata["deformation_strength"],
        "preserve_height": render_metadata["preserve_height"],
        "auto_frame_body": render_metadata["auto_frame_body"],
        "camera_margin": render_metadata["camera_margin"],
        "rigging_enabled": render_metadata["rigging_enabled"],
        "armature_detected": render_metadata["armature_detected"],
        "shape_keys_detected": render_metadata["shape_keys_detected"],
        "deformation_applied": render_metadata["deformation_applied"],
        "shape_key_matches": render_metadata["shape_key_matches"],
        "body_seed": rng_seeds["body_seed"],
        "render_seed": rng_seeds["render_seed"],
        "render_realism_enabled": bool(realism_controls["enabled"]),
        "render_realism_version": render_realism_config.get("version", ""),
    }


def resume_render_metadata(config: dict) -> dict[str, object]:
    base_mesh = config.get("base_mesh") or {}
    mesh_deformation = config.get("mesh_deformation") or {}
    rigging = config.get("rigging") or {}
    return {
        "renderer_mode": "resume_existing_render",
        "base_mesh_asset": base_mesh.get("asset_path", ""),
        "mesh_deformation_enabled": bool(mesh_deformation.get("enabled", False)),
        "fallback_used": "",
        "deformation_mode": mesh_deformation.get("mode", ""),
        "deformation_strength": mesh_deformation.get("strength", ""),
        "preserve_height": bool(mesh_deformation.get("preserve_height", False)),
        "auto_frame_body": bool((config.get("camera") or {}).get("auto_frame_body", False)),
        "camera_margin": (config.get("camera") or {}).get("margin", ""),
        "rigging_enabled": bool(rigging.get("enabled", False)),
        "armature_detected": "",
        "shape_keys_detected": "",
        "deformation_applied": "",
        "shape_key_matches": "",
    }


def resume_action_for_sample(sample_id: str, front_path: Path, side_path: Path, labeled_sample_ids: set[str], back_path: Path | None = None) -> str:
    expected_paths = [front_path, side_path]
    if back_path is not None:
        expected_paths.append(back_path)
    if all(path.exists() for path in expected_paths):
        if sample_id in labeled_sample_ids:
            return "skip"
        return "checkpoint_existing_pair"
    return "render"


def resolve_output_dir(output_dir: str) -> Path:
    raw_output_dir = output_dir.strip()
    path = Path(raw_output_dir).expanduser()
    if path.is_absolute() or PureWindowsPath(raw_output_dir).is_absolute() or PurePosixPath(raw_output_dir).is_absolute():
        return path.resolve()

    return (repo_root() / path).resolve()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root()).as_posix()
    except ValueError:
        return os.path.relpath(resolved_path, repo_root()).replace(os.sep, "/")


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute() or PureWindowsPath(path_value).is_absolute() or PurePosixPath(path_value).is_absolute():
        return path.resolve()

    return (repo_root() / path).resolve()


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = resolve_output_dir(output_dir) if isinstance(output_dir, str) else output_dir.resolve()
    paths = {
        "front": root / "images" / "front",
        "side": root / "images" / "side",
        "back": root / "images" / "back",
        "labels": root / "labels",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def generate_body_parameters(index: int, rng: random.Random, config: dict) -> dict:
    ranges = config["body_parameter_ranges"]
    anatomy = config.get("anatomy", {})
    body_shape = select_body_shape(rng, config)
    ranges = ranges_for_body_shape(ranges, body_shape, config)
    skin_tones = config.get("materials", {}).get("skin_tones") or [[0.75, 0.55, 0.42, 1.0]]
    skin_tone_id = rng.randrange(len(skin_tones))
    pose_limit = anatomy.get("pose_variation_degrees", 0) if anatomy.get("enable_pose_variation", False) else 0
    params = {
        "sample_id": f"sample_{index:06d}",
        "body_shape": body_shape,
        "skin_tone": skin_tones[skin_tone_id],
        "skin_tone_id": skin_tone_id,
        "pose_variation_degrees": round(rng.uniform(-pose_limit, pose_limit), 2),
        "shoulder_rotation_degrees": round(rng.uniform(-pose_limit, pose_limit), 2),
        "arm_angle_degrees": round(rng.uniform(-pose_limit, pose_limit), 2),
        "leg_stance_degrees": round(rng.uniform(0, pose_limit), 2),
        "generator_version": config.get("generator_version", GENERATOR_VERSION),
    }

    for key, bounds in ranges.items():
        params[key] = round(rng.uniform(bounds[0], bounds[1]), 1)

    return params


def select_body_shape(rng: random.Random, config: dict) -> str:
    controls = config.get("variation_controls") or {}
    profiles = controls.get("body_shape_profiles") or {}
    if controls.get("enabled", False) and profiles:
        return rng.choice(sorted(profiles))
    return rng.choice(BODY_SHAPES)


def ranges_for_body_shape(base_ranges: dict, body_shape: str, config: dict) -> dict:
    controls = config.get("variation_controls") or {}
    profiles = controls.get("body_shape_profiles") or {}
    if not controls.get("enabled", False) or not controls.get("profile_range_overrides_enabled", False):
        return base_ranges

    profile = profiles.get(body_shape) or {}
    range_overrides = profile.get("body_parameter_ranges") or {}
    if not range_overrides:
        return base_ranges

    ranges = {key: [*bounds] for key, bounds in base_ranges.items()}
    for key, bounds in range_overrides.items():
        if key in ranges:
            ranges[key] = bounds
    return ranges


def apply_body_shape_adjustments(params: dict) -> dict:
    adjustments = {
        "slim": {"shoulder_scale": 0.94, "chest_scale": 0.92, "waist_scale": 0.88, "hip_scale": 0.92, "limb_scale": 0.88, "depth_scale": 0.90},
        "average": {"shoulder_scale": 1.0, "chest_scale": 1.0, "waist_scale": 1.0, "hip_scale": 1.0, "limb_scale": 1.0, "depth_scale": 1.0},
        "athletic": {"shoulder_scale": 1.12, "chest_scale": 1.08, "waist_scale": 0.94, "hip_scale": 1.0, "limb_scale": 1.06, "depth_scale": 1.04},
        "curvy": {"shoulder_scale": 1.0, "chest_scale": 1.06, "waist_scale": 0.95, "hip_scale": 1.14, "limb_scale": 1.03, "depth_scale": 1.08},
        "broad": {"shoulder_scale": 1.16, "chest_scale": 1.11, "waist_scale": 1.06, "hip_scale": 1.03, "limb_scale": 1.08, "depth_scale": 1.08},
        "plus": {"shoulder_scale": 1.08, "chest_scale": 1.18, "waist_scale": 1.22, "hip_scale": 1.20, "limb_scale": 1.14, "depth_scale": 1.20},
    }
    shape_adjustment = adjustments.get(params.get("body_shape"), adjustments["average"])
    adjusted = dict(params)
    adjusted.update(shape_adjustment)
    return adjusted


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def adjusted_skin_tone(rgba: list[float], config: dict, rng: random.Random) -> list[float]:
    controls = render_realism_controls(config)
    if not controls["enabled"]:
        return rgba
    bounds = controls["materials"]["skin_tone_brightness_range"]
    multiplier = rng.uniform(bounds[0], bounds[1])
    adjusted = [_clamp(channel * multiplier, 0.0, 1.0) for channel in rgba[:3]]
    alpha = rgba[3] if len(rgba) > 3 else 1.0
    return [*adjusted, alpha]


def create_material(bpy, name: str, rgba: list[float]):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.62
    return material


def create_ellipsoid(bpy, name: str, location: tuple[float, float, float], scale: tuple[float, float, float], material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    return obj


def create_cylinder_limb(
    bpy,
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material,
    rotation: tuple[float, float, float] = (0, 0, 0),
):
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def create_body_from_config(bpy, params: dict, config: dict, material) -> dict[str, object]:
    base_mesh = config.get("base_mesh") or {}
    mesh_deformation = config.get("mesh_deformation") or {}
    rigging = config.get("rigging") or {}
    metadata = {
        "renderer_mode": "procedural_fallback",
        "base_mesh_asset": "",
        "mesh_deformation_enabled": bool(mesh_deformation.get("enabled", False)),
        "fallback_used": False,
        "deformation_mode": mesh_deformation.get("mode", ""),
        "deformation_strength": mesh_deformation.get("strength", ""),
        "preserve_height": bool(mesh_deformation.get("preserve_height", False)),
        "auto_frame_body": bool((config.get("camera") or {}).get("auto_frame_body", False)),
        "camera_margin": (config.get("camera") or {}).get("margin", ""),
        "rigging_enabled": bool(rigging.get("enabled", False)),
        "armature_detected": False,
        "shape_keys_detected": False,
        "deformation_applied": False,
        "shape_key_matches": "",
        "objects": [],
    }

    if base_mesh.get("enabled", False):
        asset_path = resolve_repo_path(base_mesh.get("asset_path", ""))
        metadata["base_mesh_asset"] = repo_relative_path(asset_path)
        imported_objects = load_base_mesh_if_available(bpy, str(asset_path), base_mesh.get("format", asset_path.suffix.lstrip(".")))
        if not imported_objects and base_mesh.get("fallback_to_static_mesh", False):
            static_asset_path = repo_root() / "assets" / "body_meshes" / "base_human.obj"
            imported_objects = load_base_mesh_if_available(bpy, str(static_asset_path), "obj")
            if imported_objects:
                print(f"Warning: rigged mesh unavailable at {asset_path}. Using static mesh fallback {static_asset_path}.")
                metadata["base_mesh_asset"] = repo_relative_path(static_asset_path)
                metadata["fallback_used"] = True

        if imported_objects:
            normalize_imported_body_orientation(bpy, imported_objects, base_mesh)
            if base_mesh.get("normalize_scale", True):
                normalize_objects_height(bpy, imported_objects, _body_dimensions(params)["height"])
            if base_mesh.get("center_on_origin", True):
                center_objects_on_origin(bpy, imported_objects)
            if mesh_deformation.get("enabled", False):
                if mesh_deformation.get("mode") == "rigged_or_shape_key_v1" or rigging.get("enabled", False):
                    deformation_result = apply_rigged_mesh_deformation(bpy, imported_objects, params, config)
                else:
                    deformation_result = apply_region_scale_deformation(bpy, imported_objects, params, config)
                if mesh_deformation.get("preserve_height", False):
                    normalize_objects_height(bpy, imported_objects, _body_dimensions(params)["height"])
                if (config.get("camera") or {}).get("center_on_body", True):
                    center_objects_on_origin(bpy, imported_objects)
                metadata.update(deformation_result)
            apply_material_to_objects(bpy, imported_objects, material)
            if metadata["renderer_mode"] == "procedural_fallback":
                metadata["renderer_mode"] = "base_mesh"
            if metadata["deformation_mode"] not in {"safe_object_scale", "static_mesh_safe_scale"}:
                metadata["fallback_used"] = False
            metadata["objects"] = imported_objects
            return metadata

        if not base_mesh.get("fallback_to_procedural", True):
            raise FileNotFoundError(f"Base mesh asset not found or unsupported: {asset_path}")

        print(f"Warning: base mesh unavailable at {asset_path}. Falling back to procedural body.")
        metadata["fallback_used"] = True

    create_procedural_body(bpy, params, material, config)
    return metadata


def compute_region_scale_factors(params: dict) -> dict[str, float]:
    strength = params.get("deformation_strength", 0.35)
    references = {
        "shoulders": ("shoulder_cm", 45),
        "chest": ("chest_cm", 100),
        "waist": ("waist_cm", 82),
        "hips": ("hip_cm", 98),
        "legs": ("thigh_cm", 55),
        "calves": ("calf_cm", 38),
        "arms": ("sleeve_cm", 62),
        "inseam": ("inseam_cm", 80),
    }
    return {
        region: _clamp(1.0 + ((params.get(key, reference) - reference) / reference) * strength, 0.75, 1.35)
        for region, (key, reference) in references.items()
    }


def apply_region_scale_deformation(bpy, objects: list, params: dict, config: dict) -> dict[str, object]:
    mesh_deformation = config.get("mesh_deformation") or {}
    deformation_params = dict(params)
    deformation_params["deformation_strength"] = mesh_deformation.get("strength", 0.35)
    scale_factors = compute_region_scale_factors(deformation_params)
    regions = mesh_deformation.get("regions", {})
    bounds = get_object_bounds_world(objects)
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    height = max(max_z - min_z, 0.0001)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    for obj in objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        inverse_world = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            world_vertex = obj.matrix_world @ vertex.co
            z_ratio = (world_vertex.z - min_z) / height
            x_scale, y_scale, z_scale = _region_scales_for_vertex(z_ratio, world_vertex, center_x, center_y, scale_factors, regions)
            world_vertex.x = center_x + (world_vertex.x - center_x) * x_scale
            world_vertex.y = center_y + (world_vertex.y - center_y) * y_scale
            world_vertex.z = min_z + (world_vertex.z - min_z) * z_scale
            vertex.co = inverse_world @ world_vertex
        obj.data.update()

    bpy.context.view_layer.update()
    return {
        "deformation_mode": mesh_deformation.get("mode", "region_scale_v1"),
        "deformation_strength": mesh_deformation.get("strength", 0.35),
        "preserve_height": bool(mesh_deformation.get("preserve_height", True)),
        "deformation_applied": True,
    }


def detect_armatures(bpy) -> list:
    return [obj for obj in bpy.context.scene.objects if getattr(obj, "type", "") == "ARMATURE"]


def detect_mesh_objects(bpy) -> list:
    return [obj for obj in bpy.context.scene.objects if getattr(obj, "type", "") == "MESH"]


def detect_shape_keys(mesh_objects: list) -> dict[str, list[str]]:
    shape_keys: dict[str, list[str]] = {}

    for obj in mesh_objects:
        key_blocks = getattr(getattr(getattr(obj, "data", None), "shape_keys", None), "key_blocks", None)
        if not key_blocks:
            continue

        names = [key_block.name for key_block in key_blocks if key_block.name.lower() != "basis"]
        if names:
            shape_keys[obj.name] = names

    return shape_keys


def summarize_rigging(imported_objects: list) -> dict[str, object]:
    armatures = [obj for obj in imported_objects if getattr(obj, "type", "") == "ARMATURE"]
    mesh_objects = [obj for obj in imported_objects if getattr(obj, "type", "") == "MESH"]
    shape_keys = detect_shape_keys(mesh_objects)

    return {
        "armatures": armatures,
        "mesh_objects": mesh_objects,
        "shape_keys": shape_keys,
        "armature_detected": bool(armatures),
        "shape_keys_detected": any(shape_keys.values()),
    }


def apply_shape_key_deformation(mesh_objects: list, params: dict, config: dict) -> dict[str, object]:
    shape_key_mapping = config.get("shape_key_mapping") or {}
    targets = compute_shape_key_targets(params, shape_key_mapping)
    matches: list[str] = []

    for obj in mesh_objects:
        key_blocks = getattr(getattr(getattr(obj, "data", None), "shape_keys", None), "key_blocks", None)
        if not key_blocks:
            continue

        for category, aliases in shape_key_mapping.items():
            target_value = targets.get(category)
            if target_value is None:
                continue

            lowered_aliases = {alias.lower() for alias in aliases}
            for key_block in key_blocks:
                key_name = key_block.name.lower()
                if key_name == "basis":
                    continue
                if key_name in lowered_aliases or any(alias in key_name for alias in lowered_aliases):
                    key_block.value = target_value
                    matches.append(f"{obj.name}:{key_block.name}")

    return {
        "deformation_mode": "shape_keys",
        "deformation_applied": bool(matches),
        "shape_key_matches": ",".join(sorted(matches)),
        "fallback_used": False,
    }


def apply_bone_scale_deformation(bpy, armatures: list, params: dict, config: dict) -> dict[str, object]:
    mesh_deformation = config.get("mesh_deformation") or {}
    strength = mesh_deformation.get("strength", 0.30)
    factors = compute_region_scale_factors({**params, "deformation_strength": strength * 0.35})
    applied = False

    bone_regions = {
        "shoulder": "shoulders",
        "clavicle": "shoulders",
        "chest": "chest",
        "spine": "waist",
        "hips": "hips",
        "pelvis": "hips",
        "upper_arm": "arms",
        "arm": "arms",
        "thigh": "legs",
        "calf": "calves",
        "shin": "calves",
    }

    for armature in armatures:
        pose_bones = getattr(getattr(armature, "pose", None), "bones", [])
        for bone in pose_bones:
            bone_name = bone.name.lower()
            region = next((region for marker, region in bone_regions.items() if marker in bone_name), None)
            if not region:
                continue

            scale = _clamp(factors.get(region, 1.0), 0.92, 1.08)
            bone.scale = (scale, scale, bone.scale[2])
            applied = True

    bpy.context.view_layer.update()
    return {
        "deformation_mode": "bones",
        "deformation_applied": applied,
        "fallback_used": not applied,
        "shape_key_matches": "",
    }


def apply_safe_object_scale_fallback(bpy, objects: list, params: dict, config: dict) -> dict[str, object]:
    mesh_deformation = config.get("mesh_deformation") or {}
    strength = mesh_deformation.get("strength", 0.30)
    factors = compute_region_scale_factors({**params, "deformation_strength": strength})
    width_scale = _clamp((factors["shoulders"] + factors["chest"] + factors["hips"]) / 3, 0.88, 1.16)
    depth_scale = _clamp((factors["chest"] + factors["waist"] + factors["hips"]) / 3, 0.88, 1.16)
    applied = False

    for obj in objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        obj.scale = (obj.scale[0] * width_scale, obj.scale[1] * depth_scale, obj.scale[2])
        applied = True

    bpy.context.view_layer.update()
    return {
        "renderer_mode": "static_mesh_safe_scale",
        "deformation_mode": mesh_deformation.get("fallback_mode", "safe_object_scale"),
        "deformation_applied": applied,
        "fallback_used": True,
        "shape_key_matches": "",
    }


def apply_rigged_mesh_deformation(bpy, imported_objects: list, params: dict, config: dict) -> dict[str, object]:
    summary = summarize_rigging(imported_objects)
    metadata = {
        "rigging_enabled": bool((config.get("rigging") or {}).get("enabled", False)),
        "armature_detected": summary["armature_detected"],
        "shape_keys_detected": summary["shape_keys_detected"],
        "deformation_applied": False,
        "shape_key_matches": "",
    }

    if summary["shape_keys_detected"]:
        shape_result = apply_shape_key_deformation(summary["mesh_objects"], params, config)
        metadata.update(shape_result)
        if shape_result["deformation_applied"]:
            return metadata

    if summary["armature_detected"]:
        bone_result = apply_bone_scale_deformation(bpy, summary["armatures"], params, config)
        metadata.update(bone_result)
        if bone_result["deformation_applied"]:
            return metadata

    fallback_result = apply_safe_object_scale_fallback(bpy, summary["mesh_objects"], params, config)
    metadata.update(fallback_result)
    return metadata


def _region_scales_for_vertex(z_ratio: float, world_vertex, center_x: float, center_y: float, scale_factors: dict[str, float], regions: dict) -> tuple[float, float, float]:
    x_scale = y_scale = z_scale = 1.0
    side_distance = abs(world_vertex.x - center_x)
    front_distance = abs(world_vertex.y - center_y)

    if 0.70 <= z_ratio <= 0.86 and regions.get("shoulders", True):
        x_scale = max(x_scale, scale_factors["shoulders"])
    if 0.56 <= z_ratio < 0.74 and regions.get("chest", True):
        x_scale = max(x_scale, scale_factors["chest"])
        y_scale = max(y_scale, scale_factors["chest"])
    if 0.42 <= z_ratio < 0.58 and regions.get("waist", True):
        x_scale *= scale_factors["waist"]
        y_scale *= scale_factors["waist"]
    if 0.32 <= z_ratio < 0.46 and regions.get("hips", True):
        x_scale = max(x_scale, scale_factors["hips"])
        y_scale = max(y_scale, scale_factors["hips"])
    if z_ratio < 0.42 and regions.get("legs", True):
        x_scale *= scale_factors["legs"] if z_ratio > 0.20 else scale_factors["calves"]
        y_scale *= scale_factors["legs"] if z_ratio > 0.20 else scale_factors["calves"]
        z_scale = max(z_scale, scale_factors["inseam"])
    if 0.32 <= z_ratio <= 0.76 and regions.get("arms", True) and side_distance > front_distance * 1.4:
        x_scale *= scale_factors["arms"]
        y_scale *= scale_factors["arms"]

    return (_clamp(x_scale, 0.75, 1.35), _clamp(y_scale, 0.75, 1.35), _clamp(z_scale, 0.88, 1.12))


def load_base_mesh_if_available(bpy, asset_path: str, asset_format: str) -> list | None:
    path = Path(asset_path)
    if not path.exists():
        return None

    return import_mesh_asset(bpy, path, asset_format)


def import_mesh_asset(bpy, asset_path: Path, asset_format: str) -> list:
    asset_format = asset_format.lower().lstrip(".")
    before = set(bpy.context.scene.objects)

    if asset_format in {"glb", "gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(asset_path))
    elif asset_format == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(asset_path))
    elif asset_format == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(asset_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(asset_path))
    elif asset_format == "blend":
        print("Blend append/link import is not implemented in Phase 2E.")
        return []
    else:
        raise ValueError(f"Unsupported base mesh format: {asset_format}")

    imported_objects = [obj for obj in bpy.context.scene.objects if obj not in before]
    return imported_objects


def normalize_imported_body_orientation(bpy, imported_objects: list, base_mesh: dict) -> None:
    source_front_axis = base_mesh.get("source_front_axis", CANONICAL_FRONT_AXIS)
    source_yaw_degrees = base_mesh.get("source_yaw_degrees", 0.0)
    rotation_z = horizontal_axis_rotation(source_front_axis, CANONICAL_FRONT_AXIS) + math.radians(source_yaw_degrees)
    if abs(rotation_z) < 0.0001:
        return

    rotate_root_objects_around_world_z(bpy, imported_objects, rotation_z)


def horizontal_axis_rotation(source_axis: str, target_axis: str) -> float:
    source_angle = _horizontal_axis_angle(source_axis)
    target_angle = _horizontal_axis_angle(target_axis)
    return _normalize_radians(target_angle - source_angle)


def _horizontal_axis_angle(axis: str) -> float:
    normalized = axis.strip().upper()
    if normalized not in HORIZONTAL_AXIS_ANGLES:
        raise ValueError(f"Unsupported horizontal axis: {axis}")
    return HORIZONTAL_AXIS_ANGLES[normalized]


def _normalize_radians(angle: float) -> float:
    while angle <= -math.pi:
        angle += math.tau
    while angle > math.pi:
        angle -= math.tau
    return angle


def rotate_root_objects_around_world_z(bpy, objects: list, angle: float) -> None:
    object_set = set(objects)
    root_objects = [obj for obj in objects if getattr(obj, "parent", None) not in object_set]
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    for obj in root_objects:
        location = getattr(obj, "location", None)
        if location is not None:
            x = location.x
            y = location.y
            location.x = x * cos_angle - y * sin_angle
            location.y = x * sin_angle + y * cos_angle

        rotation_euler = getattr(obj, "rotation_euler", None)
        if rotation_euler is not None:
            rotation_euler.z += angle

    bpy.context.view_layer.update()


def normalize_imported_mesh(bpy, imported_objects: list, target_height_units: float) -> None:
    normalize_objects_height(bpy, imported_objects, target_height_units)


def normalize_objects_height(bpy, objects: list, target_height_units: float) -> None:
    min_x, max_x, min_y, max_y, min_z, max_z = get_object_bounds_world(objects)
    height = max_z - min_z
    if height <= 0:
        return

    scale_factor = target_height_units / height
    for obj in objects:
        obj.scale = (obj.scale[0] * scale_factor, obj.scale[1] * scale_factor, obj.scale[2] * scale_factor)
    bpy.context.view_layer.update()


def get_object_bounds_world(objects: list) -> tuple[float, float, float, float, float, float]:
    bounds = _object_bounds(objects)
    return (bounds["min_x"], bounds["max_x"], bounds["min_y"], bounds["max_y"], bounds["min_z"], bounds["max_z"])


def get_combined_mesh_vertices(objects: list) -> list:
    vertices = []
    for obj in objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        vertices.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
    return vertices


def _legacy_normalize_imported_mesh(bpy, imported_objects: list, target_height_units: float) -> None:
    bounds = _object_bounds(imported_objects)
    height = bounds["max_z"] - bounds["min_z"]
    if height <= 0:
        return

    scale_factor = target_height_units / height
    for obj in imported_objects:
        obj.scale = (obj.scale[0] * scale_factor, obj.scale[1] * scale_factor, obj.scale[2] * scale_factor)
    bpy.context.view_layer.update()


def center_objects_on_origin(bpy, imported_objects: list) -> None:
    bounds = _object_bounds(imported_objects)
    center_x = (bounds["min_x"] + bounds["max_x"]) / 2
    center_y = (bounds["min_y"] + bounds["max_y"]) / 2
    min_z = bounds["min_z"]

    for obj in imported_objects:
        obj.location.x -= center_x
        obj.location.y -= center_y
        obj.location.z -= min_z
    bpy.context.view_layer.update()


def apply_material_to_objects(bpy, imported_objects: list, material) -> None:
    for obj in imported_objects:
        if hasattr(obj, "data") and hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(material)


def _object_bounds(objects: list) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for obj in objects:
        for corner in getattr(obj, "bound_box", []):
            world_corner = obj.matrix_world @ __import__("mathutils").Vector(corner)
            xs.append(world_corner.x)
            ys.append(world_corner.y)
            zs.append(world_corner.z)

    if not xs:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}

    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }


def create_procedural_body(bpy, params: dict, material, config: dict | None = None) -> None:
    config = config or {}
    anatomy = config.get("anatomy", {})
    dims = _body_dimensions(params)
    head_z = dims["height"] - dims["head_radius"]
    neck_z = dims["height"] * 0.84
    shoulder_z = dims["height"] * 0.77
    upper_chest_z = dims["height"] * 0.68
    chest_z = dims["height"] * 0.61
    waist_z = dims["height"] * 0.50
    abdomen_z = dims["height"] * 0.45
    hip_z = dims["height"] * 0.39
    knee_z = dims["height"] * 0.21
    ankle_z = dims["height"] * 0.04
    pose_radians = _degrees_to_radians(params.get("pose_variation_degrees", 0))
    shoulder_tilt = _degrees_to_radians(params.get("shoulder_rotation_degrees", 0))
    arm_angle = _degrees_to_radians(params.get("arm_angle_degrees", 0))
    leg_stance = _degrees_to_radians(params.get("leg_stance_degrees", 0))

    create_ellipsoid(bpy, "head", (0, 0, head_z), (dims["head_radius"] * 0.82, dims["head_radius"] * 0.72, dims["head_radius"] * 1.04), material)
    create_cylinder_limb(bpy, "neck", (0, 0, neck_z), dims["neck_radius"], dims["neck_length"], material)
    create_ellipsoid(bpy, "shoulder_cap", (0, 0, shoulder_z), (dims["shoulder_width"], dims["depth"] * 0.82, 0.11), material)
    create_ellipsoid(bpy, "upper_chest", (0, 0, upper_chest_z), (dims["chest_width"] * 1.02, dims["depth"] * 0.96, 0.20), material)
    create_ellipsoid(bpy, "chest", (0, 0, chest_z), (dims["chest_width"], dims["depth"], 0.30), material)
    create_ellipsoid(bpy, "waist", (0, 0, waist_z), (dims["waist_width"], dims["depth"] * 0.78, 0.22), material)
    if anatomy.get("enable_torso_taper", True):
        create_ellipsoid(bpy, "abdomen_blend", (0, 0, abdomen_z), ((dims["waist_width"] + dims["hip_width"]) * 0.48, dims["depth"] * 0.88, 0.18), material)
    create_ellipsoid(bpy, "hips", (0, 0, hip_z), (dims["hip_width"], dims["depth"] * 1.07, 0.25), material)

    for side in (-1, 1):
        shoulder_x = side * (dims["shoulder_width"] + dims["arm_radius"] * 0.55)
        elbow_z = shoulder_z - dims["upper_arm_length"]
        wrist_z = elbow_z - dims["forearm_length"]
        upper_arm_radius = dims["arm_radius"]
        forearm_radius = dims["arm_radius"] * (0.78 if anatomy.get("enable_limb_taper", True) else 0.9)
        arm_x_offset = side * (0.05 + abs(arm_angle) * 0.16)
        arm_y_offset = pose_radians * 0.10
        create_cylinder_limb(
            bpy,
            f"{side}_upper_arm",
            (shoulder_x + arm_x_offset * 0.4, arm_y_offset, (shoulder_z + elbow_z) / 2),
            upper_arm_radius,
            dims["upper_arm_length"],
            material,
            rotation=(0.15 + arm_angle, side * 0.12 + shoulder_tilt, 0),
        )
        create_cylinder_limb(
            bpy,
            f"{side}_forearm",
            (shoulder_x + arm_x_offset, arm_y_offset * 1.2, (elbow_z + wrist_z) / 2),
            forearm_radius,
            dims["forearm_length"],
            material,
            rotation=(0.12 + arm_angle * 0.8, side * 0.09 + shoulder_tilt, 0),
        )
        create_ellipsoid(bpy, f"{side}_hand", (shoulder_x + side * 0.09 + arm_x_offset, 0, wrist_z - 0.04), (forearm_radius * 0.9, forearm_radius * 0.65, forearm_radius * 1.22), material)

    for side in (-1, 1):
        leg_x = side * (dims["leg_offset"] + leg_stance * 0.12)
        create_cylinder_limb(bpy, f"{side}_thigh", (leg_x, 0, (hip_z + knee_z) / 2), dims["thigh_radius"], hip_z - knee_z, material, rotation=(side * leg_stance, 0, 0))
        create_cylinder_limb(bpy, f"{side}_calf", (leg_x + side * leg_stance * 0.04, 0, (knee_z + ankle_z) / 2), dims["calf_radius"], knee_z - ankle_z, material, rotation=(side * leg_stance * 0.55, 0, 0))
        create_ellipsoid(bpy, f"{side}_foot", (leg_x + side * 0.04, -0.09, 0.03), (dims["calf_radius"] * 1.35, dims["calf_radius"] * 2.1, 0.05), material)


def setup_camera(bpy, view: str, camera_distance: float, focal_length: float, jitter: dict | None = None):
    jitter = jitter or default_camera_jitter()
    center = camera_center_with_jitter(view, (0, 0, 1.45), jitter)
    location, rotation = camera_transform_for_view(
        view,
        center,
        max(0.1, camera_distance + jitter["distance_delta"]),
    )
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.data.lens = focal_length
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.6 * jitter["orthographic_scale_multiplier"]
    bpy.context.scene.camera = camera
    return camera


def camera_transform_for_view(
    view: str,
    center: tuple[float, float, float],
    distance: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center_x, center_y, center_z = center
    if view == "front":
        return (center_x, center_y - distance, center_z), (math.pi / 2, 0, 0)
    if view == "side":
        return (center_x - distance, center_y, center_z), (math.pi / 2, 0, -math.pi / 2)
    if view == "back":
        return (center_x, center_y + distance, center_z), (math.pi / 2, 0, math.pi)
    raise ValueError(f"Unsupported view: {view}")


def camera_center_with_jitter(view: str, center: tuple[float, float, float], jitter: dict[str, float]) -> tuple[float, float, float]:
    center_x, center_y, center_z = center
    if view in {"front", "back"}:
        return (center_x + jitter["lateral_offset"], center_y, center_z + jitter["vertical_offset"])
    if view == "side":
        return (center_x, center_y + jitter["lateral_offset"], center_z + jitter["vertical_offset"])
    raise ValueError(f"Unsupported view: {view}")


def camera_frame_dimensions(bounds: tuple[float, float, float, float, float, float], view: str) -> tuple[float, float, float]:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    height = max(max_z - min_z, 0.1)
    if view in {"front", "back"}:
        return max(max_x - min_x, 0.1), height, max(max_y - min_y, 0.1)
    if view == "side":
        return max(max_y - min_y, 0.1), height, max(max_x - min_x, 0.1)
    raise ValueError(f"Unsupported view: {view}")


def orthographic_scale_for_frame(frame_width: float, frame_height: float, aspect_ratio: float, margin: float) -> float:
    return max(frame_height, frame_width / max(aspect_ratio, 0.01)) * (1.0 + margin)


def default_camera_jitter() -> dict[str, float]:
    return {
        "distance_delta": 0.0,
        "orthographic_scale_multiplier": 1.0,
        "lateral_offset": 0.0,
        "vertical_offset": 0.0,
    }


def camera_jitter(config: dict, rng: random.Random) -> dict[str, float]:
    controls = render_realism_controls(config)
    if not controls["enabled"]:
        return default_camera_jitter()
    camera_controls = controls["camera"]
    return {
        "distance_delta": rng.uniform(*camera_controls["distance_jitter_range"]),
        "orthographic_scale_multiplier": rng.uniform(*camera_controls["orthographic_scale_jitter_range"]),
        "lateral_offset": rng.uniform(*camera_controls["lateral_offset_range"]),
        "vertical_offset": rng.uniform(*camera_controls["vertical_offset_range"]),
    }


def auto_frame_camera_to_objects(
    bpy,
    camera,
    objects: list,
    view: str,
    margin: float,
    focal_length: float,
    image_width: int | None = None,
    image_height: int | None = None,
    minimum_distance: float = 2.5,
    jitter: dict | None = None,
) -> None:
    jitter = jitter or default_camera_jitter()
    min_x, max_x, min_y, max_y, min_z, max_z = get_object_bounds_world(objects)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    frame_width, frame_height, frame_depth = camera_frame_dimensions((min_x, max_x, min_y, max_y, min_z, max_z), view)
    aspect_ratio = (image_width or 1) / max(image_height or 1, 1)
    distance = max(minimum_distance, frame_depth * 3.0 + 1.0 + jitter["distance_delta"])
    jittered_center = camera_center_with_jitter(view, (center_x, center_y, center_z), jitter)
    camera.location, camera.rotation_euler = camera_transform_for_view(view, jittered_center, distance)

    camera.data.lens = focal_length
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = orthographic_scale_for_frame(frame_width, frame_height, aspect_ratio, margin) * jitter["orthographic_scale_multiplier"]
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()


def setup_lighting(bpy, config: dict, rng: random.Random) -> None:
    randomize = config.get("lighting", {}).get("randomize_strength", False)
    strength = 650 + (rng.uniform(-120, 120) if randomize else 0)
    controls = render_realism_controls(config)
    if controls["enabled"]:
        strength *= rng.uniform(*controls["lighting"]["strength_multiplier_range"])

    for name, location, energy in (
        ("key_light", (-2.5, -3.0, 4.0), strength),
        ("fill_light", (3.0, -2.2, 3.0), strength * 0.35),
        ("rim_light", (0.0, 3.0, 3.5), strength * 0.45),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = 4


def setup_world_background(bpy, config: dict, rng: random.Random | None = None) -> None:
    color = config.get("background", {}).get("color", [1, 1, 1])
    if rng is not None:
        controls = render_realism_controls(config)
        if controls["enabled"]:
            background = controls["background"]
            brightness = rng.uniform(*background["brightness_range"])
            jitter = float(background.get("color_jitter", 0.0))
            color = [
                _clamp(channel * brightness + rng.uniform(-jitter, jitter), 0.0, 1.0)
                for channel in color[:3]
            ]
    bpy.context.scene.world.color = (color[0], color[1], color[2])
    render_engine = config.get("render_engine", "BLENDER_EEVEE_NEXT")
    try:
        bpy.context.scene.render.engine = render_engine
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"


def setup_render_quality(bpy, config: dict) -> None:
    quality = config.get("render_quality", {})
    bpy.context.scene.render.resolution_percentage = quality.get("resolution_percentage", 100)

    if hasattr(bpy.context.scene, "eevee"):
        if "ambient_occlusion" in quality and hasattr(bpy.context.scene.eevee, "use_gtao"):
            bpy.context.scene.eevee.use_gtao = bool(quality["ambient_occlusion"])
        if "contact_shadows" in quality:
            for light in bpy.data.lights:
                if hasattr(light, "use_shadow"):
                    light.use_shadow = bool(quality["contact_shadows"])


def render_view(bpy, output_path: Path, width: int, height: int) -> None:
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = str(output_path)
    print(f"Rendering: {output_path}")
    bpy.ops.render.render(write_still=True)


def write_labels_csv(rows: list[dict], labels_csv_path: Path, append: bool = False) -> None:
    ensure_labels_csv(labels_csv_path, reset=not append)
    with labels_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
        writer.writerows(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def ensure_labels_csv(labels_csv_path: Path, reset: bool = False) -> None:
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if reset or not labels_csv_path.exists() or labels_csv_path.stat().st_size == 0:
        with labels_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            csv_file.flush()
            os.fsync(csv_file.fileno())


def append_label_row(labels_csv_path: Path, row: dict) -> None:
    ensure_labels_csv(labels_csv_path, reset=False)
    with labels_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def read_labeled_sample_ids(labels_csv_path: Path) -> set[str]:
    if not labels_csv_path.exists() or labels_csv_path.stat().st_size == 0:
        return set()

    with labels_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {row["sample_id"] for row in reader if row.get("sample_id")}


def _body_dimensions(params: dict) -> dict[str, float]:
    height = _scale(params["height_cm"], 150, 205, 2.65, 3.25)
    shoulder_width = _scale(params["shoulder_cm"], 35, 60, 0.34, 0.56) * params.get("shoulder_scale", 1.0)
    chest_width = _scale(params["chest_cm"], 75, 130, 0.30, 0.53) * params.get("chest_scale", 1.0)
    waist_width = _scale(params["waist_cm"], 55, 125, 0.22, 0.48) * params.get("waist_scale", 1.0)
    hip_width = _scale(params["hip_cm"], 75, 135, 0.32, 0.56) * params.get("hip_scale", 1.0)
    depth = _scale((params["chest_cm"] + params["waist_cm"] + params["hip_cm"]) / 3, 68, 130, 0.16, 0.33) * params.get("depth_scale", 1.0)
    inseam = _scale(params["inseam_cm"], 65, 95, 1.20, 1.62)
    sleeve = _scale(params["sleeve_cm"], 50, 75, 0.82, 1.12)
    limb_scale = params.get("limb_scale", 1.0)
    thigh_radius = _scale(params["thigh_cm"], 40, 80, 0.075, 0.15) * limb_scale
    calf_radius = _scale(params["calf_cm"], 28, 55, 0.055, 0.11) * limb_scale

    return {
        "height": height,
        "head_radius": height * 0.055,
        "neck_radius": _scale(params["neck_cm"], 30, 50, 0.045, 0.078),
        "neck_length": height * 0.07,
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "hip_width": hip_width,
        "depth": depth,
        "arm_radius": _scale(params["sleeve_cm"], 50, 75, 0.052, 0.078) * limb_scale,
        "upper_arm_length": sleeve * 0.48,
        "forearm_length": sleeve * 0.43,
        "thigh_radius": thigh_radius,
        "calf_radius": calf_radius,
        "leg_offset": max(0.09, hip_width * 0.38),
        "upper_leg_depth": inseam * 0.52,
        "lower_leg_depth": inseam * 0.48,
    }


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    ratio = (value - source_min) / (source_max - source_min)
    ratio = max(0.0, min(1.0, ratio))
    return target_min + ratio * (target_max - target_min)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _degrees_to_radians(degrees: float) -> float:
    return degrees * 0.017453292519943295


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic parametric body dataset with Blender.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--append-labels", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--body-seed", type=int)
    parser.add_argument("--render-seed", type=int)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.start_index < 1:
        parser.error("--start-index must be at least 1")
    return args


if __name__ == "__main__":
    main()
