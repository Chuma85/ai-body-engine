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
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import sys

GENERATOR_VERSION = "phase_2c_blender_procedural_body_v1"
BODY_SHAPES = ("slim", "average", "athletic", "curvy", "broad", "plus")
OPTIONAL_METADATA_COLUMNS = [
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
]
LABEL_COLUMNS = [
    "sample_id",
    "front_image_path",
    "side_image_path",
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
    resolved_output_dir = resolve_output_dir(config["output_dir"])
    output_dirs = ensure_output_dirs(resolved_output_dir)
    rng = random.Random(config["random_seed"])
    rows = []

    print(f"Resolved output dir: {resolved_output_dir}")

    for index in range(1, config["sample_count"] + 1):
        params = generate_body_parameters(index, rng, config)
        if config.get("anatomy", {}).get("enable_body_shape_adjustments", False):
            params = apply_body_shape_adjustments(params)
        clear_scene(bpy)
        setup_world_background(bpy, config)
        setup_render_quality(bpy, config)
        material = create_material(bpy, f"{params['sample_id']}_skin", params["skin_tone"])
        render_metadata = create_body_from_config(bpy, params, config, material)
        setup_lighting(bpy, config, rng)

        front_path = output_dirs["front"] / f"{params['sample_id']}_front.png"
        side_path = output_dirs["side"] / f"{params['sample_id']}_side.png"

        setup_camera(bpy, "front", config["camera_distance"], config["camera_focal_length"])
        render_view(bpy, front_path, config["image_width"], config["image_height"])

        setup_camera(bpy, "side", config["camera_distance"], config["camera_focal_length"])
        render_view(bpy, side_path, config["image_width"], config["image_height"])

        rows.append(
            {
                "sample_id": params["sample_id"],
                "front_image_path": repo_relative_path(front_path),
                "side_image_path": repo_relative_path(side_path),
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
            }
        )

    write_labels_csv(rows, output_dirs["labels"] / "labels.csv")
    print(f"Rendered {len(rows)} procedural body samples to {resolved_output_dir}")


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


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
        "labels": root / "labels",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def generate_body_parameters(index: int, rng: random.Random, config: dict) -> dict:
    ranges = config["body_parameter_ranges"]
    anatomy = config.get("anatomy", {})
    body_shape = rng.choice(BODY_SHAPES)
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
    metadata = {
        "renderer_mode": "procedural_fallback",
        "base_mesh_asset": "",
        "mesh_deformation_enabled": bool(mesh_deformation.get("enabled", False)),
        "fallback_used": False,
    }

    if base_mesh.get("enabled", False):
        asset_path = resolve_repo_path(base_mesh.get("asset_path", ""))
        metadata["base_mesh_asset"] = repo_relative_path(asset_path)
        imported_objects = load_base_mesh_if_available(bpy, str(asset_path), base_mesh.get("format", asset_path.suffix.lstrip(".")))

        if imported_objects:
            if base_mesh.get("normalize_scale", True):
                normalize_imported_mesh(bpy, imported_objects, _body_dimensions(params)["height"])
            if base_mesh.get("center_on_origin", True):
                center_objects_on_origin(bpy, imported_objects)
            apply_material_to_objects(bpy, imported_objects, material)
            metadata["renderer_mode"] = "base_mesh"
            metadata["fallback_used"] = False
            return metadata

        if not base_mesh.get("fallback_to_procedural", True):
            raise FileNotFoundError(f"Base mesh asset not found or unsupported: {asset_path}")

        print(f"Warning: base mesh unavailable at {asset_path}. Falling back to procedural body.")
        metadata["fallback_used"] = True

    create_procedural_body(bpy, params, material, config)
    return metadata


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


def normalize_imported_mesh(bpy, imported_objects: list, target_height_units: float) -> None:
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


def setup_camera(bpy, view: str, camera_distance: float, focal_length: float) -> None:
    if view == "front":
        location = (0, -camera_distance, 1.45)
        rotation = (1.5708, 0, 0)
    elif view == "side":
        location = (camera_distance, 0, 1.45)
        rotation = (1.5708, 0, 1.5708)
    else:
        raise ValueError(f"Unsupported view: {view}")

    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.data.lens = focal_length
    bpy.context.scene.camera = camera


def setup_lighting(bpy, config: dict, rng: random.Random) -> None:
    randomize = config.get("lighting", {}).get("randomize_strength", False)
    strength = 650 + (rng.uniform(-120, 120) if randomize else 0)

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


def setup_world_background(bpy, config: dict) -> None:
    color = config.get("background", {}).get("color", [1, 1, 1])
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


def write_labels_csv(rows: list[dict], labels_csv_path: Path) -> None:
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


def _degrees_to_radians(degrees: float) -> float:
    return degrees * 0.017453292519943295


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic parametric body dataset with Blender.")
    parser.add_argument("--config", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
