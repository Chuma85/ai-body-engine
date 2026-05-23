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
        clear_scene(bpy)
        setup_world_background(bpy, config)
        material = create_material(bpy, f"{params['sample_id']}_skin", params["skin_tone"])
        create_procedural_body(bpy, params, material)
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
    body_shape = rng.choice(BODY_SHAPES)
    skin_tones = config.get("materials", {}).get("skin_tones") or [[0.75, 0.55, 0.42, 1.0]]
    params = {
        "sample_id": f"sample_{index:06d}",
        "body_shape": body_shape,
        "skin_tone": rng.choice(skin_tones),
        "generator_version": config.get("generator_version", GENERATOR_VERSION),
    }

    for key, bounds in ranges.items():
        params[key] = round(rng.uniform(bounds[0], bounds[1]), 1)

    return params


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
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def create_procedural_body(bpy, params: dict, material) -> None:
    dims = _body_dimensions(params)
    head_z = dims["height"] - dims["head_radius"]
    shoulder_z = dims["height"] * 0.77
    chest_z = dims["height"] * 0.63
    waist_z = dims["height"] * 0.50
    hip_z = dims["height"] * 0.42
    thigh_z = dims["height"] * 0.26
    calf_z = dims["height"] * 0.10

    create_ellipsoid(bpy, "head", (0, 0, head_z), (dims["head_radius"] * 0.82, dims["head_radius"] * 0.72, dims["head_radius"]), material)
    create_cylinder_limb(bpy, "neck", (0, 0, shoulder_z + 0.13), dims["neck_radius"], 0.25, material)
    create_ellipsoid(bpy, "chest", (0, 0, chest_z), (dims["chest_width"], dims["depth"], 0.34), material)
    create_ellipsoid(bpy, "waist", (0, 0, waist_z), (dims["waist_width"], dims["depth"] * 0.82, 0.24), material)
    create_ellipsoid(bpy, "hips", (0, 0, hip_z), (dims["hip_width"], dims["depth"] * 1.05, 0.24), material)

    arm_rotation = (0.18, 0.18, 0)
    for side in (-1, 1):
        shoulder_x = side * (dims["shoulder_width"] + dims["arm_radius"] * 0.35)
        elbow_z = dims["height"] * 0.52
        wrist_z = dims["height"] * 0.34
        create_cylinder_limb(bpy, f"{side}_upper_arm", (shoulder_x, 0, (shoulder_z + elbow_z) / 2), dims["arm_radius"], shoulder_z - elbow_z, material, rotation=arm_rotation)
        create_cylinder_limb(bpy, f"{side}_forearm", (shoulder_x + side * 0.04, 0, (elbow_z + wrist_z) / 2), dims["arm_radius"] * 0.82, elbow_z - wrist_z, material, rotation=arm_rotation)
        create_ellipsoid(bpy, f"{side}_hand", (shoulder_x + side * 0.07, 0, wrist_z - 0.04), (dims["arm_radius"] * 0.9, dims["arm_radius"] * 0.7, dims["arm_radius"] * 1.15), material)

    for side in (-1, 1):
        leg_x = side * dims["leg_offset"]
        create_cylinder_limb(bpy, f"{side}_thigh", (leg_x, 0, thigh_z), dims["thigh_radius"], dims["upper_leg_depth"], material)
        create_cylinder_limb(bpy, f"{side}_calf", (leg_x, 0, calf_z), dims["calf_radius"], dims["lower_leg_depth"], material)
        create_ellipsoid(bpy, f"{side}_foot", (leg_x + side * 0.03, -0.08, 0.03), (dims["calf_radius"] * 1.3, dims["calf_radius"] * 2.0, 0.05), material)


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
    shoulder_width = _scale(params["shoulder_cm"], 35, 60, 0.34, 0.56)
    chest_width = _scale(params["chest_cm"], 75, 130, 0.30, 0.53)
    waist_width = _scale(params["waist_cm"], 55, 125, 0.22, 0.48)
    hip_width = _scale(params["hip_cm"], 75, 135, 0.32, 0.56)
    depth = _scale((params["chest_cm"] + params["waist_cm"] + params["hip_cm"]) / 3, 68, 130, 0.16, 0.33)
    inseam = _scale(params["inseam_cm"], 65, 95, 1.20, 1.62)
    thigh_radius = _scale(params["thigh_cm"], 40, 80, 0.075, 0.15)
    calf_radius = _scale(params["calf_cm"], 28, 55, 0.055, 0.11)

    return {
        "height": height,
        "head_radius": height * 0.055,
        "neck_radius": _scale(params["neck_cm"], 30, 50, 0.045, 0.078),
        "shoulder_width": shoulder_width,
        "chest_width": chest_width,
        "waist_width": waist_width,
        "hip_width": hip_width,
        "depth": depth,
        "arm_radius": _scale(params["sleeve_cm"], 50, 75, 0.052, 0.078),
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
