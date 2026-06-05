"""Render front, side, and back samples from a prepared Blender .blend scene.

This file is intentionally import-safe outside Blender. The bpy dependency is
only imported from main(), which keeps normal pytest runs lightweight.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    CAMERA_VIEWS,
    DEFAULT_CAMERA_NAMES,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LABEL_NOISE_CM,
    DEFAULT_LABEL_MEASUREMENT_SCALE,
    DEFAULT_MOBILE_REALISM_SETTINGS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_VARIATION_DEGREES,
    DEFAULT_SAFE_FRAMING_SCALE,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_SHAPE_KEY_RANGE,
    DEFAULT_SOURCE_MODE,
    GENERATOR_VERSION,
    STATIC_BLEND_WARNING,
    SHAPE_KEY_COUPLED_LABEL_SOURCE,
    camera_set_name,
    generate_shape_key_coupled_measurements,
    LABEL_GENERATION_MODE,
    LABEL_FORMULA_VERSION,
    repo_relative_path,
    resolve_repo_path,
    shape_key_label_metadata,
    shape_key_label_metadata_for_version,
    shape_key_traceability_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a synthetic dataset from a prepared Blender .blend scene.")
    parser.add_argument("--source", choices=[DEFAULT_SOURCE_MODE], default=DEFAULT_SOURCE_MODE)
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--shape-key-range", type=float, default=DEFAULT_SHAPE_KEY_RANGE)
    parser.add_argument("--pose-variation-degrees", type=float, default=DEFAULT_POSE_VARIATION_DEGREES)
    parser.add_argument("--label-noise-cm", type=float, default=DEFAULT_LABEL_NOISE_CM)
    parser.add_argument("--label-formula-version", default=LABEL_FORMULA_VERSION)
    parser.add_argument("--label-measurement-scale", type=float, default=DEFAULT_LABEL_MEASUREMENT_SCALE)
    parser.add_argument("--view-subdirs", action="store_true")
    parser.add_argument("--safe-framing-scale", type=float, default=DEFAULT_SAFE_FRAMING_SCALE)
    parser.add_argument("--mobile-realism", action="store_true")
    parser.add_argument("--distance-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["distance_jitter"])
    parser.add_argument("--camera-height-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["camera_height_jitter"])
    parser.add_argument("--body-rotation-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["body_rotation_jitter"])
    parser.add_argument("--lighting-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["lighting_jitter"])
    parser.add_argument("--background-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["background_jitter"])
    parser.add_argument("--phone-framing-jitter", type=float, default=DEFAULT_MOBILE_REALISM_SETTINGS["phone_framing_jitter"])
    parser.add_argument("--front-camera", default=DEFAULT_CAMERA_NAMES["front"])
    parser.add_argument("--side-camera", default=DEFAULT_CAMERA_NAMES["side"])
    parser.add_argument("--back-camera", default=DEFAULT_CAMERA_NAMES["back"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if args.start_index <= 0:
        raise ValueError("--start-index must be greater than 0")
    realism_settings = mobile_realism_settings(args)

    try:
        import bpy
    except ImportError:
        print("Blender bpy module is not available. Run this script with Blender.")
        return 1

    blend_path = resolve_repo_path(args.blend_file)
    if not blend_path.exists():
        raise FileNotFoundError(f"Missing Blender .blend file: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    camera_names = {
        "front": args.front_camera,
        "side": args.side_camera,
        "back": args.back_camera,
    }
    cameras = find_required_cameras(bpy, camera_names)
    apply_safe_framing_scale(cameras, args.safe_framing_scale)
    base_camera_states = capture_camera_states(cameras)
    base_light_energies = capture_light_energies(bpy)
    base_world_color = tuple(bpy.context.scene.world.color)
    meshes = find_human_meshes(bpy)
    armatures = find_armatures(bpy)
    shape_key_blocks = collect_shape_key_blocks(meshes)
    variation_source = "shape_keys_safe_range" if shape_key_blocks else "static_blend_mesh"
    if shape_key_blocks and args.mobile_realism:
        variation_source = "shape_keys_safe_range_plus_mobile_realism"
    warnings = [] if shape_key_blocks else [STATIC_BLEND_WARNING]

    output_root = resolve_repo_path(args.out)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.view_subdirs:
        for view in CAMERA_VIEWS:
            (images_dir / view).mkdir(parents=True, exist_ok=True)
    labels_path = output_root / "labels.csv"
    metadata_path = output_root / "metadata.json"
    rng = random.Random(args.seed)
    initial_armature_rotations = {armature.name: tuple(armature.rotation_euler) for armature in armatures}
    initial_mesh_rotations = {mesh.name: tuple(mesh.rotation_euler) for mesh in meshes}
    labels: list[dict[str, object]] = []
    samples_metadata: list[dict[str, object]] = []

    setup_render_settings(bpy, args.image_width, args.image_height)

    for index in range(args.start_index, args.start_index + args.samples):
        sample_id = f"sample_{index:06d}"
        reset_shape_keys(shape_key_blocks)
        reset_armature_rotations(armatures, initial_armature_rotations)
        reset_object_rotations(meshes, initial_mesh_rotations)
        sample_rng = random.Random(f"{args.seed}:{sample_id}")
        shape_key_values = apply_shape_key_variation(shape_key_blocks, sample_rng, args.shape_key_range)
        label_payload = generate_shape_key_coupled_measurements(
            sample_id=sample_id,
            seed=args.seed,
            shape_key_values=shape_key_values,
            shape_key_range=args.shape_key_range,
            label_noise_cm=args.label_noise_cm,
            label_formula_version=args.label_formula_version,
            label_measurement_scale=args.label_measurement_scale,
        )
        pose_degrees = apply_pose_variation(armatures, sample_rng, args.pose_variation_degrees)
        image_paths = {
            view: image_path_for_view(images_dir, sample_id, view, args.view_subdirs)
            for view in CAMERA_VIEWS
        }
        sample_realism = sample_mobile_realism(realism_settings, sample_rng)
        apply_mobile_lighting_and_background(
            bpy,
            settings=realism_settings,
            sample_realism=sample_realism,
            base_light_energies=base_light_energies,
            base_world_color=base_world_color,
        )

        for view in CAMERA_VIEWS:
            reset_camera_state(cameras[view], base_camera_states[view])
            reset_object_rotations(meshes, initial_mesh_rotations)
            apply_mobile_body_rotation(meshes, sample_realism["views"][view]["body_yaw_degrees"])
            apply_mobile_camera_realism(cameras[view], sample_realism["views"][view])
            bpy.context.scene.camera = cameras[view]
            render_view(bpy, image_paths[view], args.image_width, args.image_height)

        relative_images = {
            view: image_paths[view].relative_to(output_root).as_posix()
            for view in CAMERA_VIEWS
        }
        labels.append(
            blend_label_row(
                sample_id=sample_id,
                label_payload=label_payload,
                shape_key_values=shape_key_values,
                relative_images=relative_images,
                blend_path=blend_path,
                variation_source=variation_source,
                camera_names=camera_names,
                seed=args.seed,
            )
        )
        samples_metadata.append(
            {
                "sample_id": sample_id,
                "pose": {"yaw_degrees": pose_degrees},
                "variation": {
                    "variation_source": variation_source,
                    "shape_keys": shape_key_values,
                    "body_factors": label_payload["factors"],
                    "body_shape_profile_id": label_payload["body_shape_profile_id"],
                },
                "mobile_realism": sample_realism,
                "labels": label_payload["measurements"],
                "label_generation_mode": LABEL_GENERATION_MODE,
                "label_formula_version": args.label_formula_version,
                "cameras": camera_names,
                "seed": args.seed,
            }
        )

    write_labels(labels_path, labels)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "source_mode": args.source,
        "source_blend_file": repo_relative_path(blend_path),
        "camera_set": camera_set_name(camera_names),
        "camera_names": camera_names,
        "sample_count": args.samples,
        "seed": args.seed,
        "synthetic_labels": True,
        "real_world_validated": False,
        "variation_source": variation_source,
        "shape_key_count": len(shape_key_blocks),
        "safe_framing_scale": args.safe_framing_scale,
        "mobile_realism": bool(args.mobile_realism),
        "mobile_realism_settings": realism_settings,
        **shape_key_label_metadata_for_version(
            seed=args.seed,
            label_formula_version=args.label_formula_version,
            label_measurement_scale=args.label_measurement_scale,
            shape_key_range=args.shape_key_range,
        ),
        "armature_count": len(armatures),
        "mesh_count": len(meshes),
        "warnings": warnings,
        "samples": samples_metadata,
    }
    write_metadata(metadata_path, metadata)
    print(f"Rendered {args.samples} blend samples to {output_root}")
    return 0


def find_required_cameras(bpy, camera_names: dict[str, str]) -> dict[str, object]:
    missing = [name for name in camera_names.values() if name not in bpy.data.objects]
    wrong_type = [
        name
        for name in camera_names.values()
        if name in bpy.data.objects and bpy.data.objects[name].type != "CAMERA"
    ]
    if missing or wrong_type:
        details = []
        if missing:
            details.append(f"missing cameras: {', '.join(missing)}")
        if wrong_type:
            details.append(f"objects are not cameras: {', '.join(wrong_type)}")
        raise ValueError(
            "The .blend scene must contain FrontCam, SideCam, and BackCam cameras "
            f"or matching --front-camera/--side-camera/--back-camera names; {'; '.join(details)}"
        )
    return {view: bpy.data.objects[name] for view, name in camera_names.items()}


def mobile_realism_settings(args: argparse.Namespace) -> dict[str, object]:
    settings = {
        "mobile_realism": bool(args.mobile_realism),
        "distance_jitter": float(args.distance_jitter),
        "camera_height_jitter": float(args.camera_height_jitter),
        "body_rotation_jitter": float(args.body_rotation_jitter),
        "lighting_jitter": float(args.lighting_jitter),
        "background_jitter": float(args.background_jitter),
        "phone_framing_jitter": float(args.phone_framing_jitter),
    }
    limits = {
        "distance_jitter": 0.08,
        "camera_height_jitter": 0.08,
        "body_rotation_jitter": 6.0,
        "lighting_jitter": 0.18,
        "background_jitter": 0.12,
        "phone_framing_jitter": 0.04,
    }
    for key, upper in limits.items():
        value = float(settings[key])
        if value < 0.0 or value > upper:
            raise ValueError(f"--{key.replace('_', '-')} must be between 0.0 and {upper}.")
    if not settings["mobile_realism"]:
        for key in limits:
            settings[key] = 0.0
    return settings


def capture_camera_states(cameras: dict[str, object]) -> dict[str, dict[str, object]]:
    states: dict[str, dict[str, object]] = {}
    for view, camera in cameras.items():
        data = getattr(camera, "data", None)
        states[view] = {
            "location": tuple(camera.location),
            "rotation_euler": tuple(camera.rotation_euler),
            "ortho_scale": float(getattr(data, "ortho_scale", 0.0)) if data is not None else 0.0,
            "lens": float(getattr(data, "lens", 0.0)) if data is not None else 0.0,
            "shift_x": float(getattr(data, "shift_x", 0.0)) if data is not None else 0.0,
            "shift_y": float(getattr(data, "shift_y", 0.0)) if data is not None else 0.0,
        }
    return states


def reset_camera_state(camera: object, state: dict[str, object]) -> None:
    camera.location = state["location"]
    camera.rotation_euler = state["rotation_euler"]
    data = getattr(camera, "data", None)
    if data is None:
        return
    if hasattr(data, "ortho_scale"):
        data.ortho_scale = state["ortho_scale"]
    if hasattr(data, "lens"):
        data.lens = state["lens"]
    if hasattr(data, "shift_x"):
        data.shift_x = state["shift_x"]
    if hasattr(data, "shift_y"):
        data.shift_y = state["shift_y"]


def capture_light_energies(bpy) -> dict[str, float]:
    return {obj.name: float(obj.data.energy) for obj in bpy.data.objects if obj.type == "LIGHT" and hasattr(obj.data, "energy")}


def sample_mobile_realism(settings: dict[str, object], rng: random.Random) -> dict[str, object]:
    enabled = bool(settings["mobile_realism"])
    lighting_jitter = float(settings["lighting_jitter"]) if enabled else 0.0
    background_jitter = float(settings["background_jitter"]) if enabled else 0.0
    payload: dict[str, object] = {
        "enabled": enabled,
        "lighting_multiplier": round(rng.uniform(1.0 - lighting_jitter, 1.0 + lighting_jitter), 6),
        "background_brightness": round(rng.uniform(1.0 - background_jitter, 1.0 + background_jitter), 6),
        "background_color_delta": [
            round(rng.uniform(-background_jitter, background_jitter), 6),
            round(rng.uniform(-background_jitter, background_jitter), 6),
            round(rng.uniform(-background_jitter, background_jitter), 6),
        ],
        "views": {},
    }
    for view in CAMERA_VIEWS:
        distance = float(settings["distance_jitter"]) if enabled else 0.0
        framing = float(settings["phone_framing_jitter"]) if enabled else 0.0
        height = float(settings["camera_height_jitter"]) if enabled else 0.0
        rotation = float(settings["body_rotation_jitter"]) if enabled else 0.0
        payload["views"][view] = {
            "distance_scale": round(rng.uniform(1.0 - distance, 1.0 + distance), 6),
            "camera_height_offset": round(rng.uniform(-height, height), 6),
            "framing_shift_x": round(rng.uniform(-framing, framing), 6),
            "framing_shift_y": round(rng.uniform(-framing, framing), 6),
            "body_yaw_degrees": round(rng.uniform(-rotation, rotation), 6),
        }
    return payload


def apply_mobile_lighting_and_background(
    bpy,
    *,
    settings: dict[str, object],
    sample_realism: dict[str, object],
    base_light_energies: dict[str, float],
    base_world_color: tuple[float, float, float],
) -> None:
    if not bool(settings["mobile_realism"]):
        return
    lighting_multiplier = float(sample_realism["lighting_multiplier"])
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.name in base_light_energies and hasattr(obj.data, "energy"):
            obj.data.energy = base_light_energies[obj.name] * lighting_multiplier
    brightness = float(sample_realism["background_brightness"])
    deltas = sample_realism["background_color_delta"]
    color = [
        max(0.0, min(1.0, float(base_world_color[index]) * brightness + float(deltas[index])))
        for index in range(3)
    ]
    bpy.context.scene.world.color = (color[0], color[1], color[2])


def apply_mobile_camera_realism(camera: object, view_realism: dict[str, float]) -> None:
    data = getattr(camera, "data", None)
    if data is not None and getattr(data, "type", "") == "ORTHO" and hasattr(data, "ortho_scale"):
        data.ortho_scale = float(data.ortho_scale) * float(view_realism["distance_scale"])
    elif data is not None and hasattr(data, "lens"):
        data.lens = max(1.0, float(data.lens) / float(view_realism["distance_scale"]))
    camera.location[2] += float(view_realism["camera_height_offset"])
    if data is not None and hasattr(data, "shift_x"):
        data.shift_x += float(view_realism["framing_shift_x"])
    if data is not None and hasattr(data, "shift_y"):
        data.shift_y += float(view_realism["framing_shift_y"])


def apply_mobile_body_rotation(meshes: list[object], yaw_degrees: float) -> None:
    yaw_radians = float(yaw_degrees) * math.pi / 180.0
    for mesh in meshes:
        mesh.rotation_euler[2] += yaw_radians


def apply_safe_framing_scale(cameras: dict[str, object], safe_framing_scale: float) -> None:
    scale = max(1.0, float(safe_framing_scale))
    if scale <= 1.0:
        return
    for camera in cameras.values():
        data = getattr(camera, "data", None)
        if data is None:
            continue
        if getattr(data, "type", "") == "ORTHO":
            data.ortho_scale = float(data.ortho_scale) * scale
        else:
            data.lens = max(1.0, float(data.lens) / scale)


def find_human_meshes(bpy) -> list[object]:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and getattr(obj.data, "vertices", None)]
    if not meshes:
        raise ValueError("The .blend scene must contain at least one human mesh object.")
    return meshes


def find_armatures(bpy) -> list[object]:
    return [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]


def collect_shape_key_blocks(meshes: list[object]) -> list[object]:
    blocks = []
    for mesh in meshes:
        shape_keys = getattr(mesh.data, "shape_keys", None)
        if not shape_keys:
            continue
        for block in shape_keys.key_blocks:
            if block.name.lower() != "basis":
                blocks.append(block)
    return blocks


def reset_shape_keys(shape_key_blocks: list[object]) -> None:
    for block in shape_key_blocks:
        block.value = 0.0


def apply_shape_key_variation(shape_key_blocks: list[object], rng: random.Random, safe_range: float) -> dict[str, float]:
    values: dict[str, float] = {}
    safe_range = max(0.0, min(float(safe_range), 0.3))
    for block in shape_key_blocks:
        slider_min = float(getattr(block, "slider_min", 0.0))
        slider_max = float(getattr(block, "slider_max", 1.0))
        lower = max(slider_min, -safe_range)
        upper = min(slider_max, safe_range)
        if lower > upper:
            lower = upper = max(slider_min, min(slider_max, 0.0))
        value = round(rng.uniform(lower, upper), 4)
        block.value = value
        values[block.name] = value
    return values


def image_path_for_view(images_dir: Path, sample_id: str, view: str, view_subdirs: bool) -> Path:
    if view_subdirs:
        return images_dir / view / f"{sample_id}_{view}.png"
    return images_dir / f"{sample_id}_{view}.png"


def reset_armature_rotations(armatures: list[object], initial_rotations: dict[str, tuple[float, float, float]]) -> None:
    for armature in armatures:
        if armature.name in initial_rotations:
            armature.rotation_euler = initial_rotations[armature.name]


def reset_object_rotations(objects: list[object], initial_rotations: dict[str, tuple[float, float, float]]) -> None:
    for obj in objects:
        if obj.name in initial_rotations:
            obj.rotation_euler = initial_rotations[obj.name]


def apply_pose_variation(armatures: list[object], rng: random.Random, limit_degrees: float) -> float:
    if not armatures or limit_degrees <= 0:
        return 0.0
    limit_degrees = min(float(limit_degrees), 3.0)
    yaw_degrees = round(rng.uniform(-limit_degrees, limit_degrees), 3)
    armatures[0].rotation_euler[2] += yaw_degrees * 3.141592653589793 / 180.0
    return yaw_degrees


def setup_render_settings(bpy, width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"


def render_view(bpy, output_path: Path, width: int, height: int) -> None:
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = str(output_path)
    print(f"Rendering {output_path}")
    bpy.ops.render.render(write_still=True)


def blend_label_row(
    *,
    sample_id: str,
    label_payload: dict[str, object],
    shape_key_values: dict[str, float],
    relative_images: dict[str, str],
    blend_path: Path,
    variation_source: str,
    camera_names: dict[str, str],
    seed: int,
) -> dict[str, object]:
    measurements = label_payload["measurements"]
    factors = label_payload["factors"]
    return {
        "sample_id": sample_id,
        "front_image": relative_images["front"],
        "side_image": relative_images["side"],
        "back_image": relative_images["back"],
        "height_cm": measurements["height_cm"],
        "chest_cm": measurements["chest_cm"],
        "waist_cm": measurements["waist_cm"],
        "hip_cm": measurements["hip_cm"],
        "shoulder_cm": measurements["shoulder_cm"],
        "inseam_cm": measurements["inseam_cm"],
        "source_blend_file": repo_relative_path(blend_path),
        "variation_source": variation_source,
        "camera_set": camera_set_name(camera_names),
        "seed": seed,
        "label_source": SHAPE_KEY_COUPLED_LABEL_SOURCE,
        "synthetic_labels": "true",
        "real_world_validated": "false",
        "label_generation_mode": LABEL_GENERATION_MODE,
        "height_factor": factors["height_factor"],
        "chest_factor": factors["chest_factor"],
        "waist_factor": factors["waist_factor"],
        "hip_factor": factors["hip_factor"],
        "shoulder_factor": factors["shoulder_factor"],
        "inseam_factor": factors["inseam_factor"],
        "torso_width_factor": factors["torso_width_factor"],
        "leg_length_factor": factors["leg_length_factor"],
        "shape_key_values_json": shape_key_traceability_json(shape_key_values),
        "body_shape_profile_id": label_payload["body_shape_profile_id"],
    }


def write_labels(labels_path: Path, rows: list[dict[str, object]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
