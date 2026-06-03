"""Render front, side, and back samples from a prepared Blender .blend scene.

This file is intentionally import-safe outside Blender. The bpy dependency is
only imported from main(), which keeps normal pytest runs lightweight.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_VARIATION_DEGREES,
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
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--shape-key-range", type=float, default=DEFAULT_SHAPE_KEY_RANGE)
    parser.add_argument("--pose-variation-degrees", type=float, default=DEFAULT_POSE_VARIATION_DEGREES)
    parser.add_argument("--label-noise-cm", type=float, default=DEFAULT_LABEL_NOISE_CM)
    parser.add_argument("--front-camera", default=DEFAULT_CAMERA_NAMES["front"])
    parser.add_argument("--side-camera", default=DEFAULT_CAMERA_NAMES["side"])
    parser.add_argument("--back-camera", default=DEFAULT_CAMERA_NAMES["back"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0")

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
    meshes = find_human_meshes(bpy)
    armatures = find_armatures(bpy)
    shape_key_blocks = collect_shape_key_blocks(meshes)
    variation_source = "shape_keys_safe_range" if shape_key_blocks else "static_blend_mesh"
    warnings = [] if shape_key_blocks else [STATIC_BLEND_WARNING]

    output_root = resolve_repo_path(args.out)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_root / "labels.csv"
    metadata_path = output_root / "metadata.json"
    rng = random.Random(args.seed)
    initial_armature_rotations = {armature.name: tuple(armature.rotation_euler) for armature in armatures}
    labels: list[dict[str, object]] = []
    samples_metadata: list[dict[str, object]] = []

    setup_render_settings(bpy, args.image_width, args.image_height)

    for index in range(1, args.samples + 1):
        sample_id = f"sample_{index:06d}"
        reset_shape_keys(shape_key_blocks)
        reset_armature_rotations(armatures, initial_armature_rotations)
        sample_rng = random.Random(f"{args.seed}:{sample_id}")
        shape_key_values = apply_shape_key_variation(shape_key_blocks, sample_rng, args.shape_key_range)
        label_payload = generate_shape_key_coupled_measurements(
            sample_id=sample_id,
            seed=args.seed,
            shape_key_values=shape_key_values,
            shape_key_range=args.shape_key_range,
            label_noise_cm=args.label_noise_cm,
        )
        pose_degrees = apply_pose_variation(armatures, sample_rng, args.pose_variation_degrees)
        image_paths = {
            view: images_dir / f"{sample_id}_{view}.png"
            for view in CAMERA_VIEWS
        }

        for view in CAMERA_VIEWS:
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
                "labels": label_payload["measurements"],
                "label_generation_mode": LABEL_GENERATION_MODE,
                "label_formula_version": LABEL_FORMULA_VERSION,
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
        **shape_key_label_metadata(args.seed),
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


def reset_armature_rotations(armatures: list[object], initial_rotations: dict[str, tuple[float, float, float]]) -> None:
    for armature in armatures:
        if armature.name in initial_rotations:
            armature.rotation_euler = initial_rotations[armature.name]


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
