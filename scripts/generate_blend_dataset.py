from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.blender.blend_dataset import (
    DEFAULT_BLEND_FILE,
    DEFAULT_BLENDER_SCRIPT,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LABEL_NOISE_CM,
    DEFAULT_LABEL_MEASUREMENT_SCALE,
    DEFAULT_MOBILE_REALISM_SETTINGS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_VARIATION_DEGREES,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_SAFE_FRAMING_SCALE,
    DEFAULT_SHAPE_KEY_RANGE,
    DEFAULT_SOURCE_MODE,
    LABEL_FORMULA_VERSION,
    build_blend_blender_command,
    resolve_repo_path,
    validate_blend_file_exists,
    validate_generated_blend_dataset,
)
from synthetic.blender.utils.blender_command import format_command


def blender_executable_available(blender_executable: str) -> bool:
    executable_path = Path(blender_executable)
    if executable_path.exists():
        return True
    return shutil.which(blender_executable) is not None


def generate_blend_dataset(
    *,
    source: str = DEFAULT_SOURCE_MODE,
    blend_file: str = DEFAULT_BLEND_FILE,
    out: str = DEFAULT_OUTPUT_DIR,
    samples: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    start_index: int = 1,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    blender_executable: str = "blender",
    script_path: str = DEFAULT_BLENDER_SCRIPT,
    shape_key_range: float = DEFAULT_SHAPE_KEY_RANGE,
    pose_variation_degrees: float = DEFAULT_POSE_VARIATION_DEGREES,
    label_noise_cm: float = DEFAULT_LABEL_NOISE_CM,
    label_formula_version: str | None = None,
    label_measurement_scale: float = DEFAULT_LABEL_MEASUREMENT_SCALE,
    view_subdirs: bool = False,
    safe_framing_scale: float = DEFAULT_SAFE_FRAMING_SCALE,
    mobile_realism: bool = DEFAULT_MOBILE_REALISM_SETTINGS["mobile_realism"],
    distance_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["distance_jitter"],
    camera_height_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["camera_height_jitter"],
    body_rotation_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["body_rotation_jitter"],
    lighting_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["lighting_jitter"],
    background_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["background_jitter"],
    phone_framing_jitter: float = DEFAULT_MOBILE_REALISM_SETTINGS["phone_framing_jitter"],
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if source != DEFAULT_SOURCE_MODE:
        raise ValueError(f"Unsupported source mode: {source}. Use --source blend.")
    if samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if start_index <= 0:
        raise ValueError("--start-index must be greater than 0")

    blend_path = validate_blend_file_exists(blend_file)
    output_path = resolve_repo_path(out)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_path}. "
                "Pass --overwrite to regenerate it."
            )
        if not dry_run:
            shutil.rmtree(output_path)

    command = build_blend_blender_command(
        blender_executable=blender_executable,
        script_path=script_path,
        blend_file=str(blend_path),
        output_dir=str(output_path),
        samples=samples,
        seed=seed,
        start_index=start_index,
        image_width=image_width,
        image_height=image_height,
        shape_key_range=shape_key_range,
        pose_variation_degrees=pose_variation_degrees,
        label_noise_cm=label_noise_cm,
        label_formula_version=label_formula_version or LABEL_FORMULA_VERSION,
        label_measurement_scale=label_measurement_scale,
        view_subdirs=view_subdirs,
        safe_framing_scale=safe_framing_scale,
        mobile_realism=mobile_realism,
        distance_jitter=distance_jitter,
        camera_height_jitter=camera_height_jitter,
        body_rotation_jitter=body_rotation_jitter,
        lighting_jitter=lighting_jitter,
        background_jitter=background_jitter,
        phone_framing_jitter=phone_framing_jitter,
    )
    if dry_run:
        return {
            "mode": source,
            "blend_file": str(blend_path),
            "output_dir": str(output_path),
            "sample_count": samples,
            "seed": seed,
            "start_index": start_index,
            "command": command,
            "formatted_command": format_command(command),
            "dry_run": True,
        }

    if not blender_executable_available(blender_executable):
        raise RuntimeError(
            "Blender executable was not found. Install Blender, add it to PATH, "
            "or pass --blender-executable with the full blender.exe path."
        )

    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Blender executable was not found. Install Blender, add it to PATH, "
            "or pass --blender-executable with the full blender.exe path."
        ) from exc

    validation = validate_generated_blend_dataset(output_path, expected_samples=samples)
    if not validation["valid"]:
        raise ValueError("Blend dataset validation failed: " + "; ".join(validation["errors"]))

    return {
        "mode": source,
        "blend_file": str(blend_path),
        "output_dir": str(output_path),
        "sample_count": samples,
        "seed": seed,
        "labels_csv": str(output_path / "labels.csv"),
        "metadata_json": str(output_path / "metadata.json"),
        "validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a front/side/back synthetic dataset from a Blender .blend scene.")
    parser.add_argument("--source", choices=[DEFAULT_SOURCE_MODE], default=DEFAULT_SOURCE_MODE)
    parser.add_argument("--blend-file", default=DEFAULT_BLEND_FILE)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--blender-executable", default="blender")
    parser.add_argument("--script-path", default=DEFAULT_BLENDER_SCRIPT)
    parser.add_argument("--shape-key-range", type=float, default=DEFAULT_SHAPE_KEY_RANGE)
    parser.add_argument("--pose-variation-degrees", type=float, default=DEFAULT_POSE_VARIATION_DEGREES)
    parser.add_argument("--label-noise-cm", type=float, default=DEFAULT_LABEL_NOISE_CM)
    parser.add_argument("--label-formula-version", default=None)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_blend_dataset(
        source=args.source,
        blend_file=args.blend_file,
        out=args.out,
        samples=args.samples,
        seed=args.seed,
        start_index=args.start_index,
        image_width=args.image_width,
        image_height=args.image_height,
        blender_executable=args.blender_executable,
        script_path=args.script_path,
        shape_key_range=args.shape_key_range,
        pose_variation_degrees=args.pose_variation_degrees,
        label_noise_cm=args.label_noise_cm,
        label_formula_version=args.label_formula_version,
        label_measurement_scale=args.label_measurement_scale,
        view_subdirs=args.view_subdirs,
        safe_framing_scale=args.safe_framing_scale,
        mobile_realism=args.mobile_realism,
        distance_jitter=args.distance_jitter,
        camera_height_jitter=args.camera_height_jitter,
        body_rotation_jitter=args.body_rotation_jitter,
        lighting_jitter=args.lighting_jitter,
        background_jitter=args.background_jitter,
        phone_framing_jitter=args.phone_framing_jitter,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if result.get("dry_run"):
        print(result["formatted_command"])
        return 0

    print("Generated Blender blend-file synthetic dataset.")
    print(f"Blend file: {result['blend_file']}")
    print(f"Output: {result['output_dir']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Labels: {result['labels_csv']}")
    print(f"Metadata: {result['metadata_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
