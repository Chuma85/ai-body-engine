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
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_VARIATION_DEGREES,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_SHAPE_KEY_RANGE,
    DEFAULT_SOURCE_MODE,
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
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    blender_executable: str = "blender",
    script_path: str = DEFAULT_BLENDER_SCRIPT,
    shape_key_range: float = DEFAULT_SHAPE_KEY_RANGE,
    pose_variation_degrees: float = DEFAULT_POSE_VARIATION_DEGREES,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if source != DEFAULT_SOURCE_MODE:
        raise ValueError(f"Unsupported source mode: {source}. Use --source blend.")
    if samples <= 0:
        raise ValueError("--samples must be greater than 0")

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
        image_width=image_width,
        image_height=image_height,
        shape_key_range=shape_key_range,
        pose_variation_degrees=pose_variation_degrees,
    )
    if dry_run:
        return {
            "mode": source,
            "blend_file": str(blend_path),
            "output_dir": str(output_path),
            "sample_count": samples,
            "seed": seed,
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
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--blender-executable", default="blender")
    parser.add_argument("--script-path", default=DEFAULT_BLENDER_SCRIPT)
    parser.add_argument("--shape-key-range", type=float, default=DEFAULT_SHAPE_KEY_RANGE)
    parser.add_argument("--pose-variation-degrees", type=float, default=DEFAULT_POSE_VARIATION_DEGREES)
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
        image_width=args.image_width,
        image_height=args.image_height,
        blender_executable=args.blender_executable,
        script_path=args.script_path,
        shape_key_range=args.shape_key_range,
        pose_variation_degrees=args.pose_variation_degrees,
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
