from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.blender.utils.blender_command import build_blender_command, format_command
from synthetic.generator.generate_dataset import generate_dataset
from synthetic.validate_synthetic_dataset import validate_dataset


DEFAULT_OUTPUT_DIR = "data/synthetic/phase_3t_enhanced"
DEFAULT_BLENDER_CONFIG = "synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json"
DEFAULT_BLENDER_SCRIPT = "synthetic/blender/scripts/render_parametric_body.py"
DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_SMOKE_SAMPLE_COUNT = 3
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 896
DEFAULT_SMOKE_IMAGE_WIDTH = 128
DEFAULT_SMOKE_IMAGE_HEIGHT = 192
DEFAULT_SEED = 530042
REALISTIC_MODE = "blender_realistic"
SMOKE_MODE = "lightweight_smoke"


def generate_enhanced_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    return generate_lightweight_smoke_dataset(
        output_dir=output_dir,
        sample_count=sample_count,
        width=width,
        height=height,
        seed=seed,
        overwrite=overwrite,
    )


def generate_lightweight_smoke_dataset(
    output_dir: str | Path,
    sample_count: int = DEFAULT_SMOKE_SAMPLE_COUNT,
    width: int = DEFAULT_SMOKE_IMAGE_WIDTH,
    height: int = DEFAULT_SMOKE_IMAGE_HEIGHT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_path}. "
                "Pass --overwrite to regenerate it."
            )
        shutil.rmtree(output_path)

    labels_csv = generate_dataset(
        count=sample_count,
        output_dir=str(output_path),
        width=width,
        height=height,
        seed=seed,
        include_back_view=True,
    )
    validation = validate_dataset(output_path, require_back=True)
    if not validation["valid"]:
        raise ValueError("Enhanced dataset validation failed: " + "; ".join(validation["errors"]))

    manifest = build_dataset_manifest(output_path, require_back=True)
    if not manifest["valid"]:
        raise ValueError("Enhanced dataset manifest build failed: " + "; ".join(manifest["errors"]))

    alignment = sample_alignment_summary(output_path)
    if alignment["aligned_sample_count"] != sample_count:
        raise ValueError(
            "Enhanced dataset sample alignment failed: "
            f"expected {sample_count}, got {alignment['aligned_sample_count']}."
        )

    return {
        "mode": SMOKE_MODE,
        "output_dir": str(output_path),
        "labels_csv": str(labels_csv),
        "manifest_csv": manifest["manifest_path"],
        "sample_count": sample_count,
        "width": width,
        "height": height,
        "seed": seed,
        "validation": validation,
        "manifest": manifest,
        "alignment": alignment,
    }


def generate_realistic_blender_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    overwrite: bool = False,
    blender_executable: str = "blender",
    config_path: str = DEFAULT_BLENDER_CONFIG,
    script_path: str = DEFAULT_BLENDER_SCRIPT,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_path}. "
                "Pass --overwrite to regenerate it."
            )
        if not dry_run:
            shutil.rmtree(output_path)

    command = build_realistic_blender_command(
        blender_executable=blender_executable,
        config_path=config_path,
        script_path=script_path,
        output_dir=str(output_path),
        sample_count=sample_count,
    )
    if dry_run:
        return {
            "mode": REALISTIC_MODE,
            "output_dir": str(output_path),
            "sample_count": sample_count,
            "command": command,
            "formatted_command": format_command(command),
            "dry_run": True,
        }

    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Blender executable was not found. Install Blender or pass --smoke-lightweight "
            "for tiny placeholder-only smoke verification."
        ) from exc

    validation = validate_dataset(output_path, require_back=True, require_realistic=True)
    if not validation["valid"]:
        raise ValueError("Realistic enhanced dataset validation failed: " + "; ".join(validation["errors"]))

    manifest = build_dataset_manifest(output_path, require_back=True, require_realistic=True)
    if not manifest["valid"]:
        raise ValueError("Realistic enhanced dataset manifest build failed: " + "; ".join(manifest["errors"]))

    alignment = sample_alignment_summary(output_path)
    if alignment["aligned_sample_count"] != sample_count:
        raise ValueError(
            "Realistic enhanced dataset sample alignment failed: "
            f"expected {sample_count}, got {alignment['aligned_sample_count']}."
        )

    return {
        "mode": REALISTIC_MODE,
        "output_dir": str(output_path),
        "manifest_csv": manifest["manifest_path"],
        "sample_count": sample_count,
        "validation": validation,
        "manifest": manifest,
        "alignment": alignment,
    }


def build_realistic_blender_command(
    blender_executable: str,
    config_path: str,
    script_path: str,
    output_dir: str,
    sample_count: int,
) -> list[str]:
    command = build_blender_command(
        blender_executable=blender_executable,
        script_path=script_path,
        config_path=config_path,
    )
    return [*command, "--output", output_dir, "--num-samples", str(sample_count)]


def sample_alignment_summary(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    front_ids = sample_ids(root / "images" / "front", "_front.png")
    side_ids = sample_ids(root / "images" / "side", "_side.png")
    back_ids = sample_ids(root / "images" / "back", "_back.png")
    aligned = front_ids & side_ids & back_ids
    return {
        "front_count": len(front_ids),
        "side_count": len(side_ids),
        "back_count": len(back_ids),
        "aligned_sample_count": len(aligned),
        "front_only": sorted(front_ids - side_ids - back_ids),
        "side_only": sorted(side_ids - front_ids - back_ids),
        "back_only": sorted(back_ids - front_ids - side_ids),
        "missing_back": sorted((front_ids & side_ids) - back_ids),
    }


def sample_ids(directory: Path, suffix: str) -> set[str]:
    if not directory.exists():
        return set()
    return {
        image_path.name[: -len(suffix)]
        for image_path in directory.glob(f"*{suffix}")
        if image_path.name.endswith(suffix)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Phase 3T enhanced front/side/back synthetic dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--blender-executable", default="blender")
    parser.add_argument("--config", default=DEFAULT_BLENDER_CONFIG)
    parser.add_argument("--script-path", default=DEFAULT_BLENDER_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-lightweight", action="store_true", help="Generate tiny placeholder images for smoke verification only.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --smoke-lightweight.")
    args = parser.parse_args(argv)

    use_smoke = args.smoke_lightweight or args.smoke
    if use_smoke:
        sample_count = args.sample_count or DEFAULT_SMOKE_SAMPLE_COUNT
        width = args.width or DEFAULT_SMOKE_IMAGE_WIDTH
        height = args.height or DEFAULT_SMOKE_IMAGE_HEIGHT
        result = generate_lightweight_smoke_dataset(
            output_dir=args.output_dir,
            sample_count=sample_count,
            width=width,
            height=height,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print("Generated lightweight smoke-only Phase 3T dataset.")
        print("This output is not a training candidate.")
        print(f"Labels: {result['labels_csv']}")
    else:
        sample_count = args.sample_count or DEFAULT_SAMPLE_COUNT
        result = generate_realistic_blender_dataset(
            output_dir=args.output_dir,
            sample_count=sample_count,
            overwrite=args.overwrite,
            blender_executable=args.blender_executable,
            config_path=args.config,
            script_path=args.script_path,
            dry_run=args.dry_run,
        )
        if result.get("dry_run"):
            print(result["formatted_command"])
            return 0
        print("Generated realistic Blender Phase 3T enhanced dataset.")

    alignment = result.get("alignment")
    print(f"Mode: {result['mode']}")
    print(f"Output: {result['output_dir']}")
    print(f"Samples: {result['sample_count']}")
    if result.get("manifest_csv"):
        print(f"Manifest: {result['manifest_csv']}")
    if alignment:
        print(
            "Aligned front/side/back samples: "
            f"{alignment['aligned_sample_count']} "
            f"(front={alignment['front_count']} side={alignment['side_count']} back={alignment['back_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
