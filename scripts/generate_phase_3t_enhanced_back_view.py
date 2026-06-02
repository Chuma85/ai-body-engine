from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.generator.generate_dataset import generate_dataset
from synthetic.validate_synthetic_dataset import validate_dataset


DEFAULT_OUTPUT_DIR = "data/synthetic/phase_3t_enhanced"
DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_SMOKE_SAMPLE_COUNT = 3
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 896
DEFAULT_SMOKE_IMAGE_WIDTH = 128
DEFAULT_SMOKE_IMAGE_HEIGHT = 192
DEFAULT_SEED = 530042


def generate_enhanced_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
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
    parser.add_argument("--smoke", action="store_true", help="Generate a tiny deterministic dataset for verification.")
    args = parser.parse_args(argv)

    sample_count = args.sample_count or (DEFAULT_SMOKE_SAMPLE_COUNT if args.smoke else DEFAULT_SAMPLE_COUNT)
    width = args.width or (DEFAULT_SMOKE_IMAGE_WIDTH if args.smoke else DEFAULT_IMAGE_WIDTH)
    height = args.height or (DEFAULT_SMOKE_IMAGE_HEIGHT if args.smoke else DEFAULT_IMAGE_HEIGHT)
    result = generate_enhanced_dataset(
        output_dir=args.output_dir,
        sample_count=sample_count,
        width=width,
        height=height,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    alignment = result["alignment"]
    print(f"Generated enhanced Phase 3T dataset: {result['output_dir']}")
    print(f"Samples: {result['sample_count']}")
    print(f"Labels: {result['labels_csv']}")
    print(f"Manifest: {result['manifest_csv']}")
    print(
        "Aligned front/side/back samples: "
        f"{alignment['aligned_sample_count']} "
        f"(front={alignment['front_count']} side={alignment['side_count']} back={alignment['back_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
