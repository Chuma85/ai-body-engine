import argparse
import csv
from pathlib import Path
import random

from synthetic.generator.body_profile import BodyProfile, generate_body_profile
from synthetic.generator.config import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_COUNT,
    RANDOM_SEED,
)
from synthetic.generator.render_silhouette import render_front_silhouette, render_side_silhouette

GENERATOR_VERSION = "phase_2a_python_silhouette_v1"
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


def generate_dataset(
    count: int = DEFAULT_SAMPLE_COUNT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    seed: int = RANDOM_SEED,
) -> Path:
    output_path = Path(output_dir)
    front_dir = output_path / "images" / "front"
    side_dir = output_path / "images" / "side"
    labels_dir = output_path / "labels"
    front_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    labels_csv = labels_dir / "labels.csv"

    with labels_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()

        for index in range(1, count + 1):
            profile = generate_body_profile(index, rng)
            front_path = front_dir / f"{profile.sample_id}_front.png"
            side_path = side_dir / f"{profile.sample_id}_side.png"

            render_front_silhouette(profile, str(front_path), width, height)
            render_side_silhouette(profile, str(side_path), width, height)

            writer.writerow(_label_row(profile, front_path, side_path))

    return labels_csv


def _label_row(profile: BodyProfile, front_path: Path, side_path: Path) -> dict[str, object]:
    row = profile.to_dict()
    row["front_image_path"] = front_path.as_posix()
    row["side_image_path"] = side_path.as_posix()
    row["generator_version"] = GENERATOR_VERSION
    return {column: row[column] for column in LABEL_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2A synthetic silhouette dataset.")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    labels_csv = generate_dataset(
        count=args.count,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"Generated {args.count} samples")
    print(f"Labels: {labels_csv}")


if __name__ == "__main__":
    main()
