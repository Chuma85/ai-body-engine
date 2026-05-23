from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import zlib
from typing import Any

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FRONT_SUFFIX = "_front.png"
SIDE_SUFFIX = "_side.png"
ORIENTATION_WARNING = (
    "TODO Phase 2J: front/side camera orientation is configured by the renderer "
    "but not automatically validated yet; inspect rendered views visually."
)


def validate_dataset(dataset: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset)
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_path = dataset_root / "labels" / "labels.csv"
    result: dict[str, Any] = {
        "valid": False,
        "dataset": str(dataset_root),
        "sample_count": 0,
        "label_row_count": 0,
        "front_image_count": 0,
        "side_image_count": 0,
        "errors": [],
        "warnings": [ORIENTATION_WARNING],
        "missing_paths": [],
        "unpaired_front_samples": [],
        "unpaired_side_samples": [],
        "pairs_missing_label_rows": [],
        "label_rows_missing_image_pairs": [],
        "unreadable_images": [],
    }

    if not dataset_root.exists():
        _add_missing_path(result, dataset_root)
    if not front_dir.exists():
        _add_missing_path(result, front_dir)
    if not side_dir.exists():
        _add_missing_path(result, side_dir)
    if not labels_path.exists():
        _add_missing_path(result, labels_path)

    if result["missing_paths"]:
        return result

    front_images = sorted(front_dir.glob("*.png"))
    side_images = sorted(side_dir.glob("*.png"))
    result["front_image_count"] = len(front_images)
    result["side_image_count"] = len(side_images)

    front_samples = _sample_map(front_images, FRONT_SUFFIX, result)
    side_samples = _sample_map(side_images, SIDE_SUFFIX, result)
    label_rows = _read_label_rows(labels_path, result)
    label_samples = {row.get("sample_id", "") for row in label_rows if row.get("sample_id")}

    result["label_row_count"] = len(label_rows)
    result["sample_count"] = len(front_samples & side_samples & label_samples)
    result["unpaired_front_samples"] = sorted(front_samples - side_samples)
    result["unpaired_side_samples"] = sorted(side_samples - front_samples)
    result["pairs_missing_label_rows"] = sorted((front_samples & side_samples) - label_samples)
    result["label_rows_missing_image_pairs"] = sorted(label_samples - (front_samples & side_samples))

    for key in (
        "unpaired_front_samples",
        "unpaired_side_samples",
        "pairs_missing_label_rows",
        "label_rows_missing_image_pairs",
    ):
        if result[key]:
            result["errors"].append(f"{key}: {', '.join(result[key])}")

    for image_path in [*front_images, *side_images]:
        if not is_readable_png(image_path):
            result["unreadable_images"].append(str(image_path))

    if result["unreadable_images"]:
        result["errors"].append("unreadable_images: " + ", ".join(result["unreadable_images"]))

    result["valid"] = result["sample_count"] > 0 and not result["errors"]
    return result


def is_readable_png(image_path: Path) -> bool:
    try:
        with image_path.open("rb") as image_file:
            if image_file.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                return False

            seen_ihdr = False
            while True:
                length_bytes = image_file.read(4)
                if len(length_bytes) != 4:
                    return False

                chunk_length = int.from_bytes(length_bytes, "big")
                chunk_type = image_file.read(4)
                chunk_data = image_file.read(chunk_length)
                expected_crc = image_file.read(4)

                if len(chunk_type) != 4 or len(chunk_data) != chunk_length or len(expected_crc) != 4:
                    return False

                actual_crc = zlib.crc32(chunk_type)
                actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
                if actual_crc != int.from_bytes(expected_crc, "big"):
                    return False

                if chunk_type == b"IHDR":
                    if seen_ihdr or chunk_length != 13:
                        return False
                    seen_ihdr = True
                elif not seen_ihdr:
                    return False
                elif chunk_type == b"IEND":
                    return True
    except OSError:
        return False


def _read_label_rows(labels_path: Path, result: dict[str, Any]) -> list[dict[str, str]]:
    try:
        with labels_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
    except csv.Error as error:
        result["errors"].append(f"labels.csv parse error: {error}")
        return []

    if not rows:
        result["errors"].append("labels.csv is empty")
        return []

    first_row = rows[0]
    if first_row and first_row[0] == "sample_id":
        columns = first_row
        data_rows = rows[1:]
    else:
        columns = LABEL_COLUMNS
        data_rows = rows
        result["warnings"].append("labels.csv has no header; parsed using the Phase 2G label column order.")

    parsed_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(data_rows, start=2 if columns is first_row else 1):
        if not row or all(value == "" for value in row):
            continue
        if len(row) < 3:
            result["errors"].append(f"labels.csv row {row_number} has fewer than 3 columns")
            continue

        parsed_rows.append(dict(zip(columns, row)))

    return parsed_rows


def _sample_map(image_paths: list[Path], expected_suffix: str, result: dict[str, Any]) -> set[str]:
    sample_ids: set[str] = set()
    for image_path in image_paths:
        sample_id = _sample_id_from_image_name(image_path.name, expected_suffix)
        if sample_id is None:
            result["errors"].append(f"unexpected image filename: {image_path}")
            continue
        sample_ids.add(sample_id)
    return sample_ids


def _sample_id_from_image_name(filename: str, expected_suffix: str) -> str | None:
    if not filename.endswith(expected_suffix):
        return None
    return filename[: -len(expected_suffix)]


def _add_missing_path(result: dict[str, Any], path: Path) -> None:
    result["missing_paths"].append(str(path))
    result["errors"].append(f"missing path: {path}")


def format_validation_report(result: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {result['dataset']}",
        f"Valid: {result['valid']}",
        f"Samples complete: {result['sample_count']}",
        f"Front PNGs: {result['front_image_count']}",
        f"Side PNGs: {result['side_image_count']}",
        f"Label rows: {result['label_row_count']}",
    ]

    if result["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result["warnings"])

    if result["errors"]:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result["errors"])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate synthetic image and labels dataset outputs.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root, such as data/synthetic/phase_2g.")
    args = parser.parse_args(argv)

    result = validate_dataset(args.dataset)
    print(format_validation_report(result))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
