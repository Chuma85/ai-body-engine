import argparse
import csv
from pathlib import Path
from typing import Any

from synthetic.generator.generate_dataset import LABEL_COLUMNS

MEASUREMENT_COLUMNS = [
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
]


def validate_dataset(labels_csv_path: str) -> dict[str, Any]:
    labels_path = Path(labels_csv_path)
    result: dict[str, Any] = {
        "valid": False,
        "row_count": 0,
        "missing_files": [],
        "missing_columns": [],
    }

    if not labels_path.exists():
        result["missing_files"].append(str(labels_path))
        return result

    with labels_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        result["missing_columns"] = [column for column in LABEL_COLUMNS if column not in columns]

        if result["missing_columns"]:
            return result

        rows = list(reader)

    result["row_count"] = len(rows)
    missing_files: list[str] = []
    has_missing_measurements = False

    for row in rows:
        for image_column in ("front_image_path", "side_image_path"):
            image_path = Path(row[image_column])
            if not image_path.exists():
                missing_files.append(str(image_path))

        for measurement_column in MEASUREMENT_COLUMNS:
            if row[measurement_column] in ("", None):
                has_missing_measurements = True

    result["missing_files"] = missing_files
    result["valid"] = len(rows) > 0 and not missing_files and not has_missing_measurements

    if has_missing_measurements:
        result["missing_measurements"] = True

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a synthetic dataset labels CSV.")
    parser.add_argument("labels_csv_path")
    args = parser.parse_args()

    print(validate_dataset(args.labels_csv_path))


if __name__ == "__main__":
    main()
