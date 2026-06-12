import argparse
import csv
from pathlib import Path
from typing import Any

from synthetic.generator.generate_dataset import LABEL_COLUMNS
from training.measurements.measurement_targets import ALL_MEASUREMENT_TARGETS, profile_type_from_payload, target_available_for_profile

MEASUREMENT_COLUMNS = list(ALL_MEASUREMENT_TARGETS)


def validate_dataset(labels_csv_path: str) -> dict[str, Any]:
    labels_path = Path(labels_csv_path)
    result: dict[str, Any] = {
        "valid": False,
        "row_count": 0,
        "missing_files": [],
        "missing_columns": [],
        "missing_back_files": [],
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

        if _truthy(row.get("has_back")) or row.get("back_image_path"):
            back_image_path = Path(row.get("back_image_path", ""))
            if not row.get("back_image_path") or not back_image_path.exists():
                result["missing_back_files"].append(str(back_image_path))
                missing_files.append(str(back_image_path))

        profile_type = profile_type_from_payload(row)
        for measurement_column in MEASUREMENT_COLUMNS:
            if not target_available_for_profile(measurement_column, profile_type):
                continue
            if row.get(measurement_column) in ("", None) and not _has_legacy_measurement_fallback(row, measurement_column):
                has_missing_measurements = True

    result["missing_files"] = missing_files
    result["valid"] = len(rows) > 0 and not missing_files and not has_missing_measurements

    if has_missing_measurements:
        result["missing_measurements"] = True

    return result


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _has_legacy_measurement_fallback(row: dict[str, str], measurement_column: str) -> bool:
    fallback_sources = {
        "shoulder_width_cm": ("shoulder_cm",),
        "abdomen_cm": ("waist_cm",),
        "stomach_cm": ("waist_cm",),
        "outseam_cm": ("inseam_cm",),
        "sleeve_shoulder_to_wrist_cm": ("sleeve_cm",),
        "bicep_cm": ("chest_cm",),
        "forearm_cm": ("chest_cm",),
        "wrist_cm": ("neck_cm",),
        "knee_cm": ("thigh_cm",),
        "ankle_cm": ("calf_cm",),
    }
    return any(row.get(source) not in ("", None) for source in fallback_sources.get(measurement_column, ()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a synthetic dataset labels CSV.")
    parser.add_argument("labels_csv_path")
    args = parser.parse_args()

    print(validate_dataset(args.labels_csv_path))


if __name__ == "__main__":
    main()
