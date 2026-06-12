from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any


REQUIRED_VIEWS = ("front", "side", "back")
REQUIRED_MEASUREMENTS = (
    "height",
    "weight",
    "bust_chest",
    "waist",
    "hips",
    "shoulder",
    "dress_length",
)
SUPPORTED_MEASUREMENT_SCHEMA_VERSIONS = {"field-measurements-v1"}
SUPPORTED_IMAGE_QUALITY_SCHEMA_VERSIONS = {"image-quality-v1"}
MANIFEST_FILENAME = "dataset_export_manifest.json"


class RealWorldDatasetIngestionError(ValueError):
    pass


@dataclass(frozen=True)
class RealWorldDatasetImportResult:
    dataset_version: str
    source_export_id: str
    source_app_version: str
    record_count: int
    validation_status: str
    approved_for_training: bool
    incoming_dir: Path
    processed_dir: Path
    registry_path: Path


def ingest_real_world_training_candidate(
    dataset_version: str,
    *,
    incoming_root: str | Path = "data/real_world/incoming",
    processed_root: str | Path = "data/real_world/processed",
    registry_path: str | Path = "data/real_world/dataset_registry.json",
    import_timestamp: str = "1970-01-01T00:00:00Z",
) -> RealWorldDatasetImportResult:
    incoming_dir = Path(incoming_root) / dataset_version
    processed_dir = Path(processed_root) / dataset_version
    registry_file = Path(registry_path)
    package = load_and_validate_export_package(incoming_dir, dataset_version)
    manifest = package["manifest"]
    records = package["records"]

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    (processed_dir / "images").mkdir(parents=True, exist_ok=True)

    label_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {
        "dataset_version": dataset_version,
        "source_export_id": _value(manifest, "export_id", "exportId"),
        "records": [],
    }

    for record in records:
        participant_code = str(_value(record, "participant_code", "participantCode"))
        image_refs = _record_images_by_pose(record)
        for pose in REQUIRED_VIEWS:
            image = image_refs[pose]
            source_path = _resolve_image_path(incoming_dir, image)
            target_name = f"{participant_code}_{pose}{source_path.suffix or '.jpg'}"
            shutil.copyfile(source_path, processed_dir / "images" / target_name)
            image["processed_path"] = f"images/{target_name}"

        label_row = {
            "participant_code": participant_code,
            "dataset_candidate_id": _value(record, "dataset_candidate_id", "datasetCandidateId"),
        }
        for measurement in record["measurements"]:
            label_row[str(measurement["key"])] = _value(measurement, "value_cm", "valueCm")
        label_rows.append(label_row)

        for image in record["images"]:
            sub_scores = image.get("quality_sub_scores") or image.get("qualitySubScores") or {}
            quality_rows.append(
                {
                    "participant_code": participant_code,
                    "pose_type": _value(image, "pose_type", "poseType"),
                    "overall_quality_score": _value(sub_scores, "overall_quality_score", "overallQualityScore")
                    or _value(image, "quality_score", "qualityScore")
                    or 0,
                    "framing_score": _value(sub_scores, "framing_score", "framingScore") or 0,
                    "lighting_score": _value(sub_scores, "lighting_score", "lightingScore") or 0,
                    "sharpness_score": _value(sub_scores, "sharpness_score", "sharpnessScore") or 0,
                    "pose_score": _value(sub_scores, "pose_score", "poseScore") or 0,
                    "full_body_visibility_score": _value(sub_scores, "full_body_visibility_score", "fullBodyVisibilityScore") or 0,
                    "duplicate_risk_score": _value(sub_scores, "duplicate_risk_score", "duplicateRiskScore") or 0,
                }
            )

        lineage["records"].append(
            {
                "dataset_candidate_id": _value(record, "dataset_candidate_id", "datasetCandidateId"),
                "participant_code": participant_code,
                "lineage": record["lineage"],
                "quality": record.get("quality", {}),
            }
        )

    _write_csv(processed_dir / "labels.csv", label_rows)
    _write_csv(processed_dir / "quality_scores.csv", quality_rows)
    _write_json(processed_dir / "metadata.json", {"manifest": manifest, "approved_only": True})
    _write_json(processed_dir / "lineage.json", lineage)

    registry_entry = {
        "approved_for_training": True,
        "dataset_version": dataset_version,
        "import_timestamp": import_timestamp,
        "record_count": len(records),
        "source_app_version": _value(manifest, "app_version", "appVersion"),
        "source_export_id": _value(manifest, "export_id", "exportId"),
        "validation_status": "ready_for_training",
    }
    _upsert_registry_entry(registry_file, registry_entry)

    return RealWorldDatasetImportResult(
        approved_for_training=True,
        dataset_version=dataset_version,
        incoming_dir=incoming_dir,
        processed_dir=processed_dir,
        record_count=len(records),
        registry_path=registry_file,
        source_app_version=_value(manifest, "app_version", "appVersion"),
        source_export_id=_value(manifest, "export_id", "exportId"),
        validation_status="ready_for_training",
    )


def load_and_validate_export_package(incoming_dir: str | Path, dataset_version: str | None = None) -> dict[str, Any]:
    incoming_path = Path(incoming_dir)
    manifest_path = incoming_path / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RealWorldDatasetIngestionError(f"Missing {MANIFEST_FILENAME}.")

    manifest = _read_json(manifest_path)
    records_path = incoming_path / "records.json"
    if not records_path.exists():
        raise RealWorldDatasetIngestionError("Missing records.json.")
    records_payload = _read_json(records_path)
    records = records_payload.get("records") if isinstance(records_payload, dict) else None
    if not isinstance(records, list) or not records:
        raise RealWorldDatasetIngestionError("records.json must include a non-empty records array.")

    _validate_manifest(manifest, dataset_version)
    for index, record in enumerate(records, start=1):
        _validate_record(incoming_path, manifest, record, index)

    if _value(manifest, "session_count", "sessionCount") != len(records):
        raise RealWorldDatasetIngestionError("Manifest session_count does not match records.json.")

    return {"manifest": manifest, "records": records}


def _validate_manifest(manifest: dict[str, Any], dataset_version: str | None) -> None:
    required = (
        "export_id",
        "dataset_version",
        "export_timestamp",
        "exported_by_admin_id",
        "session_count",
        "image_count",
        "measurement_count",
        "consent_version",
        "measurement_schema_version",
        "image_quality_schema_version",
        "app_version",
        "approved_only",
    )
    for key in required:
        if _value(manifest, key, _to_camel(key)) is None:
            raise RealWorldDatasetIngestionError(f"Manifest missing {key}.")
    if dataset_version and _value(manifest, "dataset_version", "datasetVersion") != dataset_version:
        raise RealWorldDatasetIngestionError("Manifest dataset_version does not match incoming folder.")
    if _value(manifest, "approved_only", "approvedOnly") is not True:
        raise RealWorldDatasetIngestionError("Manifest must declare approved_only true.")
    if _value(manifest, "measurement_schema_version", "measurementSchemaVersion") not in SUPPORTED_MEASUREMENT_SCHEMA_VERSIONS:
        raise RealWorldDatasetIngestionError("Unsupported measurement schema version.")
    if _value(manifest, "image_quality_schema_version", "imageQualitySchemaVersion") not in SUPPORTED_IMAGE_QUALITY_SCHEMA_VERSIONS:
        raise RealWorldDatasetIngestionError("Unsupported image quality schema version.")


def _validate_record(incoming_dir: Path, manifest: dict[str, Any], record: dict[str, Any], index: int) -> None:
    if _value(record, "consent_status", "consentStatus") != "granted":
        raise RealWorldDatasetIngestionError(f"Record {index} is missing granted consent metadata.")
    if _value(record, "dataset_version", "datasetVersion") != _value(manifest, "dataset_version", "datasetVersion"):
        raise RealWorldDatasetIngestionError(f"Record {index} dataset_version does not match manifest.")
    quality = record.get("quality", {}) if isinstance(record.get("quality"), dict) else {}
    if _value(quality, "readiness_state", "readinessState") not in {"approved_for_export", "exported"}:
        raise RealWorldDatasetIngestionError(f"Record {index} is not approved for export.")

    images_by_pose = _record_images_by_pose(record)
    missing_views = [pose for pose in REQUIRED_VIEWS if pose not in images_by_pose]
    if missing_views:
        raise RealWorldDatasetIngestionError(f"Record {index} missing required views: {', '.join(missing_views)}.")
    for pose, image in images_by_pose.items():
        if _value(image, "review_status", "reviewStatus") != "approved":
            raise RealWorldDatasetIngestionError(f"Record {index} {pose} image is not approved.")
        if not _resolve_image_path(incoming_dir, image).exists():
            raise RealWorldDatasetIngestionError(f"Record {index} {pose} image file is missing.")

    measurements = record.get("measurements")
    if not isinstance(measurements, list):
        raise RealWorldDatasetIngestionError(f"Record {index} missing measurements.")
    approved_measurements = {
        measurement.get("key")
        for measurement in measurements
        if _value(measurement, "review_status", "reviewStatus") == "approved"
        and _value(measurement, "value_cm", "valueCm") is not None
    }
    missing_measurements = [key for key in REQUIRED_MEASUREMENTS if key not in approved_measurements]
    if missing_measurements:
        raise RealWorldDatasetIngestionError(
            f"Record {index} missing required measurements: {', '.join(missing_measurements)}."
        )
    if not record.get("lineage"):
        raise RealWorldDatasetIngestionError(f"Record {index} missing lineage.")


def _record_images_by_pose(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    images = record.get("images")
    if not isinstance(images, list):
        raise RealWorldDatasetIngestionError("Record images must be an array.")
    return {str(_value(image, "pose_type", "poseType")): image for image in images if isinstance(image, dict)}


def _resolve_image_path(incoming_dir: Path, image: dict[str, Any]) -> Path:
    value = _value(image, "image_path", "imagePath", "local_path", "localPath", "image_asset_id", "imageAssetId")
    if not value:
        return incoming_dir / "__missing__"
    path = Path(str(value))
    return path if path.is_absolute() else incoming_dir / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise RealWorldDatasetIngestionError(f"{path.name} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_registry_entry(path: Path, entry: dict[str, Any]) -> None:
    registry = {"datasets": []}
    if path.exists():
        registry = _read_json(path)
    datasets = [item for item in registry.get("datasets", []) if item.get("dataset_version") != entry["dataset_version"]]
    datasets.append(entry)
    registry["datasets"] = sorted(datasets, key=lambda item: item["dataset_version"])
    _write_json(path, registry)


def _value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
