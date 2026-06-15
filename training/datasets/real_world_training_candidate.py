from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import csv
import json
from pathlib import Path
import shutil
from typing import Any


MANIFEST_FILENAME = "dataset_export_manifest.json"
DEFAULT_REGISTRY_PATH = Path("dataset_registry/datasets.json")
DEFAULT_REPORT_PATH = Path("reports/dataset_validation_report.json")
DEFAULT_REAL_WORLD_ROOT = Path("data/real_world")

SUPPORTED_SCHEMA_VERSIONS = {"real-world-dataset-export-v1"}
SUPPORTED_MEASUREMENT_SCHEMA_VERSIONS = {"field-measurements-v1"}
SUPPORTED_IMAGE_QUALITY_SCHEMA_VERSIONS = {"image-quality-v1"}
DATASET_STATUSES = ("imported", "validating", "validated", "rejected", "approved_for_training")
TRAINING_STATUS_NOT_STARTED = "not_started"
MIN_QUALITY_SCORE = 70.0

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
MEASUREMENT_ALIASES = {
    "chest": "bust_chest",
    "hip": "hips",
}


class RealWorldDatasetIngestionError(ValueError):
    pass


@dataclass(frozen=True)
class RealWorldDatasetImportResult:
    dataset_version: str
    source_export_id: str
    source_app_version: str
    record_count: int
    validation_status: str
    training_status: str
    status: str
    quality_score: float
    approved_for_training: bool
    incoming_dir: Path
    validated_dir: Path
    registry_path: Path
    report_path: Path

    @property
    def processed_dir(self) -> Path:
        return self.validated_dir


def import_real_world_dataset(
    manifest_path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    real_world_root: str | Path = DEFAULT_REAL_WORLD_ROOT,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    import_timestamp: str | None = None,
    validation_timestamp: str | None = None,
) -> RealWorldDatasetImportResult:
    import_time = import_timestamp or _utc_now()
    validation_time = validation_timestamp or import_time
    manifest_file = Path(manifest_path)
    registry_file = Path(registry_path)
    staging_root = Path(real_world_root)
    report_file = Path(report_path)
    _ensure_staging_dirs(staging_root)

    report = _new_report(manifest_file, import_time, validation_time)
    if not manifest_file.exists():
        report["schema_mismatches"].append(f"Manifest does not exist: {manifest_file}")
        _finalize_report(report, report_file, "rejected", "rejected")
        raise RealWorldDatasetIngestionError("Manifest does not exist.")

    manifest = _read_json(manifest_file)
    package_dir = manifest_file.parent
    dataset_version = str(_manifest_value(manifest, "dataset_version", "datasetVersion") or "")
    source_export_id = str(
        _manifest_value(manifest, "source_export_id", "sourceExportId", "export_id", "exportId") or ""
    )
    source_app_version = str(_manifest_value(manifest, "source_app_version", "sourceAppVersion", "app_version", "appVersion") or "")
    source_system = str(_manifest_value(manifest, "source_system", "sourceSystem") or "CUSTOM-FASHION-MARKETPLACE")
    source_dataset_version = str(
        _manifest_value(manifest, "source_dataset_version", "sourceDatasetVersion", "dataset_version", "datasetVersion") or ""
    )
    schema_version = str(_manifest_value(manifest, "schema_version", "schemaVersion") or "")

    report["dataset_version"] = dataset_version
    report["source_export_id"] = source_export_id
    report["source_system"] = source_system
    report["schema_version"] = schema_version

    if not dataset_version:
        report["schema_mismatches"].append("Manifest missing dataset_version.")
    if not source_export_id:
        report["schema_mismatches"].append("Manifest missing source_export_id/export_id.")
    if dataset_version and source_export_id and _registry_has_duplicate(registry_file, dataset_version, source_export_id):
        report["schema_mismatches"].append(
            f"Duplicate import for dataset_version={dataset_version} or source_export_id={source_export_id}."
        )
        _finalize_report(report, report_file, "rejected", "rejected")
        raise RealWorldDatasetIngestionError("Duplicate dataset import.")

    _upsert_registry_entry(
        registry_file,
        _registry_entry(
            manifest=manifest,
            report=report,
            import_timestamp=import_time,
            validation_timestamp=None,
            status="imported",
            validation_status="imported",
            training_status=TRAINING_STATUS_NOT_STARTED,
            validated_path=None,
            rejected_path=None,
            report_path=report_file,
        ),
    )
    _upsert_registry_entry(
        registry_file,
        _registry_entry(
            manifest=manifest,
            report=report,
            import_timestamp=import_time,
            validation_timestamp=None,
            status="validating",
            validation_status="validating",
            training_status=TRAINING_STATUS_NOT_STARTED,
            validated_path=None,
            rejected_path=None,
            report_path=report_file,
        ),
    )

    records = _load_records(package_dir, manifest, report)
    _validate_manifest(manifest, report)
    quality_scores = _validate_records(package_dir, manifest, records, report)
    _validate_counts(manifest, records, report)
    quality_summary = _quality_summary(records, quality_scores, report)
    report["quality_summary"] = quality_summary
    report["quality_score"] = quality_summary["quality_score"]
    report["record_count"] = len(records)

    is_valid = _report_has_no_failures(report)
    validation_status = "validated" if is_valid else "rejected"
    status = "approved_for_training" if is_valid else "rejected"
    target_dir = (
        staging_root / "validated" / _safe_name(dataset_version or source_export_id)
        if is_valid
        else staging_root / "rejected" / _safe_name(dataset_version or source_export_id or "unknown_export")
    )
    if package_dir.exists():
        _copy_package(package_dir, target_dir)
        archive_dir = staging_root / "archived" / _safe_name(f"{dataset_version or 'unknown'}_{source_export_id or 'unknown'}")
        _copy_package(package_dir, archive_dir)

    _finalize_report(report, report_file, validation_status, status)
    _write_lineage_file(
        target_dir,
        {
            "source_export_id": source_export_id,
            "source_app_version": source_app_version,
            "source_dataset_version": source_dataset_version,
            "source_system": source_system,
            "import_timestamp": import_time,
            "validation_timestamp": validation_time,
        },
    )

    registry_entry = _registry_entry(
        manifest=manifest,
        report=report,
        import_timestamp=import_time,
        validation_timestamp=validation_time,
        status=status,
        validation_status=validation_status,
        training_status=TRAINING_STATUS_NOT_STARTED,
        validated_path=target_dir if is_valid else None,
        rejected_path=target_dir if not is_valid else None,
        report_path=report_file,
    )
    _upsert_registry_entry(registry_file, registry_entry)

    if not is_valid:
        raise RealWorldDatasetIngestionError("Dataset export failed validation.")

    return RealWorldDatasetImportResult(
        approved_for_training=True,
        dataset_version=dataset_version,
        incoming_dir=package_dir,
        quality_score=float(quality_summary["quality_score"]),
        record_count=len(records),
        registry_path=registry_file,
        report_path=report_file,
        source_app_version=source_app_version,
        source_export_id=source_export_id,
        status=status,
        training_status=TRAINING_STATUS_NOT_STARTED,
        validated_dir=target_dir,
        validation_status=validation_status,
    )


def ingest_real_world_training_candidate(
    dataset_version: str,
    *,
    incoming_root: str | Path = "data/real_world/incoming",
    processed_root: str | Path = "data/real_world/validated",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    import_timestamp: str = "1970-01-01T00:00:00Z",
) -> RealWorldDatasetImportResult:
    incoming_dir = Path(incoming_root) / dataset_version
    real_world_root = Path(processed_root).parent if Path(processed_root).name == "validated" else Path(processed_root).parent
    return import_real_world_dataset(
        incoming_dir / MANIFEST_FILENAME,
        import_timestamp=import_timestamp,
        real_world_root=real_world_root,
        registry_path=registry_path,
    )


def load_and_validate_export_package(incoming_dir: str | Path, dataset_version: str | None = None) -> dict[str, Any]:
    package_dir = Path(incoming_dir)
    report = _new_report(package_dir / MANIFEST_FILENAME, _utc_now(), _utc_now())
    manifest = _read_json(package_dir / MANIFEST_FILENAME)
    records = _load_records(package_dir, manifest, report)
    _validate_manifest(manifest, report)
    _validate_records(package_dir, manifest, records, report)
    _validate_counts(manifest, records, report)
    if dataset_version and _manifest_value(manifest, "dataset_version", "datasetVersion") != dataset_version:
        report["schema_mismatches"].append("Manifest dataset_version does not match incoming folder.")
    if not _report_has_no_failures(report):
        raise RealWorldDatasetIngestionError(_first_report_failure(report))
    return {"manifest": manifest, "records": records}


def load_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        return {"datasets": []}
    registry = _read_json(path)
    datasets = registry.get("datasets", [])
    if not isinstance(datasets, list):
        return {"datasets": []}
    return {"datasets": datasets}


def format_dataset_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> str:
    registry = load_registry(registry_path)
    rows = sorted(registry["datasets"], key=lambda item: str(item.get("dataset_version", "")))
    if not rows:
        return "No datasets registered."
    header = f"{'dataset_version':<20} {'status':<22} {'record_count':>12} {'quality_score':>13}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('dataset_version', '')):<20} "
            f"{str(row.get('status', row.get('validation_status', ''))):<22} "
            f"{int(row.get('record_count', row.get('participant_count', 0)) or 0):>12} "
            f"{float(row.get('quality_score', 0) or 0):>13.1f}"
        )
    return "\n".join(lines)


def _ensure_staging_dirs(root: Path) -> None:
    for name in ("incoming", "validated", "rejected", "archived"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _new_report(manifest_path: Path, import_timestamp: str, validation_timestamp: str) -> dict[str, Any]:
    return {
        "report_version": "real-world-dataset-validation-v1",
        "manifest_path": str(manifest_path),
        "dataset_version": None,
        "source_export_id": None,
        "source_system": None,
        "schema_version": None,
        "import_timestamp": import_timestamp,
        "validation_timestamp": validation_timestamp,
        "status": "validating",
        "validation_status": "validating",
        "record_count": 0,
        "quality_score": 0.0,
        "quality_summary": {},
        "missing_images": [],
        "missing_labels": [],
        "duplicate_participants": [],
        "low_quality_records": [],
        "invalid_measurements": [],
        "schema_mismatches": [],
        "missing_metadata": [],
        "missing_consent_metadata": [],
        "missing_views": [],
        "rejected_records": [],
    }


def _finalize_report(report: dict[str, Any], report_path: Path, validation_status: str, status: str) -> None:
    report["validation_status"] = validation_status
    report["status"] = status
    _write_json(report_path, report)


def _validate_manifest(manifest: dict[str, Any], report: dict[str, Any]) -> None:
    required = (
        ("dataset_version", "datasetVersion"),
        ("source_export_id", "sourceExportId", "export_id", "exportId"),
        ("export_timestamp", "exportTimestamp"),
        ("schema_version", "schemaVersion"),
        ("image_count", "imageCount"),
        ("measurement_count", "measurementCount"),
        ("participant_count", "participantCount", "session_count", "sessionCount"),
        ("approved_only", "approvedOnly"),
    )
    for keys in required:
        if _manifest_value(manifest, *keys) is None:
            report["schema_mismatches"].append(f"Manifest missing {keys[0]}.")

    schema_version = _manifest_value(manifest, "schema_version", "schemaVersion")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        report["schema_mismatches"].append(f"Unsupported schema version: {schema_version}.")
    if _manifest_value(manifest, "approved_only", "approvedOnly") is not True:
        report["schema_mismatches"].append("Manifest approved_only must be true.")

    measurement_schema = _manifest_value(manifest, "measurement_schema_version", "measurementSchemaVersion")
    if measurement_schema and measurement_schema not in SUPPORTED_MEASUREMENT_SCHEMA_VERSIONS:
        report["schema_mismatches"].append(f"Unsupported measurement schema version: {measurement_schema}.")
    image_schema = _manifest_value(manifest, "image_quality_schema_version", "imageQualitySchemaVersion")
    if image_schema and image_schema not in SUPPORTED_IMAGE_QUALITY_SCHEMA_VERSIONS:
        report["schema_mismatches"].append(f"Unsupported image quality schema version: {image_schema}.")

    rejected_records = _manifest_value(manifest, "rejected_records", "rejectedRecords")
    if isinstance(rejected_records, list) and rejected_records:
        report["rejected_records"].append("Manifest includes rejected_records.")
    excluded = _manifest_value(manifest, "excluded_records", "excludedRecords")
    if isinstance(excluded, dict):
        rejected_count = excluded.get("rejected") or excluded.get("rejectedRecords") or 0
        if rejected_count:
            report["rejected_records"].append("Manifest excluded_records reports rejected entries.")


def _load_records(package_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    records = _manifest_value(manifest, "records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    records_path = _resolve_package_path(package_dir, _manifest_value(manifest, "records_path", "recordsPath") or "records.json")
    if not records_path.exists():
        report["schema_mismatches"].append(f"Missing records file: {_relative_or_name(package_dir, records_path)}.")
        return []
    payload = _read_json(records_path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        report["schema_mismatches"].append("Records file must include a non-empty records array.")
        return []
    return [record for record in records if isinstance(record, dict)]


def _validate_records(
    package_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[float]:
    label_path = _first_existing_package_file(
        package_dir,
        _manifest_value(manifest, "labels_path", "labelsPath", "label_path", "labelPath"),
        ("labels.json", "labels.csv"),
    )
    metadata_path = _first_existing_package_file(
        package_dir,
        _manifest_value(manifest, "metadata_path", "metadataPath"),
        ("metadata.json",),
    )
    consent_path = _first_existing_package_file(
        package_dir,
        _manifest_value(manifest, "consent_metadata_path", "consentMetadataPath"),
        ("consent_metadata.json",),
    )

    if label_path is None:
        report["missing_labels"].append("Missing labels file.")
    if metadata_path is None:
        report["missing_metadata"].append("Missing metadata file.")
    if consent_path is None:
        report["missing_consent_metadata"].append("Missing consent metadata file.")

    labeled_participants = _label_participants(label_path) if label_path else set()
    participants: set[str] = set()
    quality_scores: list[float] = []
    image_count = 0
    measurement_count = 0

    for index, record in enumerate(records, start=1):
        participant_id = str(
            _value(record, "participant_code", "participantCode", "participant_id", "participantId", "tester_id", "testerId", "id")
            or f"record_{index}"
        )
        if participant_id in participants:
            report["duplicate_participants"].append(participant_id)
        participants.add(participant_id)
        if labeled_participants and participant_id not in labeled_participants:
            report["missing_labels"].append(f"{participant_id}: missing label row.")

        if _value(record, "review_status", "reviewStatus", "status") == "rejected":
            report["rejected_records"].append(f"{participant_id}: record status is rejected.")
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        readiness = _value(quality, "readiness_state", "readinessState")
        if readiness and readiness not in {"approved_for_export", "exported", "approved"}:
            report["rejected_records"].append(f"{participant_id}: readiness_state={readiness}.")

        consent_status = _value(record, "consent_status", "consentStatus")
        consent = record.get("consent") if isinstance(record.get("consent"), dict) else {}
        consent_granted = _value(consent, "granted")
        if consent_status != "granted" and consent_granted is not True:
            report["missing_consent_metadata"].append(f"{participant_id}: consent is not granted.")

        images_by_view = _record_images_by_view(record)
        missing_views = [view for view in REQUIRED_VIEWS if view not in images_by_view]
        for view in missing_views:
            report["missing_views"].append(f"{participant_id}: missing {view} view.")
        for view, image in images_by_view.items():
            if _value(image, "review_status", "reviewStatus") == "rejected":
                report["rejected_records"].append(f"{participant_id}: {view} image status is rejected.")
            image_path = _resolve_image_path(package_dir, image)
            if not image_path.exists():
                report["missing_images"].append(f"{participant_id}: {view} image missing at {_relative_or_name(package_dir, image_path)}.")
            image_count += 1

        measurement_map = _measurement_values(record)
        for key in REQUIRED_MEASUREMENTS:
            value = measurement_map.get(key)
            if value is None:
                report["invalid_measurements"].append(f"{participant_id}: missing {key}.")
                continue
            if not isinstance(value, (int, float)) or value <= 0:
                report["invalid_measurements"].append(f"{participant_id}: invalid {key}={value}.")
            measurement_count += 1

        record_quality = _record_quality_score(record, images_by_view)
        quality_scores.append(record_quality)
        if record_quality < MIN_QUALITY_SCORE:
            report["low_quality_records"].append(f"{participant_id}: quality_score={record_quality:.1f}.")

    report["_actual_image_count"] = image_count
    report["_actual_measurement_count"] = measurement_count
    report["_actual_participant_count"] = len(participants)
    return quality_scores


def _validate_counts(manifest: dict[str, Any], records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    expected_images = _as_int(_manifest_value(manifest, "image_count", "imageCount"))
    expected_measurements = _as_int(_manifest_value(manifest, "measurement_count", "measurementCount"))
    expected_participants = _as_int(
        _manifest_value(manifest, "participant_count", "participantCount", "session_count", "sessionCount")
    )
    actual_images = int(report.pop("_actual_image_count", 0))
    actual_measurements = int(report.pop("_actual_measurement_count", 0))
    actual_participants = int(report.pop("_actual_participant_count", len(records)))
    if expected_images is not None and expected_images != actual_images:
        report["schema_mismatches"].append(f"image_count mismatch: manifest={expected_images}, actual={actual_images}.")
    if expected_measurements is not None and expected_measurements != actual_measurements:
        report["schema_mismatches"].append(
            f"measurement_count mismatch: manifest={expected_measurements}, actual={actual_measurements}."
        )
    if expected_participants is not None and expected_participants != actual_participants:
        report["schema_mismatches"].append(
            f"participant_count mismatch: manifest={expected_participants}, actual={actual_participants}."
        )


def _quality_summary(records: list[dict[str, Any]], scores: list[float], report: dict[str, Any]) -> dict[str, Any]:
    score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {
        "quality_score": score,
        "record_count": len(records),
        "missing_image_count": len(report["missing_images"]),
        "missing_label_count": len(report["missing_labels"]),
        "duplicate_participant_count": len(report["duplicate_participants"]),
        "low_quality_record_count": len(report["low_quality_records"]),
        "invalid_measurement_count": len(report["invalid_measurements"]),
        "schema_mismatch_count": len(report["schema_mismatches"]),
    }


def _registry_entry(
    *,
    manifest: dict[str, Any],
    report: dict[str, Any],
    import_timestamp: str,
    validation_timestamp: str | None,
    status: str,
    validation_status: str,
    training_status: str,
    validated_path: Path | None,
    rejected_path: Path | None,
    report_path: Path,
) -> dict[str, Any]:
    dataset_version = str(_manifest_value(manifest, "dataset_version", "datasetVersion") or report.get("dataset_version") or "")
    source_export_id = str(
        _manifest_value(manifest, "source_export_id", "sourceExportId", "export_id", "exportId")
        or report.get("source_export_id")
        or ""
    )
    source_app_version = str(
        _manifest_value(manifest, "source_app_version", "sourceAppVersion", "app_version", "appVersion") or ""
    )
    source_dataset_version = str(
        _manifest_value(manifest, "source_dataset_version", "sourceDatasetVersion", "dataset_version", "datasetVersion") or ""
    )
    entry = {
        "dataset_version": dataset_version,
        "source_system": _manifest_value(manifest, "source_system", "sourceSystem") or "CUSTOM-FASHION-MARKETPLACE",
        "source_export_id": source_export_id,
        "export_timestamp": _manifest_value(manifest, "export_timestamp", "exportTimestamp"),
        "import_timestamp": import_timestamp,
        "schema_version": _manifest_value(manifest, "schema_version", "schemaVersion"),
        "image_count": _as_int(_manifest_value(manifest, "image_count", "imageCount")) or 0,
        "measurement_count": _as_int(_manifest_value(manifest, "measurement_count", "measurementCount")) or 0,
        "participant_count": _as_int(
            _manifest_value(manifest, "participant_count", "participantCount", "session_count", "sessionCount")
        )
        or 0,
        "record_count": int(report.get("record_count") or 0),
        "quality_summary": report.get("quality_summary") or {},
        "quality_score": float(report.get("quality_score") or 0),
        "validation_status": validation_status,
        "training_status": training_status,
        "status": status,
        "lineage": {
            "source_export_id": source_export_id,
            "source_app_version": source_app_version,
            "source_dataset_version": source_dataset_version,
            "import_timestamp": import_timestamp,
            "validation_timestamp": validation_timestamp,
        },
        "report_path": str(report_path),
    }
    if validated_path:
        entry["validated_path"] = str(validated_path)
    if rejected_path:
        entry["rejected_path"] = str(rejected_path)
    return entry


def _upsert_registry_entry(path: Path, entry: dict[str, Any]) -> None:
    registry = load_registry(path)
    datasets = [
        item
        for item in registry["datasets"]
        if item.get("dataset_version") != entry["dataset_version"]
        and item.get("source_export_id") != entry["source_export_id"]
    ]
    datasets.append(entry)
    registry["datasets"] = sorted(datasets, key=lambda item: str(item.get("dataset_version", "")))
    _write_json(path, registry)


def _registry_has_duplicate(path: Path, dataset_version: str, source_export_id: str) -> bool:
    registry = load_registry(path)
    return any(
        item.get("dataset_version") == dataset_version or item.get("source_export_id") == source_export_id
        for item in registry["datasets"]
    )


def _report_has_no_failures(report: dict[str, Any]) -> bool:
    failure_keys = (
        "missing_images",
        "missing_labels",
        "duplicate_participants",
        "low_quality_records",
        "invalid_measurements",
        "schema_mismatches",
        "missing_metadata",
        "missing_consent_metadata",
        "missing_views",
        "rejected_records",
    )
    return all(not report[key] for key in failure_keys)


def _first_report_failure(report: dict[str, Any]) -> str:
    for key, value in report.items():
        if isinstance(value, list) and value:
            return str(value[0])
    return "Dataset export failed validation."


def _copy_package(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _write_lineage_file(target_dir: Path, lineage: dict[str, Any]) -> None:
    if target_dir.exists():
        _write_json(target_dir / "lineage.json", lineage)


def _record_images_by_view(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    images = record.get("images")
    if isinstance(images, list):
        return _image_list_by_view(images)
    if isinstance(images, dict):
        return _image_mapping_by_view(images)

    photos = record.get("photos")
    if isinstance(photos, dict):
        return _image_mapping_by_view(photos)
    return {}


def _image_list_by_view(images: list[Any]) -> dict[str, dict[str, Any]]:
    by_view: dict[str, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        view = _value(image, "view", "pose_type", "poseType", "viewType")
        if view:
            by_view[str(view)] = image
    return by_view


def _image_mapping_by_view(images: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_view: dict[str, dict[str, Any]] = {}
    for view, image in images.items():
        if not view:
            continue
        view_name = str(view)
        if isinstance(image, dict):
            image_payload = dict(image)
            image_payload.setdefault("poseType", view_name)
            by_view[view_name] = image_payload
        elif image not in (None, ""):
            by_view[view_name] = {"imageAssetId": str(image), "poseType": view_name}
    return by_view


def _measurement_values(record: dict[str, Any]) -> dict[str, float | None]:
    measurements = record.get("measurements")
    values: dict[str, float | None] = {}
    if isinstance(measurements, dict):
        for key, value in measurements.items():
            values[str(key)] = _as_float(value)
        return _with_measurement_aliases(values)
    if isinstance(measurements, list):
        for measurement in measurements:
            if not isinstance(measurement, dict):
                continue
            if _value(measurement, "review_status", "reviewStatus") == "rejected":
                values[str(_value(measurement, "key") or "")] = None
                continue
            key = _value(measurement, "key", "name")
            if key:
                values[str(key)] = _as_float(_value(measurement, "value_cm", "valueCm", "value"))
        return _with_measurement_aliases(values)

    measurements_cm = record.get("measurements_cm")
    if isinstance(measurements_cm, dict):
        for key, value in measurements_cm.items():
            values[str(key)] = _as_float(value)
    return _with_measurement_aliases(values)


def _with_measurement_aliases(values: dict[str, float | None]) -> dict[str, float | None]:
    for source_key, target_key in MEASUREMENT_ALIASES.items():
        if target_key not in values and source_key in values:
            values[target_key] = values[source_key]
    return values


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("session_id", "sessionId", "tester_id", "testerId", "source", "created_at", "createdAt"):
        value = record.get(key)
        if value not in (None, ""):
            metadata[key] = value
    garment_context = record.get("garment_context")
    if isinstance(garment_context, dict):
        metadata["garment_context"] = dict(garment_context)
    return metadata


def _record_quality_score(record: dict[str, Any], images_by_view: dict[str, dict[str, Any]]) -> float:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    score = _as_float(
        _value(quality, "overall_dataset_readiness_score", "overallDatasetReadinessScore", "quality_score", "qualityScore")
    )
    if score is not None:
        return score
    image_scores = [
        _as_float(
            _value(
                image.get("qualitySubScores", {}) if isinstance(image.get("qualitySubScores"), dict) else {},
                "overallQualityScore",
                "overall_quality_score",
            )
        )
        or _as_float(_value(image, "quality_score", "qualityScore"))
        for image in images_by_view.values()
    ]
    image_scores = [value for value in image_scores if value is not None]
    return sum(image_scores) / len(image_scores) if image_scores else 100.0


def _resolve_image_path(package_dir: Path, image: dict[str, Any]) -> Path:
    image_value = _value(image, "image_path", "imagePath", "local_path", "localPath", "image_asset_id", "imageAssetId")
    if not image_value:
        return package_dir / "__missing_image__"
    return _resolve_package_path(package_dir, image_value)


def _first_existing_package_file(package_dir: Path, explicit: Any, defaults: tuple[str, ...]) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(defaults)
    for candidate in candidates:
        path = _resolve_package_path(package_dir, candidate)
        if path.exists():
            return path
    return None


def _label_participants(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as labels_file:
            rows = csv.DictReader(labels_file)
            return {
                str(row.get("participant_code") or row.get("participantCode") or row.get("participant_id") or "")
                for row in rows
                if row
            }
    payload = _read_json(path)
    records = payload.get("labels") or payload.get("records") or payload.get("participants")
    if isinstance(records, dict):
        return {str(key) for key in records}
    if isinstance(records, list):
        return {
            str(_value(record, "participant_code", "participantCode", "participant_id", "participantId", "id") or "")
            for record in records
            if isinstance(record, dict)
        }
    return set()


def _resolve_package_path(package_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else package_dir / path


def _relative_or_name(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


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


def _manifest_value(mapping: dict[str, Any], *keys: str) -> Any:
    return _value(mapping, *keys)


def _value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return cleaned or "unknown"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
