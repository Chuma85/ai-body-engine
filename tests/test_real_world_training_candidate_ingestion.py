from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from training.datasets.real_world_training_candidate import (
    RealWorldDatasetIngestionError,
    ingest_real_world_training_candidate,
    load_and_validate_export_package,
)


def test_valid_export_imports_and_registers_dataset(tmp_path: Path) -> None:
    incoming_root = tmp_path / "incoming"
    processed_root = tmp_path / "processed"
    registry_path = tmp_path / "dataset_registry.json"
    package_dir = _write_export_package(incoming_root, "v1")

    result = ingest_real_world_training_candidate(
        "v1",
        incoming_root=incoming_root,
        processed_root=processed_root,
        registry_path=registry_path,
        import_timestamp="2026-06-12T12:00:00Z",
    )

    assert result.dataset_version == "v1"
    assert result.validation_status == "ready_for_training"
    assert result.approved_for_training is True
    assert result.record_count == 1
    assert (processed_root / "v1" / "images" / "P001_front.jpg").exists()
    assert (processed_root / "v1" / "labels.csv").read_text(encoding="utf-8").startswith("participant_code")
    assert (processed_root / "v1" / "metadata.json").exists()
    assert (processed_root / "v1" / "quality_scores.csv").exists()
    assert (processed_root / "v1" / "lineage.json").exists()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["datasets"][0]["dataset_version"] == "v1"
    assert registry["datasets"][0]["source_export_id"] == "export_1"
    assert registry["datasets"][0]["approved_for_training"] is True
    assert package_dir.exists()


def test_missing_consent_fails(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "incoming", "v1")
    _mutate_record(package_dir, lambda record: record.update({"consentStatus": "withdrawn"}))

    with pytest.raises(RealWorldDatasetIngestionError, match="consent"):
        load_and_validate_export_package(package_dir, "v1")


def test_rejected_record_fails(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "incoming", "v1")
    _mutate_record(package_dir, lambda record: record["quality"].update({"readinessState": "rejected"}))

    with pytest.raises(RealWorldDatasetIngestionError, match="approved for export"):
        load_and_validate_export_package(package_dir, "v1")


def test_missing_image_fails(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "incoming", "v1")
    (package_dir / "images" / "back.jpg").unlink()

    with pytest.raises(RealWorldDatasetIngestionError, match="image file is missing"):
        load_and_validate_export_package(package_dir, "v1")


def test_unsupported_schema_fails(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "incoming", "v1")
    manifest_path = package_dir / "dataset_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["measurementSchemaVersion"] = "field-measurements-v999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RealWorldDatasetIngestionError, match="Unsupported measurement schema"):
        load_and_validate_export_package(package_dir, "v1")


def test_missing_measurement_fails(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "incoming", "v1")
    _mutate_record(
        package_dir,
        lambda record: record.update(
            {"measurements": [item for item in record["measurements"] if item["key"] != "waist"]}
        ),
    )

    with pytest.raises(RealWorldDatasetIngestionError, match="missing required measurements"):
        load_and_validate_export_package(package_dir, "v1")


def _write_export_package(incoming_root: Path, dataset_version: str) -> Path:
    package_dir = incoming_root / dataset_version
    image_dir = package_dir / "images"
    image_dir.mkdir(parents=True)
    for pose in ("front", "side", "back"):
        (image_dir / f"{pose}.jpg").write_bytes(f"{pose}-image".encode("utf-8"))

    manifest = {
        "appVersion": "field-data-beta",
        "approvedOnly": True,
        "consentVersion": "field-data-consent-v1",
        "datasetVersion": dataset_version,
        "excludedRecords": {"images": [], "measurements": []},
        "exportId": "export_1",
        "exportTimestamp": "2026-06-12T12:00:00Z",
        "exportedByAdminId": "dataset_admin_1",
        "imageChecksums": {},
        "imageCount": 3,
        "imageQualitySchemaVersion": "image-quality-v1",
        "measurementCount": 7,
        "measurementSchemaVersion": "field-measurements-v1",
        "sessionCount": 1,
    }
    record = {
        "collectorId": "collector_1",
        "consentStatus": "granted",
        "datasetCandidateId": "candidate_1",
        "datasetVersion": dataset_version,
        "images": [
            _image("front"),
            _image("side"),
            _image("back"),
        ],
        "lineage": {"collectionSessionId": "session_1", "imageVersion": 1, "measurementVersion": 1},
        "measurements": [
            {"key": key, "reviewStatus": "approved", "valueCm": 100}
            for key in ("height", "weight", "bust_chest", "waist", "hips", "shoulder", "dress_length")
        ],
        "participantCode": "P001",
        "quality": {"overallDatasetReadinessScore": 98, "readinessState": "approved_for_export"},
        "schemaVersions": {
            "appVersion": "field-data-beta",
            "consentVersion": "field-data-consent-v1",
            "imageQualitySchemaVersion": "image-quality-v1",
            "measurementSchemaVersion": "field-measurements-v1",
        },
    }
    (package_dir / "dataset_export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (package_dir / "records.json").write_text(json.dumps({"records": [record]}), encoding="utf-8")
    return package_dir


def _image(pose: str) -> dict[str, object]:
    return {
        "imageAssetId": f"images/{pose}.jpg",
        "poseType": pose,
        "qualityScore": 100,
        "qualitySubScores": {
            "duplicateRiskScore": 100,
            "framingScore": 100,
            "fullBodyVisibilityScore": 100,
            "lightingScore": 100,
            "overallQualityScore": 100,
            "poseScore": 100,
            "sharpnessScore": 100,
        },
        "reviewStatus": "approved",
    }


def _mutate_record(package_dir: Path, update: Callable[[dict[str, object]], None]) -> None:
    records_path = package_dir / "records.json"
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    update(payload["records"][0])
    records_path.write_text(json.dumps(payload), encoding="utf-8")
