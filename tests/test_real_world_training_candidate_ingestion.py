from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from training.datasets.real_world_training_candidate import (
    RealWorldDatasetIngestionError,
    format_dataset_registry,
    import_real_world_dataset,
    _measurement_values,
    _record_images_by_view,
    _record_metadata,
)


def test_valid_import_registers_approved_training_candidate(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-2026-06-12")
    registry_path = tmp_path / "dataset_registry" / "datasets.json"
    report_path = tmp_path / "reports" / "dataset_validation_report.json"

    result = import_real_world_dataset(
        package_dir / "dataset_export_manifest.json",
        import_timestamp="2026-06-12T12:00:00Z",
        real_world_root=tmp_path / "data" / "real_world",
        registry_path=registry_path,
        report_path=report_path,
    )

    assert result.dataset_version == "rw-2026-06-12"
    assert result.status == "approved_for_training"
    assert result.validation_status == "validated"
    assert result.training_status == "not_started"
    assert result.record_count == 1
    assert result.quality_score == 98.0
    assert (tmp_path / "data" / "real_world" / "validated" / "rw-2026-06-12" / "lineage.json").exists()
    assert (tmp_path / "data" / "real_world" / "archived" / "rw-2026-06-12_export_1").exists()

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["datasets"][0]
    assert entry["dataset_version"] == "rw-2026-06-12"
    assert entry["source_system"] == "CUSTOM-FASHION-MARKETPLACE"
    assert entry["source_export_id"] == "export_1"
    assert entry["schema_version"] == "real-world-dataset-export-v1"
    assert entry["image_count"] == 3
    assert entry["measurement_count"] == 7
    assert entry["participant_count"] == 1
    assert entry["validation_status"] == "validated"
    assert entry["training_status"] == "not_started"
    assert entry["status"] == "approved_for_training"
    assert entry["lineage"]["source_app_version"] == "fashionapp-field-data-beta"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["missing_images"] == []
    assert report["missing_labels"] == []
    assert report["schema_mismatches"] == []


def test_missing_image_rejects_and_reports(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    (package_dir / "images" / "back.jpg").unlink()
    report_path = tmp_path / "reports" / "dataset_validation_report.json"

    with pytest.raises(RealWorldDatasetIngestionError, match="failed validation"):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=tmp_path / "dataset_registry" / "datasets.json",
            report_path=report_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "rejected"
    assert report["missing_images"] == ["P001: back image missing at images\\back.jpg."] or report[
        "missing_images"
    ] == ["P001: back image missing at images/back.jpg."]
    assert (tmp_path / "data" / "real_world" / "rejected" / "rw-1").exists()


def test_missing_label_rejects(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    (package_dir / "labels.json").unlink()

    with pytest.raises(RealWorldDatasetIngestionError):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=tmp_path / "dataset_registry" / "datasets.json",
            report_path=tmp_path / "reports" / "dataset_validation_report.json",
        )


def test_missing_consent_metadata_rejects(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    (package_dir / "consent_metadata.json").unlink()

    with pytest.raises(RealWorldDatasetIngestionError):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=tmp_path / "dataset_registry" / "datasets.json",
            report_path=tmp_path / "reports" / "dataset_validation_report.json",
        )


def test_unsupported_schema_rejects(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    _mutate_manifest(package_dir, lambda manifest: manifest.update({"schemaVersion": "real-world-dataset-export-v999"}))

    with pytest.raises(RealWorldDatasetIngestionError):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=tmp_path / "dataset_registry" / "datasets.json",
            report_path=tmp_path / "reports" / "dataset_validation_report.json",
        )


def test_duplicate_import_rejects_without_overwriting_registry(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    registry_path = tmp_path / "dataset_registry" / "datasets.json"
    report_path = tmp_path / "reports" / "dataset_validation_report.json"
    import_real_world_dataset(
        package_dir / "dataset_export_manifest.json",
        real_world_root=tmp_path / "data" / "real_world",
        registry_path=registry_path,
        report_path=report_path,
    )

    with pytest.raises(RealWorldDatasetIngestionError, match="Duplicate"):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=registry_path,
            report_path=report_path,
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert len(registry["datasets"]) == 1


def test_registry_viewer_shows_dataset_status_and_quality(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1")
    registry_path = tmp_path / "dataset_registry" / "datasets.json"
    import_real_world_dataset(
        package_dir / "dataset_export_manifest.json",
        real_world_root=tmp_path / "data" / "real_world",
        registry_path=registry_path,
        report_path=tmp_path / "reports" / "dataset_validation_report.json",
    )

    output = format_dataset_registry(registry_path)

    assert "dataset_version" in output
    assert "rw-1" in output
    assert "approved_for_training" in output
    assert "98.0" in output


def test_duplicate_participants_rejects_and_updates_registry(tmp_path: Path) -> None:
    package_dir = _write_export_package(tmp_path / "data" / "real_world" / "incoming", "rw-1", duplicate=True)
    registry_path = tmp_path / "dataset_registry" / "datasets.json"
    report_path = tmp_path / "reports" / "dataset_validation_report.json"

    with pytest.raises(RealWorldDatasetIngestionError):
        import_real_world_dataset(
            package_dir / "dataset_export_manifest.json",
            real_world_root=tmp_path / "data" / "real_world",
            registry_path=registry_path,
            report_path=report_path,
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["datasets"][0]["status"] == "rejected"
    assert registry["datasets"][0]["validation_status"] == "rejected"
    assert registry["datasets"][0]["training_status"] == "not_started"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["duplicate_participants"] == ["P001"]


def test_fashionapp_tester_record_photos_measurements_cm_and_garment_context_parse() -> None:
    record = {
        "session_id": "test-session-001",
        "tester_id": "tester-001",
        "photos": {
            "front": "front.jpg",
            "side": "side.jpg",
            "back": "back.jpg",
        },
        "measurements_cm": {
            "height": 170,
            "chest": 95,
            "waist": 80,
            "hip": 98,
        },
        "garment_context": {
            "garment_type": "Dress",
            "garment_type_other": None,
            "fabric_type": "Ankara",
            "fabric_type_other": None,
            "fit_preference": "Regular",
            "fit_preference_other": None,
            "occasion": "Wedding",
            "occasion_other": None,
        },
        "source": "real_tester_collection",
        "created_at": "2026-01-01T00:00:00Z",
    }

    images_by_view = _record_images_by_view(record)
    measurements = _measurement_values(record)
    metadata = _record_metadata(record)

    assert set(images_by_view) == {"front", "side", "back"}
    assert images_by_view["front"]["imageAssetId"] == "front.jpg"
    assert images_by_view["side"]["poseType"] == "side"
    assert measurements["height"] == 170
    assert measurements["chest"] == 95
    assert measurements["waist"] == 80
    assert measurements["hip"] == 98
    assert measurements["bust_chest"] == 95
    assert measurements["hips"] == 98
    assert metadata["garment_context"] == record["garment_context"]
    assert "garment_context" not in measurements
    assert all(key not in measurements for key in record["garment_context"])


def test_dict_images_and_measurements_remain_supported() -> None:
    record = {
        "images": {
            "front": "front.jpg",
            "side": "side.jpg",
            "back": "back.jpg",
        },
        "measurements": {
            "height": 170,
            "chest": 95,
            "waist": 80,
            "hip": 98,
        },
    }

    images_by_view = _record_images_by_view(record)
    measurements = _measurement_values(record)

    assert set(images_by_view) == {"front", "side", "back"}
    assert images_by_view["back"]["imageAssetId"] == "back.jpg"
    assert measurements["height"] == 170
    assert measurements["chest"] == 95
    assert measurements["waist"] == 80
    assert measurements["hip"] == 98
    assert measurements["bust_chest"] == 95
    assert measurements["hips"] == 98


def _write_export_package(incoming_root: Path, dataset_version: str, *, duplicate: bool = False) -> Path:
    package_dir = incoming_root / dataset_version
    image_dir = package_dir / "images"
    image_dir.mkdir(parents=True)
    for pose in ("front", "side", "back"):
        (image_dir / f"{pose}.jpg").write_bytes(f"{pose}-image".encode("utf-8"))

    records = [_record(dataset_version, "P001")]
    if duplicate:
        records.append(_record(dataset_version, "P001"))
        for pose in ("front", "side", "back"):
            (image_dir / f"{pose}-duplicate.jpg").write_bytes(f"{pose}-duplicate".encode("utf-8"))
        for image in records[1]["images"]:
            image["imageAssetId"] = f"images/{image['poseType']}-duplicate.jpg"

    manifest = {
        "approvedOnly": True,
        "datasetVersion": dataset_version,
        "exportId": "export_1",
        "exportTimestamp": "2026-06-12T12:00:00Z",
        "imageCount": 3 * len(records),
        "imageQualitySchemaVersion": "image-quality-v1",
        "labelsPath": "labels.json",
        "measurementCount": 7 * len(records),
        "measurementSchemaVersion": "field-measurements-v1",
        "metadataPath": "metadata.json",
        "participantCount": len({record["participantCode"] for record in records}),
        "recordsPath": "records.json",
        "schemaVersion": "real-world-dataset-export-v1",
        "sourceAppVersion": "fashionapp-field-data-beta",
        "sourceDatasetVersion": dataset_version,
        "sourceExportId": "export_1",
        "sourceSystem": "CUSTOM-FASHION-MARKETPLACE",
        "consentMetadataPath": "consent_metadata.json",
    }
    labels = {
        "labels": [
            {"participantCode": "P001", "measurements": {key: 100 for key in _measurement_keys()}},
        ]
    }
    metadata = {"source": "CUSTOM-FASHION-MARKETPLACE", "datasetVersion": dataset_version}
    consent = {"participants": [{"participantCode": "P001", "consentStatus": "granted", "consentVersion": "field-v1"}]}

    (package_dir / "dataset_export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (package_dir / "records.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (package_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (package_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (package_dir / "consent_metadata.json").write_text(json.dumps(consent), encoding="utf-8")
    return package_dir


def _record(dataset_version: str, participant_code: str) -> dict[str, object]:
    return {
        "consentStatus": "granted",
        "datasetVersion": dataset_version,
        "images": [_image("front"), _image("side"), _image("back")],
        "lineage": {"collectionSessionId": "session_1", "imageVersion": 1, "measurementVersion": 1},
        "measurements": [
            {"key": key, "reviewStatus": "approved", "valueCm": 100}
            for key in _measurement_keys()
        ],
        "participantCode": participant_code,
        "quality": {"overallDatasetReadinessScore": 98, "readinessState": "approved_for_export"},
    }


def _image(pose: str) -> dict[str, object]:
    return {
        "imageAssetId": f"images/{pose}.jpg",
        "poseType": pose,
        "qualityScore": 98,
        "reviewStatus": "approved",
    }


def _measurement_keys() -> tuple[str, ...]:
    return ("height", "weight", "bust_chest", "waist", "hips", "shoulder", "dress_length")


def _mutate_manifest(package_dir: Path, update: Callable[[dict[str, object]], None]) -> None:
    manifest_path = package_dir / "dataset_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    update(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
