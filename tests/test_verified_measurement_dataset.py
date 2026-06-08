import json
from pathlib import Path
import zlib

import pytest

from training.datasets.verified_measurement_dataset import (
    VerifiedMeasurementDatasetError,
    VerifiedMeasurementDatasetLoader,
    format_quality_report,
)


def test_loads_verified_measurement_manifest_and_preserves_lineage(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path)

    loader = VerifiedMeasurementDatasetLoader(dataset_root)
    sample = loader[0]

    assert len(loader) == 2
    assert sample["dataset_version"] == "v1"
    assert sample["front_image_path"].exists()
    assert sample["side_image_path"].exists()
    assert sample["back_image_path"].exists()
    assert sample["lineage"]["ai_estimate"]["chest_cm"] == 94.0
    assert sample["lineage"]["customer_edit"]["waist_cm"] == 80.0
    assert sample["lineage"]["maker_adjustment"]["chest_cm"] == 96.0
    assert sample["lineage"]["final_approved"]["chest_cm"] == 96.5
    assert sample["final_approved_measurements"] == sample["lineage"]["final_approved"]


def test_supports_camel_case_jsonl_v2_exports(tmp_path: Path) -> None:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True)
    for view in ("front", "side", "back"):
        _write_png(dataset_root / "images" / f"sample-a-{view}.png")

    record = {
        "sampleId": "sample-a",
        "datasetVersion": "v2",
        "imageReferences": {
            "front": {"path": "images/sample-a-front.png"},
            "side": {"path": "images/sample-a-side.png"},
            "back": {"path": "images/sample-a-back.png"},
        },
        "poseMetadataSummary": {"front": {"poseConfidence": 0.91}},
        "validationMetadataSummary": {"front": {"qualityScore": 0.89}},
        "verificationMetadataSummary": {"verifiedBy": "maker_1"},
        "measurementLineage": {
            "aiEstimate": {"chest_cm": 94.0},
            "customerEdit": {"chest_cm": 95.0},
            "makerAdjustment": {"chest_cm": 96.0},
            "finalApprovedMeasurements": {"chest_cm": {"value_cm": 96.0, "confidenceTier": "high_confidence"}},
        },
        "correctionDeltas": {"chest_cm": 2.0},
        "confidenceMetadata": {"overall": {"confidenceTier": "high_confidence"}},
        "eligibilityMetadata": {"eligibleForTraining": True},
    }
    (dataset_root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    loader = VerifiedMeasurementDatasetLoader(dataset_root)

    assert loader[0]["sample_id"] == "sample-a"
    assert loader[0]["lineage"]["final_approved"]["chest_cm"]["value_cm"] == 96.0
    assert loader.statistics()["dataset_versions"] == {"v2": 1}
    assert loader.statistics()["confidence_distribution"]["high_confidence"] == 2


def test_validation_requires_three_views_final_approved_and_lineage(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path)
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["records"][0]["back_image_reference"] = "images/missing-back.png"
    payload["records"][0]["lineage"].pop("final_approved")
    (dataset_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VerifiedMeasurementDatasetError, match="missing back_image"):
        VerifiedMeasurementDatasetLoader(dataset_root)

    loader = VerifiedMeasurementDatasetLoader(dataset_root, validate=False)

    assert loader.validation["valid"] is False
    assert loader.validation["missing_field_counts"]["back_image"] == 1
    assert loader.validation["missing_field_counts"]["final_approved_measurements"] == 1


def test_statistics_and_quality_report_capture_coverage_confidence_and_corrections(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path)
    loader = VerifiedMeasurementDatasetLoader(dataset_root)

    stats = loader.statistics()
    report = loader.quality_report()
    markdown = format_quality_report(report)

    assert stats["record_count"] == 2
    assert stats["measurement_coverage"]["chest_cm"] == {"count": 2, "coverage": 1.0}
    assert stats["measurement_coverage"]["waist_cm"] == {"count": 1, "coverage": 0.5}
    assert stats["confidence_distribution"] == {"high_confidence": 1, "medium_confidence": 1}
    assert stats["correction_distribution"]["chest_cm"]["mean_abs_delta"] == 2.0
    assert stats["missing_field_counts"] == {}
    assert report["report_design"]["training_policy"].startswith("Ingestion does not retrain")
    assert "Verified Measurement Dataset Quality Report" in markdown
    assert "Training Boundary" in markdown


def test_write_quality_report_outputs_json_and_markdown(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path)
    loader = VerifiedMeasurementDatasetLoader(dataset_root)

    outputs = loader.write_quality_report(tmp_path / "report")

    assert Path(outputs["json"]).exists()
    assert Path(outputs["markdown"]).exists()
    assert json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))["valid"] is True


def test_unsupported_dataset_version_fails_clearly(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path)
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["dataset_version"] = "beta"
    (dataset_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VerifiedMeasurementDatasetError, match="unsupported dataset_version"):
        VerifiedMeasurementDatasetLoader(dataset_root)


def _write_verified_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True)
    records = []
    for index in range(1, 3):
        sample_id = f"sample-{index}"
        for view in ("front", "side", "back"):
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png")
        records.append(
            {
                "sample_id": sample_id,
                "front_image_reference": f"images/{sample_id}-front.png",
                "side_image_reference": f"images/{sample_id}-side.png",
                "back_image_reference": f"images/{sample_id}-back.png",
                "pose_metadata_summary": {"front": {"pose_confidence": 0.95}},
                "validation_metadata_summary": {"front": {"quality_score": 0.93}},
                "verification_metadata_summary": {"status": "verified"},
                "lineage": {
                    "ai_estimate": {"chest_cm": 94.0, "waist_cm": 79.0},
                    "customer_edit": {"waist_cm": 80.0},
                    "maker_adjustment": {"chest_cm": 96.0},
                    "final_approved": {"chest_cm": 96.5, **({"waist_cm": 80.0} if index == 1 else {})},
                },
                "correction_deltas": {"chest_cm": 2.5 if index == 1 else 1.5},
                "confidence_metadata": {"confidence_tier": "high_confidence" if index == 1 else "medium_confidence"},
                "eligibility_metadata": {"eligible_for_training": True, "holdout_candidate": index == 2},
            }
        )
    (dataset_root / "manifest.json").write_text(
        json.dumps({"dataset_version": "v1", "records": records}, indent=2),
        encoding="utf-8",
    )
    return dataset_root


def _write_png(path: Path) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")

    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    raw_scanline = b"\x00\x00\x00\x00\xff"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )
