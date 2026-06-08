import json
from pathlib import Path
import zlib

import numpy as np

from training.datasets.multimodal_verified_dataset import (
    ImagePreprocessor,
    ImageResolver,
    MultimodalVerifiedDataset,
    READINESS_METADATA_ONLY,
    READINESS_MULTIMODAL_READY,
)


def test_image_resolver_supports_local_paths_storage_keys_and_signed_urls(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=1)
    sample = MultimodalVerifiedDataset(dataset_root, include_tensors=False).loader[0]
    resolver = ImageResolver(dataset_root)

    local = resolver.resolve(sample, "front")
    assert local.reference_kind == "local_path"
    assert local.status == "resolved"
    assert local.resolved_path is not None
    assert local.resolved_path.exists()

    storage_root = tmp_path / "storage"
    storage_image = storage_root / "bucket" / "side.png"
    storage_image.parent.mkdir(parents=True)
    _write_png(storage_image, width=3, height=5)
    storage_sample = {
        **sample,
        "raw_record": {"imageReferences": {"side": {"storageKey": "bucket/side.png"}}},
    }
    storage = ImageResolver(dataset_root, storage_root=storage_root).resolve(storage_sample, "side")
    assert storage.reference_kind == "storage_key"
    assert storage.status == "resolved"
    assert storage.resolved_path == storage_image

    signed_sample = {
        **sample,
        "raw_record": {"imageReferences": {"back": {"signedUrl": "https://example.invalid/back.png"}}},
    }
    signed = resolver.resolve(signed_sample, "back")
    assert signed.reference_kind == "signed_url"
    assert signed.status == "signed_url_pending"
    assert signed.resolved_path is None


def test_image_validation_and_preprocessing(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=1, image_size=(8, 4))
    dataset = MultimodalVerifiedDataset(dataset_root, image_size=(6, 6))
    sample = dataset.loader[0]
    resolution = dataset.resolver.resolve(sample, "front")

    validation = dataset.preprocessor.validate(resolution)
    processed = dataset.preprocessor.preprocess(resolution)

    assert validation.valid is True
    assert validation.width == 8
    assert validation.height == 4
    assert processed.success is True
    assert isinstance(processed.tensor, np.ndarray)
    assert processed.tensor.shape == (6, 6, 3)
    assert float(processed.tensor.max()) <= 1.0
    assert processed.metadata["resizeMode"] == "pad_to_square"
    assert processed.metadata["orientationNormalized"] is True


def test_unreadable_and_missing_images_are_reported_without_training(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=1)
    (dataset_root / "images" / "sample-1-front.png").write_text("not an image", encoding="utf-8")
    (dataset_root / "images" / "sample-1-back.png").unlink()

    dataset = MultimodalVerifiedDataset(dataset_root, include_tensors=False)
    sample = dataset[0]
    report = dataset.report()

    assert sample["readinessState"] == READINESS_METADATA_ONLY
    assert sample["frontImage"]["validation"]["status"] == "unreadable_image"
    assert sample["backImage"]["validation"]["status"] == "missing_file"
    assert report["unreadableImageCounts"]["front"] == 1
    assert report["missingImageCounts"]["back"] == 1
    assert report["datasetReadiness"] == READINESS_METADATA_ONLY


def test_multimodal_dataset_preserves_view_aware_schema_and_lineage(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=2)
    dataset = MultimodalVerifiedDataset(dataset_root, image_size=(4, 4))

    sample = dataset[0]

    assert sample["readinessState"] == READINESS_MULTIMODAL_READY
    assert sample["frontImage"]["view"] == "front"
    assert sample["sideImage"]["view"] == "side"
    assert sample["backImage"]["view"] == "back"
    assert sample["frontImage"]["resolution"]["resolved_path"] != sample["sideImage"]["resolution"]["resolved_path"]
    assert sample["frontImage"]["tensor"].shape == (4, 4, 3)
    assert sample["poseMetadata"]["front"]["pose_confidence"] == 0.91
    assert sample["validationMetadata"]["front"]["quality_score"] == 0.92
    assert sample["verificationMetadata"]["verified"] is True
    assert sample["finalApprovedMeasurements"]["chest_cm"] == 96.0
    assert set(sample["lineage"]) == {"ai_estimate", "customer_edit", "maker_adjustment", "final_approved"}
    assert sample["lineage"]["customer_edit"]["waist_cm"] == 79.5


def test_image_dataset_report_and_write_report(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=2)
    dataset = MultimodalVerifiedDataset(dataset_root, include_tensors=False)

    report = dataset.report()
    outputs = dataset.write_report(tmp_path / "report")
    written = json.loads(Path(outputs["image_dataset_report"]).read_text(encoding="utf-8"))

    assert report["recordCount"] == 2
    assert report["imageCoverage"] == {"resolvedViews": 6, "totalViews": 6, "coverage": 1.0}
    assert report["preprocessingSuccessRate"] == 1.0
    assert report["brokenReferenceCount"] == 0
    assert report["readinessCounts"][READINESS_MULTIMODAL_READY] == 2
    assert report["datasetReadiness"] == READINESS_MULTIMODAL_READY
    assert written["datasetReadiness"] == READINESS_MULTIMODAL_READY


def test_metadata_gaps_keep_image_ready_short_of_multimodal_ready(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=1)
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["records"][0]["verification_metadata_summary"] = {}
    (dataset_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = MultimodalVerifiedDataset(dataset_root, include_tensors=False)
    sample = dataset[0]
    report = dataset.report()

    assert sample["readinessState"] == "image_ready"
    assert report["datasetReadiness"] == "image_ready"


def _write_verified_dataset(tmp_path: Path, *, count: int, image_size: tuple[int, int] = (4, 4)) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True)
    records = []
    for index in range(1, count + 1):
        sample_id = f"sample-{index}"
        for view in ("front", "side", "back"):
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png", width=image_size[0], height=image_size[1])
        records.append(
            {
                "sample_id": sample_id,
                "dataset_version": "v1",
                "front_image_reference": f"images/{sample_id}-front.png",
                "side_image_reference": f"images/{sample_id}-side.png",
                "back_image_reference": f"images/{sample_id}-back.png",
                "pose_metadata_summary": {
                    "front": {"pose_confidence": 0.91},
                    "side": {"pose_confidence": 0.89},
                    "back": {"pose_confidence": 0.87},
                },
                "validation_metadata_summary": {
                    "front": {"quality_score": 0.92},
                    "side": {"quality_score": 0.90},
                    "back": {"quality_score": 0.88},
                },
                "verification_metadata_summary": {"verified": True, "maker_review_score": 0.95},
                "lineage": {
                    "ai_estimate": {"chest_cm": 94.0, "waist_cm": 78.0},
                    "customer_edit": {"chest_cm": 95.0, "waist_cm": 79.5},
                    "maker_adjustment": {"chest_cm": 96.0, "waist_cm": 80.0},
                    "final_approved": {"chest_cm": 96.0, "waist_cm": 80.0},
                },
                "correction_deltas": {"chest_cm": 2.0, "waist_cm": 2.0},
                "confidence_metadata": {"confidence_tier": "high_confidence"},
                "eligibility_metadata": {"eligible_for_training": True},
            }
        )
    (dataset_root / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return dataset_root


def _write_png(path: Path, *, width: int = 4, height: int = 4) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")

    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    raw_scanline = b"".join(b"\x00" + (b"\xff\x00\x00\xff" * width) for _row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )
