from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from training.datasets.verified_measurement_dataset import REQUIRED_IMAGE_VIEWS, VerifiedMeasurementDatasetLoader


IMAGE_DATASET_REPORT_FILENAME = "image_dataset_report.json"
READINESS_METADATA_ONLY = "metadata_only"
READINESS_IMAGE_READY = "image_ready"
READINESS_MULTIMODAL_READY = "multimodal_ready"
LOCAL_REFERENCE_KEYS = ("path", "uri", "url", "reference", "image_reference", "imageReference")
STORAGE_KEY_KEYS = ("storage_key", "storageKey", "key")
SIGNED_URL_KEYS = ("signed_url", "signedUrl", "downloadUrl", "download_url")


class MultimodalDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class ImageResolution:
    view: str
    reference_kind: str
    original_reference: str | None
    resolved_path: Path | None
    status: str
    warnings: list[str]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_path"] = str(self.resolved_path) if self.resolved_path is not None else None
        return payload


@dataclass(frozen=True)
class ImageValidation:
    view: str
    valid: bool
    status: str
    path: Path | None
    error: str | None
    width: int | None
    height: int | None
    mode: str | None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path) if self.path is not None else None
        return payload


@dataclass(frozen=True)
class ImagePreprocessingResult:
    view: str
    success: bool
    tensor: np.ndarray | None
    metadata: dict[str, Any]
    error: str | None

    def payload_without_tensor(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "success": self.success,
            "metadata": self.metadata,
            "error": self.error,
        }


class ImageResolver:
    def __init__(self, dataset_root: str | Path, storage_root: str | Path | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.storage_root = Path(storage_root) if storage_root is not None else None

    def resolve(self, sample: dict[str, Any], view: str) -> ImageResolution:
        raw_reference = image_reference_from_raw(sample.get("raw_record", {}), view)
        fallback_path = sample.get(f"{view}_image_path")

        if isinstance(raw_reference, dict):
            return self._resolve_structured_reference(raw_reference, fallback_path, view)
        if raw_reference not in ("", None):
            return self._resolve_local_reference(str(raw_reference), view, fallback_path)
        if isinstance(fallback_path, Path):
            return ImageResolution(view, "local_path", str(fallback_path), fallback_path, "resolved", [])
        return ImageResolution(view, "missing_reference", None, None, "missing_reference", [f"{view} image reference is missing."])

    def _resolve_structured_reference(self, reference: dict[str, Any], fallback_path: Any, view: str) -> ImageResolution:
        storage_key = first_present(reference, *STORAGE_KEY_KEYS)
        if storage_key:
            if self.storage_root is None:
                return ImageResolution(
                    view,
                    "storage_key",
                    str(storage_key),
                    None,
                    "storage_key_unresolved",
                    ["Storage key resolution requires a storage_root; no download or remote lookup was attempted."],
                )
            return self._resolve_path(self.storage_root / str(storage_key), view, "storage_key", str(storage_key))

        signed_url = first_present(reference, *SIGNED_URL_KEYS)
        if signed_url:
            return ImageResolution(
                view,
                "signed_url",
                str(signed_url),
                None,
                "signed_url_pending",
                ["Signed URL download is reserved for a future storage integration; no network fetch was attempted."],
            )

        local_reference = first_present(reference, *LOCAL_REFERENCE_KEYS)
        if local_reference:
            return self._resolve_local_reference(str(local_reference), view, fallback_path)
        return ImageResolution(view, "missing_reference", None, None, "missing_reference", [f"{view} image reference is missing."])

    def _resolve_local_reference(self, reference: str, view: str, fallback_path: Any = None) -> ImageResolution:
        if reference.startswith(("http://", "https://")):
            return ImageResolution(
                view,
                "signed_url",
                reference,
                None,
                "signed_url_pending",
                ["HTTP image references are recorded for future signed URL resolution; no network fetch was attempted."],
            )
        path = Path(reference)
        if not path.is_absolute():
            path = self.dataset_root / path
        if not path.exists() and isinstance(fallback_path, Path):
            path = fallback_path
        return self._resolve_path(path, view, "local_path", reference)

    @staticmethod
    def _resolve_path(path: Path, view: str, reference_kind: str, original_reference: str) -> ImageResolution:
        status = "resolved" if path.exists() else "missing_file"
        warnings = [] if path.exists() else [f"{view} image file does not exist: {path}"]
        return ImageResolution(view, reference_kind, original_reference, path, status, warnings)


class ImagePreprocessor:
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        *,
        normalize: bool = True,
        preserve_aspect_ratio: bool = True,
        fill_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.image_size = image_size
        self.normalize = normalize
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.fill_color = fill_color

    def validate(self, resolution: ImageResolution) -> ImageValidation:
        if resolution.resolved_path is None:
            return ImageValidation(resolution.view, False, resolution.status, None, "; ".join(resolution.warnings), None, None, None)
        if not resolution.resolved_path.exists():
            return ImageValidation(
                resolution.view,
                False,
                "missing_file",
                resolution.resolved_path,
                f"Image file does not exist: {resolution.resolved_path}",
                None,
                None,
                None,
            )
        if not resolution.resolved_path.is_file():
            return ImageValidation(
                resolution.view,
                False,
                "broken_reference",
                resolution.resolved_path,
                f"Image reference is not a file: {resolution.resolved_path}",
                None,
                None,
                None,
            )
        try:
            with Image.open(resolution.resolved_path) as image:
                image.load()
                return ImageValidation(
                    resolution.view,
                    True,
                    "valid",
                    resolution.resolved_path,
                    None,
                    int(image.width),
                    int(image.height),
                    str(image.mode),
                )
        except (OSError, UnidentifiedImageError) as error:
            return ImageValidation(
                resolution.view,
                False,
                "unreadable_image",
                resolution.resolved_path,
                str(error),
                None,
                None,
                None,
            )

    def preprocess(self, resolution: ImageResolution) -> ImagePreprocessingResult:
        validation = self.validate(resolution)
        if not validation.valid or validation.path is None:
            return ImagePreprocessingResult(
                resolution.view,
                False,
                None,
                {
                    "validation": validation.to_payload(),
                    "targetSize": list(self.image_size),
                    "normalized": self.normalize,
                    "preserveAspectRatio": self.preserve_aspect_ratio,
                    "orientationNormalized": False,
                },
                validation.error,
            )
        try:
            with Image.open(validation.path) as image:
                oriented = ImageOps.exif_transpose(image)
                rgb = oriented.convert("RGB")
                if self.preserve_aspect_ratio:
                    processed = ImageOps.pad(rgb, self.image_size, method=Image.Resampling.BILINEAR, color=self.fill_color)
                    resize_mode = "pad_to_square"
                else:
                    processed = rgb.resize(self.image_size, Image.Resampling.BILINEAR)
                    resize_mode = "direct_resize"
                array = np.asarray(processed, dtype=np.float32)
                if self.normalize:
                    array = array / 255.0
                metadata = {
                    "validation": validation.to_payload(),
                    "originalSize": [validation.width, validation.height],
                    "processedSize": [processed.width, processed.height],
                    "sourceMode": validation.mode,
                    "processedMode": processed.mode,
                    "resizeMode": resize_mode,
                    "normalized": self.normalize,
                    "normalizationRange": [0.0, 1.0] if self.normalize else [0.0, 255.0],
                    "preserveAspectRatio": self.preserve_aspect_ratio,
                    "orientationNormalized": True,
                }
                return ImagePreprocessingResult(resolution.view, True, array, metadata, None)
        except (OSError, UnidentifiedImageError) as error:
            return ImagePreprocessingResult(
                resolution.view,
                False,
                None,
                {
                    "validation": validation.to_payload(),
                    "targetSize": list(self.image_size),
                    "normalized": self.normalize,
                    "preserveAspectRatio": self.preserve_aspect_ratio,
                    "orientationNormalized": False,
                },
                str(error),
            )


class MultimodalVerifiedDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        records_file: str | Path | None = None,
        *,
        storage_root: str | Path | None = None,
        image_size: tuple[int, int] = (224, 224),
        include_tensors: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.loader = VerifiedMeasurementDatasetLoader(dataset_root, records_file, validate=False)
        self.resolver = ImageResolver(dataset_root, storage_root)
        self.preprocessor = ImagePreprocessor(image_size)
        self.include_tensors = include_tensors
        self.samples = list(self.loader)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        images = self._build_images(sample)
        return {
            "sampleId": sample["sample_id"],
            "datasetVersion": sample["dataset_version"],
            "profileType": sample["profile_type"],
            "frontImage": images["front"],
            "sideImage": images["side"],
            "backImage": images["back"],
            "poseMetadata": sample["pose_metadata_summary"],
            "validationMetadata": sample["validation_metadata_summary"],
            "verificationMetadata": sample["verification_metadata_summary"],
            "confidenceMetadata": sample["confidence_metadata"],
            "eligibilityMetadata": sample["eligibility_metadata"],
            "finalApprovedMeasurements": sample["final_approved_measurements"],
            "targetAvailability": sample["target_availability"],
            "lineage": sample["lineage"],
            "correctionDeltas": sample["correction_deltas"],
            "rawRecord": sample["raw_record"],
            "readinessState": readiness_state_for_sample(images, sample),
        }

    def report(self) -> dict[str, Any]:
        records = [self[index] for index in range(len(self))]
        total_views = len(records) * len(REQUIRED_IMAGE_VIEWS)
        resolved_count = 0
        missing_counts = {view: 0 for view in REQUIRED_IMAGE_VIEWS}
        unreadable_counts = {view: 0 for view in REQUIRED_IMAGE_VIEWS}
        broken_reference_count = 0
        preprocessing_success = 0
        view_distribution = {view: {"records": len(records), "valid": 0, "preprocessed": 0} for view in REQUIRED_IMAGE_VIEWS}
        readiness_counts = {READINESS_METADATA_ONLY: 0, READINESS_IMAGE_READY: 0, READINESS_MULTIMODAL_READY: 0}

        for record in records:
            readiness_counts[record["readinessState"]] += 1
            for view in REQUIRED_IMAGE_VIEWS:
                image = record[f"{view}Image"]
                resolution_status = image["resolution"]["status"]
                validation_status = image["validation"]["status"]
                if resolution_status == "resolved":
                    resolved_count += 1
                if validation_status == "valid":
                    view_distribution[view]["valid"] += 1
                if image["preprocessing"]["success"]:
                    preprocessing_success += 1
                    view_distribution[view]["preprocessed"] += 1
                if validation_status == "missing_file":
                    missing_counts[view] += 1
                if validation_status == "unreadable_image":
                    unreadable_counts[view] += 1
                if resolution_status in {"missing_reference", "storage_key_unresolved", "signed_url_pending"} or validation_status == "broken_reference":
                    broken_reference_count += 1

        successful_records = readiness_counts[READINESS_MULTIMODAL_READY]
        dataset_readiness = dataset_readiness_state(readiness_counts, len(records))
        return {
            "schemaVersion": "image_dataset_report_v1",
            "dataset": str(self.dataset_root),
            "recordsFile": str(self.loader.records_path),
            "recordCount": len(records),
            "imageCoverage": {
                "resolvedViews": resolved_count,
                "totalViews": total_views,
                "coverage": round(resolved_count / total_views, 4) if total_views else 0.0,
            },
            "missingImageCounts": missing_counts,
            "unreadableImageCounts": unreadable_counts,
            "viewDistribution": view_distribution,
            "preprocessingSuccessRate": round(preprocessing_success / total_views, 4) if total_views else 0.0,
            "brokenReferenceCount": broken_reference_count,
            "readinessCounts": readiness_counts,
            "datasetReadiness": dataset_readiness,
            "multimodalReadyRecords": successful_records,
            "limitations": [
                "Signed URLs are recognized but not downloaded in Phase H.4.",
                "Storage keys require an explicit local storage_root mapping for offline resolution.",
                "This module prepares tensors and metadata only; it does not train or promote a model.",
            ],
        }

    def write_report(self, output_dir: str | Path) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / IMAGE_DATASET_REPORT_FILENAME
        write_json(report_path, self.report())
        return {"image_dataset_report": str(report_path)}

    def _build_images(self, sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for view in REQUIRED_IMAGE_VIEWS:
            resolution = self.resolver.resolve(sample, view)
            validation = self.preprocessor.validate(resolution)
            preprocessing = self.preprocessor.preprocess(resolution)
            image_payload = {
                "view": view,
                "resolution": resolution.to_payload(),
                "validation": validation.to_payload(),
                "preprocessing": preprocessing.payload_without_tensor(),
            }
            if self.include_tensors:
                image_payload["tensor"] = preprocessing.tensor
            output[view] = image_payload
        return output


def readiness_state_for_sample(images: dict[str, dict[str, Any]], sample: dict[str, Any]) -> str:
    images_ready = all(images[view]["preprocessing"]["success"] for view in REQUIRED_IMAGE_VIEWS)
    if not images_ready:
        return READINESS_METADATA_ONLY
    metadata_ready = bool(sample["pose_metadata_summary"] and sample["validation_metadata_summary"] and sample["verification_metadata_summary"])
    lineage_ready = all(sample["lineage"].get(key) for key in ("ai_estimate", "customer_edit", "maker_adjustment", "final_approved"))
    if metadata_ready and lineage_ready and sample["final_approved_measurements"]:
        return READINESS_MULTIMODAL_READY
    return READINESS_IMAGE_READY


def dataset_readiness_state(readiness_counts: dict[str, int], record_count: int) -> str:
    if record_count and readiness_counts[READINESS_MULTIMODAL_READY] == record_count:
        return READINESS_MULTIMODAL_READY
    if record_count and readiness_counts[READINESS_IMAGE_READY] + readiness_counts[READINESS_MULTIMODAL_READY] == record_count:
        return READINESS_IMAGE_READY
    return READINESS_METADATA_ONLY


def image_reference_from_raw(raw_record: dict[str, Any], view: str) -> Any:
    for key in (
        f"{view}_image_reference",
        f"{view}ImageReference",
        f"{view}_image_path",
        f"{view}ImagePath",
        f"{view}_storage_key",
        f"{view}StorageKey",
        f"{view}_signed_url",
        f"{view}SignedUrl",
    ):
        value = raw_record.get(key)
        if value not in ("", None):
            if key.endswith("StorageKey") or key.endswith("_storage_key"):
                return {"storageKey": value}
            if key.endswith("SignedUrl") or key.endswith("_signed_url"):
                return {"signedUrl": value}
            return value

    image_references = first_present(raw_record, "image_references", "imageReferences", "images", "views")
    if isinstance(image_references, dict):
        value = image_references.get(view)
        if value not in ("", None):
            return value
    return None


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in ("", None):
            return value
    return None


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True, default=default)
        output_file.write("\n")


def format_cli_summary(report: dict[str, Any], outputs: dict[str, str] | None = None) -> str:
    lines = [
        f"Dataset readiness: {report['datasetReadiness']}",
        f"Records: {report['recordCount']}",
        f"Image coverage: {report['imageCoverage']['resolvedViews']}/{report['imageCoverage']['totalViews']}",
        f"Preprocessing success rate: {report['preprocessingSuccessRate']:.4f}",
        f"Broken references: {report['brokenReferenceCount']}",
    ]
    if outputs:
        lines.append(f"Image dataset report: {outputs['image_dataset_report']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and report a verified multimodal image+metadata dataset foundation.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--storage-root", help="Optional local root for resolving storage keys.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size for preprocessing.")
    parser.add_argument("--output", help="Optional output directory for image_dataset_report.json.")
    args = parser.parse_args(argv)

    dataset = MultimodalVerifiedDataset(
        args.dataset,
        args.records_file,
        storage_root=args.storage_root,
        image_size=(args.image_size, args.image_size),
        include_tensors=False,
    )
    report = dataset.report()
    outputs = dataset.write_report(args.output) if args.output else None
    print(format_cli_summary(report, outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
