from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterator


DEFAULT_RECORD_FILES = (
    "records.jsonl",
    "verified_measurements.jsonl",
    "records.json",
    "verified_measurements.json",
    "manifest.json",
)
LINEAGE_KEYS = ("ai_estimate", "customer_edit", "maker_adjustment", "final_approved")
REQUIRED_IMAGE_VIEWS = ("front", "side", "back")
SUPPORTED_VERSION_PATTERN = re.compile(r"^v[1-9][0-9]*$")


class VerifiedMeasurementDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedMeasurementRecord:
    sample_id: str
    dataset_version: str
    front_image_path: Path
    side_image_path: Path
    back_image_path: Path
    pose_metadata_summary: dict[str, Any]
    validation_metadata_summary: dict[str, Any]
    verification_metadata_summary: dict[str, Any]
    lineage: dict[str, dict[str, Any]]
    correction_deltas: dict[str, Any]
    confidence_metadata: dict[str, Any]
    eligibility_metadata: dict[str, Any]
    raw_record: dict[str, Any]

    @property
    def final_approved_measurements(self) -> dict[str, Any]:
        return self.lineage["final_approved"]

    @property
    def ai_measurements(self) -> dict[str, Any]:
        return self.lineage["ai_estimate"]

    @property
    def customer_edits(self) -> dict[str, Any]:
        return self.lineage["customer_edit"]

    @property
    def maker_adjustments(self) -> dict[str, Any]:
        return self.lineage["maker_adjustment"]


class VerifiedMeasurementDatasetLoader:
    """Load verified FashionApp measurement exports without training or model mutation."""

    def __init__(
        self,
        dataset_root: str | Path,
        records_file: str | Path | None = None,
        *,
        validate: bool = True,
        load_images: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.load_images = load_images
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.records_path = self._resolve_records_path(records_file)
        self.dataset_metadata, raw_records = self._load_raw_records(self.records_path)
        self.records = [self._normalize_record(record, index) for index, record in enumerate(raw_records, start=1)]
        self.validation = self.validate_records()
        if validate and not self.validation["valid"]:
            raise VerifiedMeasurementDatasetError("; ".join(self.validation["errors"]))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = {
            "sample_id": record.sample_id,
            "dataset_version": record.dataset_version,
            "front_image_path": record.front_image_path,
            "side_image_path": record.side_image_path,
            "back_image_path": record.back_image_path,
            "pose_metadata_summary": record.pose_metadata_summary,
            "validation_metadata_summary": record.validation_metadata_summary,
            "verification_metadata_summary": record.verification_metadata_summary,
            "lineage": record.lineage,
            "correction_deltas": record.correction_deltas,
            "confidence_metadata": record.confidence_metadata,
            "eligibility_metadata": record.eligibility_metadata,
            "final_approved_measurements": record.final_approved_measurements,
            "raw_record": record.raw_record,
        }
        if self.load_images:
            sample["front_image_bytes"] = record.front_image_path.read_bytes()
            sample["side_image_bytes"] = record.side_image_path.read_bytes()
            sample["back_image_bytes"] = record.back_image_path.read_bytes()
        return sample

    def validate_records(self) -> dict[str, Any]:
        missing_field_counts: dict[str, int] = {}
        errors: list[str] = []
        invalid_records: list[dict[str, Any]] = []

        for record in self.records:
            record_errors: list[str] = []
            for view in REQUIRED_IMAGE_VIEWS:
                path = getattr(record, f"{view}_image_path")
                if not path.exists():
                    field = f"{view}_image"
                    _increment(missing_field_counts, field)
                    record_errors.append(f"missing {field}: {path}")

            if not record.final_approved_measurements:
                _increment(missing_field_counts, "final_approved_measurements")
                record_errors.append("missing final approved measurements")

            for lineage_key in LINEAGE_KEYS:
                if not record.lineage.get(lineage_key) or not isinstance(record.lineage[lineage_key], dict):
                    _increment(missing_field_counts, f"lineage.{lineage_key}")
                    record_errors.append(f"missing lineage.{lineage_key}")

            for field_name in (
                "pose_metadata_summary",
                "validation_metadata_summary",
                "verification_metadata_summary",
                "correction_deltas",
                "confidence_metadata",
                "eligibility_metadata",
            ):
                if not getattr(record, field_name):
                    _increment(missing_field_counts, field_name)

            if record_errors:
                invalid_records.append({"sample_id": record.sample_id, "errors": record_errors})
                errors.extend(f"{record.sample_id}: {error}" for error in record_errors)

        return {
            "valid": not errors and len(self.records) > 0,
            "record_count": len(self.records),
            "errors": errors,
            "invalid_records": invalid_records,
            "missing_field_counts": missing_field_counts,
        }

    def statistics(self) -> dict[str, Any]:
        measurement_counts: dict[str, int] = {}
        confidence_distribution: dict[str, int] = {}
        correction_values: dict[str, list[float]] = {}
        versions: dict[str, int] = {}

        for record in self.records:
            _increment(versions, record.dataset_version)
            for measurement, value in record.final_approved_measurements.items():
                if _is_present(value):
                    _increment(measurement_counts, measurement)
            for tier in _confidence_tokens(record.confidence_metadata, record.final_approved_measurements):
                _increment(confidence_distribution, tier)
            for target, value in _flatten_numeric_values(record.correction_deltas).items():
                correction_values.setdefault(target, []).append(value)

        record_count = len(self.records)
        return {
            "record_count": record_count,
            "dataset_versions": versions,
            "measurement_coverage": {
                target: {
                    "count": count,
                    "coverage": round(count / record_count, 4) if record_count else 0.0,
                }
                for target, count in sorted(measurement_counts.items())
            },
            "confidence_distribution": dict(sorted(confidence_distribution.items())),
            "correction_distribution": _summarize_corrections(correction_values),
            "missing_field_counts": dict(sorted(self.validation["missing_field_counts"].items())),
        }

    def quality_report(self) -> dict[str, Any]:
        stats = self.statistics()
        return {
            "dataset": str(self.dataset_root),
            "records_file": str(self.records_path),
            "valid": self.validation["valid"],
            "validation": self.validation,
            "statistics": stats,
            "report_design": {
                "purpose": "Pre-training audit for verified real-world measurement exports.",
                "lineage_policy": "AI estimate, customer edit, maker adjustment, and final approved values remain separate.",
                "training_policy": "Ingestion does not retrain, promote, or overwrite production model artifacts.",
            },
        }

    def write_quality_report(self, output_dir: str | Path) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report = self.quality_report()
        json_path = output_path / "verified_measurement_dataset_quality_report.json"
        md_path = output_path / "verified_measurement_dataset_quality_report.md"
        _write_json(json_path, report)
        md_path.write_text(format_quality_report(report), encoding="utf-8")
        return {"json": str(json_path), "markdown": str(md_path)}

    def _resolve_records_path(self, records_file: str | Path | None) -> Path:
        if records_file is not None:
            path = Path(records_file)
            return path if path.is_absolute() else (self.dataset_root / path)
        for filename in DEFAULT_RECORD_FILES:
            candidate = self.dataset_root / filename
            if candidate.exists():
                return candidate
        expected = ", ".join(DEFAULT_RECORD_FILES)
        raise FileNotFoundError(f"Missing verified dataset records file in {self.dataset_root}; expected one of: {expected}")

    def _load_raw_records(self, records_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if records_path.suffix == ".jsonl":
            records = []
            with records_path.open("r", encoding="utf-8") as record_file:
                for line_number, line in enumerate(record_file, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as error:
                        raise VerifiedMeasurementDatasetError(f"{records_path}:{line_number} is not valid JSON: {error}") from error
                    if not isinstance(payload, dict):
                        raise VerifiedMeasurementDatasetError(f"{records_path}:{line_number} must contain a JSON object.")
                    records.append(payload)
            return {}, records

        with records_path.open("r", encoding="utf-8") as record_file:
            payload = json.load(record_file)

        if isinstance(payload, list):
            return {}, payload
        if not isinstance(payload, dict):
            raise VerifiedMeasurementDatasetError(f"{records_path} must contain a JSON object, array, or JSONL records.")

        records = _first_present(payload, "records", "samples", "items")
        if not isinstance(records, list):
            raise VerifiedMeasurementDatasetError(f"{records_path} must include a records, samples, or items array.")
        metadata = _coerce_dict(_first_present(payload, "metadata", "dataset_metadata", "datasetMetadata"))
        for version_key in ("dataset_version", "datasetVersion", "version"):
            if version_key in payload and version_key not in metadata:
                metadata[version_key] = payload[version_key]
        return metadata, records

    def _normalize_record(self, raw_record: dict[str, Any], row_number: int) -> VerifiedMeasurementRecord:
        if not isinstance(raw_record, dict):
            raise VerifiedMeasurementDatasetError(f"Record {row_number} must be a JSON object.")

        dataset_version = str(
            _first_present(raw_record, "dataset_version", "datasetVersion", "version")
            or _first_present(self.dataset_metadata, "dataset_version", "datasetVersion", "version")
            or ""
        )
        if not SUPPORTED_VERSION_PATTERN.match(dataset_version):
            raise VerifiedMeasurementDatasetError(
                f"Record {row_number} has unsupported dataset_version '{dataset_version}'. Expected v1, v2, v3, ..."
            )

        sample_id = str(_first_present(raw_record, "sample_id", "sampleId", "id") or f"record_{row_number:06d}")
        image_refs = _coerce_dict(_first_present(raw_record, "image_references", "imageReferences", "images", "views"))
        front_ref = _image_reference(raw_record, image_refs, "front")
        side_ref = _image_reference(raw_record, image_refs, "side")
        back_ref = _image_reference(raw_record, image_refs, "back")

        lineage = _normalize_lineage(raw_record)
        return VerifiedMeasurementRecord(
            sample_id=sample_id,
            dataset_version=dataset_version,
            front_image_path=self._resolve_dataset_path(front_ref),
            side_image_path=self._resolve_dataset_path(side_ref),
            back_image_path=self._resolve_dataset_path(back_ref),
            pose_metadata_summary=_coerce_dict(_first_present(raw_record, "pose_metadata_summary", "poseMetadataSummary", "pose_metadata", "poseMetadata")),
            validation_metadata_summary=_coerce_dict(
                _first_present(raw_record, "validation_metadata_summary", "validationMetadataSummary", "validation_metadata", "validationMetadata")
            ),
            verification_metadata_summary=_coerce_dict(
                _first_present(raw_record, "verification_metadata_summary", "verificationMetadataSummary", "verification_metadata", "verificationMetadata")
            ),
            lineage=lineage,
            correction_deltas=_coerce_dict(_first_present(raw_record, "correction_deltas", "correctionDeltas")),
            confidence_metadata=_coerce_dict(_first_present(raw_record, "confidence_metadata", "confidenceMetadata")),
            eligibility_metadata=_coerce_dict(_first_present(raw_record, "eligibility_metadata", "eligibilityMetadata")),
            raw_record=raw_record,
        )

    def _resolve_dataset_path(self, value: Any) -> Path:
        if value in ("", None):
            return self.dataset_root / "__missing__"
        path = Path(str(value))
        return path if path.is_absolute() else (self.dataset_root / path)


def _normalize_lineage(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lineage = _coerce_dict(_first_present(record, "lineage", "measurement_lineage", "measurementLineage"))
    return {
        "ai_estimate": _coerce_dict(
            _first_present(lineage, "ai_estimate", "aiEstimate", "ai_measurements", "aiMeasurements")
            or _first_present(record, "ai_measurements", "aiMeasurements", "ai_estimate", "aiEstimate")
        ),
        "customer_edit": _coerce_dict(
            _first_present(lineage, "customer_edit", "customerEdit", "customer_edits", "customerEdits")
            or _first_present(record, "customer_edits", "customerEdits", "customer_edit", "customerEdit")
        ),
        "maker_adjustment": _coerce_dict(
            _first_present(lineage, "maker_adjustment", "makerAdjustment", "maker_adjustments", "makerAdjustments")
            or _first_present(record, "maker_adjustments", "makerAdjustments", "maker_adjustment", "makerAdjustment")
        ),
        "final_approved": _coerce_dict(
            _first_present(lineage, "final_approved", "finalApproved", "final_approved_measurements", "finalApprovedMeasurements")
            or _first_present(record, "final_approved_measurements", "finalApprovedMeasurements")
        ),
    }


def _image_reference(record: dict[str, Any], image_refs: dict[str, Any], view: str) -> Any:
    direct = _first_present(record, f"{view}_image_reference", f"{view}ImageReference", f"{view}_image_path", f"{view}ImagePath")
    if direct not in ("", None):
        return _reference_value(direct)
    view_ref = image_refs.get(view)
    return _reference_value(view_ref)


def _reference_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _first_present(value, "path", "uri", "url", "reference", "image_reference", "imageReference")
    return value


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in ("", None):
            return value
    return None


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_present(value: Any) -> bool:
    return value not in ("", None, {})


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _confidence_tokens(*payloads: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for payload in payloads:
        _collect_confidence_tokens(payload, tokens)
    return tokens or ["unknown"]


def _collect_confidence_tokens(value: Any, tokens: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = _normalize_token(key)
            if normalized_key in {"confidence_tier", "overall_confidence_tier", "tier", "confidence"} and isinstance(nested, str):
                tokens.append(_normalize_token(nested))
            else:
                _collect_confidence_tokens(nested, tokens)
    elif isinstance(value, list):
        for item in value:
            _collect_confidence_tokens(item, tokens)


def _flatten_numeric_values(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten_numeric_values(value, name))
            continue
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _summarize_corrections(values_by_target: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    all_values: list[float] = []
    for target, values in sorted(values_by_target.items()):
        all_values.extend(values)
        abs_values = [abs(value) for value in values]
        summary[target] = {
            "count": len(values),
            "mean_delta": round(statistics.fmean(values), 4),
            "mean_abs_delta": round(statistics.fmean(abs_values), 4),
            "max_abs_delta": round(max(abs_values), 4),
        }
    if all_values:
        abs_all = [abs(value) for value in all_values]
        summary["_overall"] = {
            "count": len(all_values),
            "mean_delta": round(statistics.fmean(all_values), 4),
            "mean_abs_delta": round(statistics.fmean(abs_all), 4),
            "max_abs_delta": round(max(abs_all), 4),
        }
    return summary


def _normalize_token(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", value.strip())
    return spaced.replace("-", "_").replace(" ", "_").lower()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def format_quality_report(report: dict[str, Any]) -> str:
    stats = report["statistics"]
    validation = report["validation"]
    lines = [
        "# Verified Measurement Dataset Quality Report",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Records file: `{report['records_file']}`",
        f"Valid for ingestion: `{report['valid']}`",
        f"Record count: `{stats['record_count']}`",
        "",
        "## Dataset Versions",
        "",
    ]
    for version, count in stats["dataset_versions"].items():
        lines.append(f"- `{version}`: {count}")

    lines.extend(["", "## Measurement Coverage", "", "| Measurement | Count | Coverage |", "| --- | ---: | ---: |"])
    for measurement, coverage in stats["measurement_coverage"].items():
        lines.append(f"| `{measurement}` | {coverage['count']} | {coverage['coverage']:.2%} |")

    lines.extend(["", "## Confidence Distribution", ""])
    if stats["confidence_distribution"]:
        lines.extend(f"- `{tier}`: {count}" for tier, count in stats["confidence_distribution"].items())
    else:
        lines.append("- No confidence metadata found.")

    lines.extend(["", "## Correction Distribution", "", "| Target | Count | Mean delta | Mean abs delta | Max abs delta |", "| --- | ---: | ---: | ---: | ---: |"])
    for target, correction in stats["correction_distribution"].items():
        lines.append(
            f"| `{target}` | {correction['count']} | {correction['mean_delta']:.4f} | "
            f"{correction['mean_abs_delta']:.4f} | {correction['max_abs_delta']:.4f} |"
        )

    lines.extend(["", "## Missing Field Counts", ""])
    if stats["missing_field_counts"]:
        lines.extend(f"- `{field}`: {count}" for field, count in stats["missing_field_counts"].items())
    else:
        lines.append("- No missing required or tracked metadata fields.")

    if validation["errors"]:
        lines.extend(["", "## Validation Errors", ""])
        lines.extend(f"- {error}" for error in validation["errors"])

    lines.extend(
        [
            "",
            "## Training Boundary",
            "",
            "This report is ingestion-only. It does not retrain, promote, or overwrite production model artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize verified FashionApp measurement dataset exports.")
    parser.add_argument("--dataset", required=True, help="Verified dataset root.")
    parser.add_argument("--records-file", help="Records file relative to the dataset root, or an absolute path.")
    parser.add_argument("--output", help="Optional output directory for quality report JSON and Markdown.")
    parser.add_argument("--allow-invalid", action="store_true", help="Write/report validation errors instead of failing fast.")
    args = parser.parse_args(argv)

    loader = VerifiedMeasurementDatasetLoader(args.dataset, args.records_file, validate=not args.allow_invalid)
    report = loader.quality_report()
    if args.output:
        outputs = loader.write_quality_report(args.output)
        print(f"Quality report JSON: {outputs['json']}")
        print(f"Quality report Markdown: {outputs['markdown']}")
    else:
        print(format_quality_report(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
