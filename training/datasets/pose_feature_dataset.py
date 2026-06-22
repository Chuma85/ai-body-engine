from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from app.schemas.pose_metadata import CapturedPhotoPoseMetadata
from training.features.pose_features import (
    POSE_DATASET_FEATURE_NAMES,
    extract_pose_features,
    missing_required_landmarks,
)


DEFAULT_QUALITY_THRESHOLD = 0.75
SUPPORTED_VIEWS = {"front", "side", "back"}

POSE_METADATA_FIELD_NAMES = [
    "participant_id",
    "tester_id",
    "capture_session_id",
    "view",
    "image_path",
    "timestamp",
    "pose_quality_score",
    "pose_quality_score_normalized",
    "quality_reasons",
    "segmentation_available",
    "manual_capture",
    "approved_for_training",
]

POSE_FEATURE_DATASET_COLUMNS = [*POSE_METADATA_FIELD_NAMES, *POSE_DATASET_FEATURE_NAMES]
REJECTED_ROW_COLUMNS = [
    "source_file",
    "participant_id",
    "tester_id",
    "capture_session_id",
    "view",
    "image_path",
    "pose_quality_score",
    "rejection_reasons",
]


class PoseFeatureDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PoseFeatureDatasetBuildResult:
    input_path: Path
    output_path: Path
    rejected_report_path: Path
    accepted_count: int
    rejected_count: int
    total_count: int


def build_pose_feature_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    rejected_report_path: str | Path | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> PoseFeatureDatasetBuildResult:
    input_root = Path(input_path)
    output_file = Path(output_path)
    rejected_file = Path(rejected_report_path) if rejected_report_path is not None else _default_rejected_report_path(output_file)
    if not input_root.exists():
        raise PoseFeatureDatasetError(f"Input path does not exist: {input_root}")

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for source_file, raw_record in iter_capture_metadata(input_root):
        metadata, parse_reasons = _parse_metadata(raw_record)
        if metadata is None:
            rejected_rows.append(_rejected_row(source_file, raw_record, parse_reasons))
            continue

        rejection_reasons = validate_pose_capture(metadata, quality_threshold=quality_threshold)
        if rejection_reasons:
            rejected_rows.append(_rejected_row(source_file, raw_record, rejection_reasons, metadata=metadata))
            continue

        features = extract_pose_features(metadata)
        accepted_rows.append(
            {
                **metadata.metadata_fields(),
                "pose_quality_score_normalized": normalize_quality_score(metadata.pose_quality_score),
                **features,
            }
        )

    _write_csv(output_file, POSE_FEATURE_DATASET_COLUMNS, accepted_rows)
    _write_csv(rejected_file, REJECTED_ROW_COLUMNS, rejected_rows)
    return PoseFeatureDatasetBuildResult(
        accepted_count=len(accepted_rows),
        input_path=input_root,
        output_path=output_file,
        rejected_count=len(rejected_rows),
        rejected_report_path=rejected_file,
        total_count=len(accepted_rows) + len(rejected_rows),
    )


def iter_capture_metadata(input_path: str | Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    root = Path(input_path)
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*.json") if path.is_file())
    for file_path in files:
        payload = _read_json(file_path)
        for record in _records_from_payload(payload):
            if isinstance(record, dict):
                yield file_path, record
            else:
                yield file_path, {"_raw_record": record}


def validate_pose_capture(metadata: CapturedPhotoPoseMetadata, *, quality_threshold: float = DEFAULT_QUALITY_THRESHOLD) -> list[str]:
    reasons: list[str] = []
    if metadata.view.value not in SUPPORTED_VIEWS:
        reasons.append("wrong_view_type")
    if not metadata.image_path.strip():
        reasons.append("image_path_missing")
    if normalize_quality_score(metadata.pose_quality_score) < quality_threshold:
        reasons.append("low_pose_quality_score")
    missing = missing_required_landmarks(metadata)
    if missing:
        reasons.append(f"required_landmarks_missing:{'|'.join(missing)}")
    if metadata.manual_capture and not metadata.approved_for_training:
        reasons.append("manual_capture_without_admin_approval")
    return reasons


def normalize_quality_score(score: float) -> float:
    value = float(score)
    if value > 1.0:
        return value / 100.0
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = build_pose_feature_dataset(
        args.input,
        args.output,
        rejected_report_path=args.rejected_report,
        quality_threshold=args.quality_threshold,
    )
    print(
        "Pose feature dataset built: "
        f"{result.accepted_count} accepted, {result.rejected_count} rejected, "
        f"output={result.output_path}, rejected_report={result.rejected_report_path}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert captured MediaPipe pose metadata JSON into a model-ready pose feature CSV."
    )
    parser.add_argument("--input", required=True, help="JSON file or directory containing captured photo metadata JSON.")
    parser.add_argument("--output", required=True, help="CSV path to write accepted pose feature rows.")
    parser.add_argument(
        "--rejected-report",
        default=None,
        help="CSV path to write rejected capture rows. Defaults to <output_stem>_rejected_rows.csv.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLD,
        help="Minimum normalized pose quality score. Values above 1 are treated as 0-100 input scores.",
    )
    return parser


def _parse_metadata(raw_record: dict[str, Any]) -> tuple[CapturedPhotoPoseMetadata | None, list[str]]:
    try:
        return CapturedPhotoPoseMetadata.model_validate(raw_record), []
    except ValidationError as error:
        return None, _validation_reasons(error)


def _validation_reasons(error: ValidationError) -> list[str]:
    reasons: list[str] = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", []))
        issue_type = str(issue.get("type", "validation_error"))
        if location == "view":
            reasons.append("wrong_view_type")
        elif issue_type == "missing":
            reasons.append(f"missing_required_field:{location}")
        elif location == "imagePath":
            reasons.append("image_path_missing")
        else:
            reasons.append(f"invalid_field:{location or issue_type}")
    return reasons or ["invalid_metadata"]


def _records_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return [{"_raw_record": payload}]
    for key in ("captures", "records", "photos", "metadata", "pose_metadata", "poseMetadata"):
        nested = payload.get(key)
        if isinstance(nested, list):
            return nested
    return [payload]


def _rejected_row(
    source_file: Path,
    raw_record: dict[str, Any],
    reasons: list[str],
    *,
    metadata: CapturedPhotoPoseMetadata | None = None,
) -> dict[str, Any]:
    if metadata is not None:
        values = metadata.metadata_fields()
        quality_score = values["pose_quality_score"]
    else:
        values = {
            "participant_id": _first_present(raw_record, "participant_id", "participantId"),
            "tester_id": _first_present(raw_record, "tester_id", "testerId"),
            "capture_session_id": _first_present(raw_record, "capture_session_id", "captureSessionId"),
            "view": _first_present(raw_record, "view"),
            "image_path": _first_present(raw_record, "image_path", "imagePath"),
        }
        quality_score = _first_present(raw_record, "pose_quality_score", "poseQualityScore")
    return {
        "source_file": str(source_file),
        "participant_id": values.get("participant_id"),
        "tester_id": values.get("tester_id"),
        "capture_session_id": values.get("capture_session_id"),
        "view": values.get("view"),
        "image_path": values.get("image_path"),
        "pose_quality_score": quality_score,
        "rejection_reasons": "|".join(reasons),
    }


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _default_rejected_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_rejected_rows.csv")


if __name__ == "__main__":
    raise SystemExit(main())
