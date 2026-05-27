from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from training.measurements.body_ai_inference import run_body_ai_measurement
from training.measurements.measurement_result_schema import SUPPORTED_TARGETS, MeasurementResult, write_json

SNAPSHOT_SCHEMA_VERSION = "phase_4i_measurement_snapshot_v1"
SAMPLE_SNAPSHOT_JSON = "sample_snapshot.json"
SNAPSHOT_SUMMARY_MD = "snapshot_store_summary.md"
DEFAULT_OUTPUT_DIR = "artifacts/phase_4i_measurement_snapshots"


class MeasurementSnapshotError(ValueError):
    """Raised when a measurement snapshot cannot be validated or loaded."""


def create_snapshot(
    scan_id: str,
    measurement_result: MeasurementResult | dict[str, Any],
    front_image_path: str | Path,
    side_image_path: str | Path,
    height_cm: float | None = None,
    user_id: str | None = None,
    order_id: str | None = None,
    snapshot_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if not scan_id:
        raise MeasurementSnapshotError("scan_id is required for measurement snapshots.")
    payload = measurement_payload(measurement_result)
    validate_measurement_result_payload(payload)
    metadata = payload["metadata"]
    timestamp = created_at or utc_now()
    snapshot = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id or f"measurement_snapshot_{uuid4().hex}",
        "scan_id": scan_id,
        "user_id": user_id,
        "order_id": order_id,
        "front_image_path": str(front_image_path),
        "side_image_path": str(side_image_path),
        "height_cm": height_cm,
        "measurement_result": payload,
        "model_version": metadata["model_version"],
        "pipeline_version": metadata["pipeline_version"],
        "calibration_version": metadata["calibration_version"],
        "synthetic_calibrated_only": bool(metadata["synthetic_calibrated_only"]),
        "real_world_validated": bool(metadata["real_world_validated"]),
        "created_at": timestamp,
        "updated_at": updated_at,
    }
    validate_snapshot(snapshot)
    return snapshot


def save_snapshot(
    snapshot_dir: str | Path,
    scan_id: str,
    measurement_result: MeasurementResult | dict[str, Any],
    front_image_path: str | Path,
    side_image_path: str | Path,
    height_cm: float | None = None,
    user_id: str | None = None,
    order_id: str | None = None,
    snapshot_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> Path:
    snapshot = create_snapshot(
        scan_id=scan_id,
        user_id=user_id,
        order_id=order_id,
        front_image_path=front_image_path,
        side_image_path=side_image_path,
        height_cm=height_cm,
        measurement_result=measurement_result,
        snapshot_id=snapshot_id,
        created_at=created_at,
        updated_at=updated_at,
    )
    output_dir = Path(snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{snapshot['snapshot_id']}.json"
    write_json(output_path, snapshot)
    return output_path


def load_snapshot(path: str | Path) -> dict[str, Any]:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Measurement snapshot does not exist: {snapshot_path}")
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeasurementSnapshotError(f"Measurement snapshot is not valid JSON: {snapshot_path}") from exc
    validate_snapshot(payload)
    return payload


def list_snapshots(snapshot_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(snapshot_dir)
    if not directory.exists():
        return []
    snapshots = [load_snapshot(path) for path in sorted(directory.glob("*.json"))]
    return sorted(snapshots, key=lambda snapshot: (str(snapshot["created_at"]), str(snapshot["snapshot_id"])))


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    required = {
        "snapshot_schema_version",
        "snapshot_id",
        "scan_id",
        "front_image_path",
        "side_image_path",
        "measurement_result",
        "model_version",
        "pipeline_version",
        "calibration_version",
        "synthetic_calibrated_only",
        "real_world_validated",
        "created_at",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise MeasurementSnapshotError(f"Measurement snapshot is missing required fields: {', '.join(missing)}")
    if snapshot["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise MeasurementSnapshotError(f"Unsupported snapshot schema version: {snapshot['snapshot_schema_version']}")
    if not snapshot["scan_id"]:
        raise MeasurementSnapshotError("Measurement snapshot requires a non-empty scan_id.")
    validate_measurement_result_payload(snapshot["measurement_result"])
    metadata = snapshot["measurement_result"]["metadata"]
    if bool(snapshot["synthetic_calibrated_only"]) != bool(metadata["synthetic_calibrated_only"]):
        raise MeasurementSnapshotError("Snapshot synthetic_calibrated_only does not match measurement metadata.")
    if bool(snapshot["real_world_validated"]) != bool(metadata["real_world_validated"]):
        raise MeasurementSnapshotError("Snapshot real_world_validated does not match measurement metadata.")
    if snapshot["real_world_validated"]:
        raise MeasurementSnapshotError("Real-world validated snapshots are not supported until real validation exists.")


def validate_measurement_result_payload(payload: dict[str, Any]) -> None:
    required = {"result_id", "sample_id", "dataset_split", "targets", "metadata", "caveats"}
    missing = sorted(required - set(payload))
    if missing:
        raise MeasurementSnapshotError(f"Measurement result payload is missing required fields: {', '.join(missing)}")
    metadata = payload.get("metadata") or {}
    metadata_required = {
        "model_version",
        "pipeline_version",
        "calibration_version",
        "training_dataset_id",
        "readiness_level",
        "synthetic_calibrated_only",
        "real_world_validated",
        "generated_at",
    }
    missing_metadata = sorted(metadata_required - set(metadata))
    if missing_metadata:
        raise MeasurementSnapshotError(f"Measurement result metadata is missing required fields: {', '.join(missing_metadata)}")
    if metadata["real_world_validated"]:
        raise MeasurementSnapshotError("Measurement result must have real_world_validated=false for Phase 4I snapshots.")
    if not metadata["synthetic_calibrated_only"]:
        raise MeasurementSnapshotError("Measurement result must preserve synthetic_calibrated_only=true for Phase 4I snapshots.")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise MeasurementSnapshotError("Measurement result payload requires a non-empty targets list.")
    target_names = [target.get("target") for target in targets]
    if target_names != SUPPORTED_TARGETS:
        raise MeasurementSnapshotError(f"Measurement targets must match stable order: {SUPPORTED_TARGETS}")
    for target in targets:
        validate_target_payload(target)


def validate_target_payload(target: dict[str, Any]) -> None:
    required = {"target", "estimate_cm", "interval", "confidence_tier", "product_action", "source", "quality_flags", "notes"}
    missing = sorted(required - set(target))
    if missing:
        raise MeasurementSnapshotError(f"Measurement target {target.get('target', '<unknown>')} is missing fields: {', '.join(missing)}")
    if not target["product_action"]:
        raise MeasurementSnapshotError(f"Measurement target {target['target']} requires a product_action.")
    interval = target.get("interval") or {}
    estimate = target.get("estimate_cm")
    if estimate is None:
        return
    for field in ("low_cm", "high_cm", "estimated_error_cm"):
        if interval.get(field) is None:
            raise MeasurementSnapshotError(f"Measurement target {target['target']} interval requires {field}.")
    if float(interval["low_cm"]) > float(estimate) or float(interval["high_cm"]) < float(estimate):
        raise MeasurementSnapshotError(f"Measurement target {target['target']} has an interval that does not contain the estimate.")


def measurement_payload(measurement_result: MeasurementResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(measurement_result, MeasurementResult):
        return measurement_result.to_payload()
    if not isinstance(measurement_result, dict):
        raise MeasurementSnapshotError("measurement_result must be a MeasurementResult or payload dictionary.")
    return measurement_result


def export_sample_snapshot(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    front_image_path: str | Path = "data/synthetic/phase_3t/images/front/sample_000001_front.png",
    side_image_path: str | Path = "data/synthetic/phase_3t/images/side/sample_000001_side.png",
    scan_id: str = "sample_000007",
    user_id: str | None = "demo_user",
    order_id: str | None = "demo_order",
    height_cm: float | None = 172.0,
    created_at: str | None = None,
) -> dict[str, Any]:
    result = run_body_ai_measurement(
        scan_id=scan_id,
        user_id=user_id,
        order_id=order_id,
        front_image_path=front_image_path,
        side_image_path=side_image_path,
        height_cm=height_cm,
        generated_at=created_at,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = create_snapshot(
        snapshot_id="sample_snapshot_phase_4i",
        scan_id=scan_id,
        user_id=user_id,
        order_id=order_id,
        front_image_path=front_image_path,
        side_image_path=side_image_path,
        height_cm=height_cm,
        measurement_result=result,
        created_at=created_at,
    )
    write_json(output_path / SAMPLE_SNAPSHOT_JSON, snapshot)
    (output_path / SNAPSHOT_SUMMARY_MD).write_text(format_snapshot_summary(snapshot), encoding="utf-8")
    return {
        "sample_snapshot_json": str(output_path / SAMPLE_SNAPSHOT_JSON),
        "snapshot_store_summary_md": str(output_path / SNAPSHOT_SUMMARY_MD),
        "snapshot": snapshot,
    }


def format_snapshot_summary(snapshot: dict[str, Any]) -> str:
    metadata = snapshot["measurement_result"]["metadata"]
    return "\n".join(
        [
            "# Phase 4I Measurement Snapshot Persistence",
            "",
            f"Snapshot: `{snapshot['snapshot_id']}`",
            f"Scan: `{snapshot['scan_id']}`",
            f"Pipeline: `{snapshot['pipeline_version']}`",
            f"Calibration: `{snapshot['calibration_version']}`",
            f"Synthetic calibrated only: `{snapshot['synthetic_calibrated_only']}`",
            f"Real-world validated: `{snapshot['real_world_validated']}`",
            f"Readiness: `{metadata['readiness_level']}`",
            "",
            "This snapshot is a durable local JSON record for audit and future FashionApp persistence mapping.",
            "It is synthetic-calibrated only and must not be treated as production tape-measure validation.",
            "",
        ]
    )


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a local Body AI measurement snapshot.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--front-image", default="data/synthetic/phase_3t/images/front/sample_000001_front.png")
    parser.add_argument("--side-image", default="data/synthetic/phase_3t/images/side/sample_000001_side.png")
    parser.add_argument("--scan-id", default="sample_000007")
    parser.add_argument("--user-id")
    parser.add_argument("--order-id")
    parser.add_argument("--height-cm", type=float, default=172.0)
    args = parser.parse_args(argv)
    result = export_sample_snapshot(
        output_dir=args.output,
        front_image_path=args.front_image,
        side_image_path=args.side_image,
        scan_id=args.scan_id,
        user_id=args.user_id,
        order_id=args.order_id,
        height_cm=args.height_cm,
    )
    print(f"Sample snapshot: {result['sample_snapshot_json']}")
    print(f"Snapshot summary: {result['snapshot_store_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
