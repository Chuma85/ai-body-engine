import json
from pathlib import Path

import pytest

from training.measurements import measurement_snapshot_store as snapshots
from training.measurements import measurement_result_schema as schema


def test_snapshot_save_load_round_trip(tmp_path: Path) -> None:
    result = _measurement_result()

    path = snapshots.save_snapshot(
        tmp_path,
        snapshot_id="snapshot_a",
        scan_id="scan_a",
        user_id="user_a",
        order_id="order_a",
        front_image_path="front.png",
        side_image_path="side.png",
        height_cm=172.0,
        measurement_result=result,
        created_at="2026-05-27T00:00:00Z",
    )
    loaded = snapshots.load_snapshot(path)

    assert loaded["snapshot_id"] == "snapshot_a"
    assert loaded["scan_id"] == "scan_a"
    assert loaded["measurement_result"]["metadata"]["synthetic_calibrated_only"] is True
    assert loaded["measurement_result"]["metadata"]["real_world_validated"] is False


def test_invalid_payload_fails_clearly() -> None:
    with pytest.raises(snapshots.MeasurementSnapshotError, match="missing required fields"):
        snapshots.create_snapshot(
            scan_id="scan_a",
            front_image_path="front.png",
            side_image_path="side.png",
            measurement_result={"metadata": {}},
            created_at="2026-05-27T00:00:00Z",
        )


def test_missing_scan_id_fails_clearly() -> None:
    with pytest.raises(snapshots.MeasurementSnapshotError, match="scan_id is required"):
        snapshots.create_snapshot(
            scan_id="",
            front_image_path="front.png",
            side_image_path="side.png",
            measurement_result=_measurement_result(),
            created_at="2026-05-27T00:00:00Z",
        )


def test_flags_are_preserved() -> None:
    snapshot = snapshots.create_snapshot(
        scan_id="scan_a",
        front_image_path="front.png",
        side_image_path="side.png",
        measurement_result=_measurement_result(),
        created_at="2026-05-27T00:00:00Z",
    )

    assert snapshot["synthetic_calibrated_only"] is True
    assert snapshot["real_world_validated"] is False


def test_list_snapshots_returns_stable_ordering(tmp_path: Path) -> None:
    result = _measurement_result()
    snapshots.save_snapshot(tmp_path, "scan_b", result, "front.png", "side.png", snapshot_id="snapshot_b", created_at="2026-05-27T00:00:02Z")
    snapshots.save_snapshot(tmp_path, "scan_a", result, "front.png", "side.png", snapshot_id="snapshot_a", created_at="2026-05-27T00:00:01Z")

    listed = snapshots.list_snapshots(tmp_path)

    assert [snapshot["snapshot_id"] for snapshot in listed] == ["snapshot_a", "snapshot_b"]


def test_corrupt_json_fails_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    with pytest.raises(snapshots.MeasurementSnapshotError, match="not valid JSON"):
        snapshots.load_snapshot(corrupt)


def test_snapshot_rejects_mismatched_metadata_flags() -> None:
    snapshot = snapshots.create_snapshot(
        scan_id="scan_a",
        front_image_path="front.png",
        side_image_path="side.png",
        measurement_result=_measurement_result(),
        created_at="2026-05-27T00:00:00Z",
    )
    snapshot["real_world_validated"] = True

    with pytest.raises(snapshots.MeasurementSnapshotError, match="does not match measurement metadata"):
        snapshots.validate_snapshot(snapshot)


def test_sample_snapshot_artifact_schema(tmp_path: Path) -> None:
    result = _measurement_result()
    snapshot = snapshots.create_snapshot(
        snapshot_id="sample_snapshot_phase_4i",
        scan_id="scan_a",
        front_image_path="front.png",
        side_image_path="side.png",
        height_cm=172.0,
        measurement_result=result,
        created_at="2026-05-27T00:00:00Z",
    )
    output_json = tmp_path / snapshots.SAMPLE_SNAPSHOT_JSON
    snapshots.write_json(output_json, snapshot)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))

    assert loaded["snapshot_schema_version"] == snapshots.SNAPSHOT_SCHEMA_VERSION
    assert "measurement_result" in loaded
    assert loaded["measurement_result"]["metadata"]["real_world_validated"] is False


def _measurement_result() -> schema.MeasurementResult:
    return schema.MeasurementResult(
        result_id="result_scan_a",
        sample_id="scan_a",
        dataset_split="test",
        targets=[
            schema.target_result_for_name(target, None)
            if target not in schema.AI_RESIDUAL_TARGETS
            else schema.MeasurementTargetResult(
                target=target,
                estimate_cm=100.0,
                interval=schema.MeasurementInterval(low_cm=98.0, high_cm=102.0, estimated_error_cm=2.0),
                confidence_tier=schema.MeasurementConfidence.HIGH,
                product_action=schema.MeasurementProductAction.ACCEPT_AS_AI_ESTIMATE,
                source=schema.MeasurementSource.AI_GEOMETRY_RESIDUAL,
                geometry_estimate_cm=99.0,
                residual_correction_cm=1.0,
                quality_flags=[schema.MeasurementQualityFlag.SYNTHETIC_CALIBRATED_ONLY],
            )
            for target in schema.SUPPORTED_TARGETS
        ],
        metadata=schema.MeasurementModelMetadata(
            model_version="model",
            pipeline_version="pipeline",
            calibration_version="calibration",
            training_dataset_id="dataset",
            generated_at="2026-05-27T00:00:00Z",
        ),
        caveats=["synthetic only"],
    )
