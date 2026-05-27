import json
from pathlib import Path

import pytest

from training.measurements import measurement_result_schema as schema


def test_schema_serialization_is_deterministic() -> None:
    result = _result()

    first = json.dumps(result.to_payload(), sort_keys=True)
    second = json.dumps(result.to_payload(), sort_keys=True)

    assert first == second
    assert result.to_payload()["metadata"]["real_world_validated"] is False


def test_interval_validation_works() -> None:
    with pytest.raises(ValueError, match="expected low <= estimate <= high"):
        schema.MeasurementTargetResult(
            target="chest",
            estimate_cm=100.0,
            interval=schema.MeasurementInterval(low_cm=101.0, high_cm=105.0, estimated_error_cm=2.0),
            confidence_tier=schema.MeasurementConfidence.HIGH,
            product_action=schema.MeasurementProductAction.ACCEPT_AS_AI_ESTIMATE,
            source=schema.MeasurementSource.AI_GEOMETRY_RESIDUAL,
        )


def test_missing_confidence_fails_for_ai_targets() -> None:
    with pytest.raises(ValueError, match="requires a confidence tier"):
        schema.MeasurementTargetResult(
            target="waist",
            estimate_cm=88.0,
            interval=schema.MeasurementInterval(low_cm=86.0, high_cm=90.0, estimated_error_cm=2.0),
            confidence_tier=schema.MeasurementConfidence.NOT_APPLICABLE,
            product_action=schema.MeasurementProductAction.ACCEPT_AS_AI_ESTIMATE,
            source=schema.MeasurementSource.AI_GEOMETRY_RESIDUAL,
        )


def test_real_world_validated_defaults_false() -> None:
    metadata = schema.MeasurementModelMetadata(
        model_version="model",
        pipeline_version="pipeline",
        calibration_version="calibration",
        training_dataset_id="dataset",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert metadata.real_world_validated is False
    assert metadata.synthetic_calibrated_only is True


def test_weak_targets_map_to_manual_or_landmark_required_actions() -> None:
    height = schema.target_result_for_name("height", None)
    inseam = schema.target_result_for_name("inseam", None)
    neck = schema.target_result_for_name("neck", None)
    sleeve = schema.target_result_for_name("sleeve", None)

    assert height.product_action == schema.MeasurementProductAction.USER_INPUT_REQUIRED
    assert height.source == schema.MeasurementSource.MANUAL_USER_INPUT_REQUIRED
    assert {inseam.source, neck.source, sleeve.source} == {schema.MeasurementSource.LANDMARK_REQUIRED}
    assert all(result.product_action == schema.MeasurementProductAction.REQUIRE_MANUAL_CONFIRMATION for result in (inseam, neck, sleeve))


def test_sample_payload_contains_expected_fields(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_prediction_fixture(predictions)

    result = schema.export_sample_measurement_result(
        predictions,
        tmp_path / "out",
        run_name="geometry_plus_residual__gradient_boosting",
        sample_id="sample_003",
        generated_at="2026-05-27T00:00:00Z",
    )
    payload = result["payload"]

    assert Path(result["sample_measurement_result_json"]).exists()
    assert Path(result["measurement_schema_summary_md"]).exists()
    assert payload["sample_id"] == "sample_003"
    assert [target["target"] for target in payload["targets"]] == schema.SUPPORTED_TARGETS
    chest = payload["targets"][0]
    assert {
        "estimate_cm",
        "interval",
        "confidence_tier",
        "product_action",
        "geometry_estimate_cm",
        "residual_correction_cm",
        "source",
        "quality_flags",
        "notes",
    } <= set(chest)
    thigh = next(target for target in payload["targets"] if target["target"] == "thigh")
    assert "calibration_risk" in thigh["quality_flags"]
    height = next(target for target in payload["targets"] if target["target"] == "height")
    assert height["product_action"] == "user_input_required"
    assert payload["metadata"]["real_world_validated"] is False


def _result() -> schema.MeasurementResult:
    return schema.MeasurementResult(
        result_id="result",
        sample_id="sample_001",
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
    )


def _write_prediction_fixture(path: Path) -> None:
    rows = []
    for split, sample_id, error in (("train", "sample_001", 1.0), ("val", "sample_002", 2.0), ("test", "sample_003", 1.5)):
        for target in ("chest_cm", "waist_cm", "hip_cm", "thigh_cm"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_split": split,
                    "target": target,
                    "model_name": "gradient_boosting",
                    "run_name": "geometry_plus_residual__gradient_boosting",
                    "geometry_estimate_cm": "95.0",
                    "calibrated_label_cm": str(100.0 + error),
                    "residual_cm": "5.0",
                    "predicted_residual_cm": "5.0",
                    "final_estimate_cm": "100.0",
                    "abs_error_cm": str(error),
                    "confidence_flags": "ok",
                    "geometry_quality_flags": "ok",
                }
            )
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        import csv

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
