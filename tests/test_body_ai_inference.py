import csv
import json
from pathlib import Path

import pytest

from training.measurements import body_ai_inference as inference


def test_inference_wrapper_returns_valid_measurement_result(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)

    result = inference.run_body_ai_measurement(
        scan_id="sample_003",
        front_image_path=front,
        side_image_path=side,
        height_cm=172.5,
        predictions_csv=predictions,
        generated_at="2026-05-27T00:00:00Z",
    )
    payload = result.to_payload()

    assert payload["sample_id"] == "sample_003"
    assert payload["metadata"]["synthetic_calibrated_only"] is True
    assert payload["metadata"]["real_world_validated"] is False
    assert payload["metadata"]["pipeline_version"] == "phase_4h_body_ai_inference_wrapper"
    chest = payload["targets"][0]
    assert chest["source"] == "ai_geometry_residual"
    assert chest["interval"]["low_cm"] <= chest["estimate_cm"] <= chest["interval"]["high_cm"]


def test_missing_height_maps_to_user_input_required(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)

    result = inference.run_body_ai_measurement(
        scan_id="sample_003",
        front_image_path=front,
        side_image_path=side,
        predictions_csv=predictions,
        generated_at="2026-05-27T00:00:00Z",
    )
    height = next(target for target in result.to_payload()["targets"] if target["target"] == "height")

    assert height["estimate_cm"] is None
    assert height["product_action"] == "user_input_required"
    assert height["source"] == "manual_user_input_required"


def test_missing_front_or_side_image_fails_clearly(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)

    with pytest.raises(inference.BodyAIInferenceError, match="Missing front image"):
        inference.run_body_ai_measurement("sample_003", tmp_path / "missing_front.png", side, predictions_csv=predictions)
    with pytest.raises(inference.BodyAIInferenceError, match="Missing side image"):
        inference.run_body_ai_measurement("sample_003", front, tmp_path / "missing_side.png", predictions_csv=predictions)


def test_unavailable_model_artifact_fails_clearly(tmp_path: Path) -> None:
    front, side = _image_fixtures(tmp_path)

    with pytest.raises(FileNotFoundError, match="Local demo prediction artifact is unavailable"):
        inference.run_body_ai_measurement("sample_003", front, side, predictions_csv=tmp_path / "missing_predictions.csv")


def test_invalid_image_extension_fails_clearly(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front = tmp_path / "front.txt"
    side = tmp_path / "side.png"
    front.write_text("not an image", encoding="utf-8")
    side.write_bytes(b"side")

    with pytest.raises(inference.BodyAIInferenceError, match="PNG or JPEG"):
        inference.run_body_ai_measurement("sample_003", front, side, predictions_csv=predictions)


def test_weak_targets_map_to_manual_or_landmark_actions(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)
    result = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        predictions_csv=predictions,
        generated_at="2026-05-27T00:00:00Z",
    )
    targets = {target["target"]: target for target in result.to_payload()["targets"]}

    assert targets["height"]["product_action"] == "user_input_required"
    assert targets["inseam"]["source"] == "landmark_required"
    assert targets["neck"]["source"] == "landmark_required"
    assert targets["sleeve"]["source"] == "landmark_required"


def test_json_serialization_is_deterministic(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)
    result = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-05-27T00:00:00Z",
    )

    first = inference.to_json(result)
    second = inference.to_json(result)
    output_path = tmp_path / "result.json"
    inference.save_result_json(result, output_path)

    assert first == second
    assert json.loads(output_path.read_text(encoding="utf-8")) == json.loads(first)


def test_export_sample_inference_artifacts(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front, side = _image_fixtures(tmp_path)
    service = inference.BodyAIMeasurementService(predictions_csv=predictions)
    result = service.predict("sample_003", front, side, generated_at="2026-05-27T00:00:00Z")
    output = tmp_path / "out"
    inference.save_result_json(result, output / inference.SAMPLE_INFERENCE_RESULT_JSON)
    (output / inference.INFERENCE_SUMMARY_MD).write_text(inference.format_inference_summary(result), encoding="utf-8")

    assert (output / inference.SAMPLE_INFERENCE_RESULT_JSON).exists()
    assert (output / inference.INFERENCE_SUMMARY_MD).exists()


def _image_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    front.write_bytes(b"front")
    side.write_bytes(b"side")
    return front, side


def _prediction_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.csv"
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
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
