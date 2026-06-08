from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import three_view_measurements as route_module
from app.main import app
from app.services.three_view_measurements import ThreeViewMeasurementService


client = TestClient(app)


@pytest.fixture(autouse=True)
def use_fixture_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ThreeViewMeasurementService(predictions_csv=_prediction_fixture(tmp_path))
    monkeypatch.setattr(route_module, "three_view_measurement_service", service)


def test_three_view_payload_is_accepted() -> None:
    response = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["scanSessionId"] == "scan_session_a"
    assert payload["estimatedMeasurements"]["height"] == 172.5
    assert payload["normalizedInputs"]["front"]["hasStorageKey"] is True
    assert payload["normalizedInputs"]["side"]["hasStorageKey"] is True
    assert payload["normalizedInputs"]["back"]["hasStorageKey"] is True
    assert payload["scanQualitySummary"]["frontQuality"]["score"] > 0.8
    assert payload["scanQualitySummary"]["backQuality"]["score"] > 0.8
    assert payload["compatibilityMode"] is True


def test_front_only_payload_is_rejected() -> None:
    payload = _three_view_payload()
    for key in ("sideImageStorageKey", "backImageStorageKey"):
        payload.pop(key)

    response = client.post("/v1/body-ai/measurements/three-view", json=payload)

    assert response.status_code == 422
    assert "side, back" in response.text


def test_front_and_side_payload_is_rejected_without_back() -> None:
    payload = _three_view_payload()
    payload.pop("backImageStorageKey")

    response = client.post("/v1/body-ai/measurements/three-view", json=payload)

    assert response.status_code == 422
    assert "back" in response.text


def test_missing_back_is_rejected_even_with_back_metadata() -> None:
    payload = _three_view_payload()
    payload.pop("backImageStorageKey")
    payload["backPoseMetadata"] = {"confidenceScore": 0.95}

    response = client.post("/v1/body-ai/measurements/three-view", json=payload)

    assert response.status_code == 422
    assert "back" in response.text


def test_missing_height_is_rejected() -> None:
    payload = _three_view_payload()
    payload.pop("heightCm")

    response = client.post("/v1/body-ai/measurements/three-view", json=payload)

    assert response.status_code == 422
    assert "heightCm" in response.text


def test_pose_metadata_is_accepted_and_summarized() -> None:
    response = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload())

    assert response.status_code == 200
    pose = response.json()["poseSummary"]
    assert pose["metadataAvailable"] == {"front": True, "side": True, "back": True}
    assert pose["frontPoseConfidence"] == 0.96
    assert pose["overallPoseConfidence"] > 0.9


def test_validation_metadata_is_accepted_and_summarized() -> None:
    response = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload())

    assert response.status_code == 200
    validation = response.json()["validationSummary"]
    assert validation["metadataAvailable"] == {"front": True, "side": True, "back": True}
    assert validation["frontValidationScore"] == 0.94
    assert validation["overallValidationScore"] > 0.9


def test_source_types_map_is_accepted() -> None:
    payload = _three_view_payload()
    for key in ("frontSourceType", "sideSourceType", "backSourceType"):
        payload.pop(key)
    payload["sourceTypes"] = {"front": "camera", "side": "upload", "back": "camera"}

    response = client.post("/v1/body-ai/measurements/three-view", json=payload)

    assert response.status_code == 200
    scan_quality = response.json()["scanQualitySummary"]
    assert scan_quality["frontQuality"]["sourceType"] == "camera"
    assert scan_quality["sideQuality"]["sourceType"] == "upload"
    assert scan_quality["backQuality"]["sourceType"] == "camera"


def test_confidence_changes_when_validation_and_pose_quality_are_low() -> None:
    high = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload()).json()
    low_payload = _three_view_payload()
    low_payload["frontPoseMetadata"] = {"confidenceScore": 0.35, "missingBodyRegions": ["ankles"]}
    low_payload["sidePoseMetadata"] = {"confidenceScore": 0.4, "missingBodyRegions": ["shoulder"]}
    low_payload["backPoseMetadata"] = {"confidenceScore": 0.3, "missingBodyRegions": ["neck"]}
    low_payload["frontValidationMetadata"] = {"qualityScore": 0.4, "isValid": False, "missingBodyRegions": ["feet"]}
    low_payload["sideValidationMetadata"] = {"qualityScore": 0.35, "warnings": ["blurred"]}
    low_payload["backValidationMetadata"] = {"qualityScore": 0.3, "errors": ["body partially out of frame"]}

    low_response = client.post("/v1/body-ai/measurements/three-view", json=low_payload)

    assert low_response.status_code == 200
    low = low_response.json()
    assert low["overallScanConfidence"]["score"] < high["overallScanConfidence"]["score"]
    assert low["perMeasurementConfidence"]["chest"]["score"] < high["perMeasurementConfidence"]["chest"]["score"]
    assert low["scanQualitySummary"]["overallQuality"]["tier"] == "low"


def test_response_keeps_maker_review_and_real_world_validation_gates() -> None:
    response = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["makerReviewRequired"] is True
    assert payload["realWorldValidationStatus"] == "pending"


def test_response_warns_when_compatibility_estimator_is_used() -> None:
    response = client.post("/v1/body-ai/measurements/three-view", json=_three_view_payload())

    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("front/side-oriented" in warning for warning in warnings)
    assert any("back view" in warning for warning in warnings)
    assert any("Pose metadata" in warning for warning in warnings)
    assert any("Validation metadata" in warning for warning in warnings)


def _three_view_payload() -> dict:
    return {
        "scanSessionId": "scan_session_a",
        "heightCm": 172.5,
        "weightKg": 70.0,
        "requestPayloadVersion": "mobile_phase_d_three_view_v1",
        "userId": "user_a",
        "customerId": "customer_a",
        "orderId": "order_a",
        "frontImageStorageKey": "body-ai/scans/scan_session_a/front.jpg",
        "sideImageStorageKey": "body-ai/scans/scan_session_a/side.jpg",
        "backImageStorageKey": "body-ai/scans/scan_session_a/back.jpg",
        "frontSourceType": "camera",
        "sideSourceType": "camera",
        "backSourceType": "upload",
        "frontPoseMetadata": {"confidenceScore": 0.96, "visibleBodyRegions": ["torso", "legs"]},
        "sidePoseMetadata": {"confidenceScore": 0.93, "visibleBodyRegions": ["torso", "arms"]},
        "backPoseMetadata": {"confidenceScore": 0.91, "visibleBodyRegions": ["shoulders", "back"]},
        "frontValidationMetadata": {"qualityScore": 0.94, "isValid": True},
        "sideValidationMetadata": {"qualityScore": 0.92, "isValid": True},
        "backValidationMetadata": {"qualityScore": 0.9, "isValid": True},
    }


def _prediction_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.csv"
    rows = []
    for split, sample_id, error in (
        ("train", "sample_001", 1.0),
        ("val", "sample_002", 2.0),
        ("test", "sample_003", 1.5),
        ("test", "sample_000007", 1.4),
    ):
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
