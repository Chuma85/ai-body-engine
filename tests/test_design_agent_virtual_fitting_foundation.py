from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.design_agent_virtual_fitting import BETA_FITTING_DISCLAIMER
from training.measurements import measurement_result_schema as schema


client = TestClient(app)


def test_design_session_can_be_created_from_measurement_context() -> None:
    response = client.post("/v1/body-ai/design-sessions", json=_create_session_payload())

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "created"
    assert payload["measurement_result_id"] == "result_scan_a"
    assert payload["body_profile_snapshot"]["measurement_summary_cm"]["chest"] == 100.0
    assert BETA_FITTING_DISCLAIMER in payload["warnings"]


def test_design_generation_returns_structured_options() -> None:
    session = _create_session()

    response = client.post(f"/v1/body-ai/design-sessions/{session['design_session_id']}/generate", json={"option_count": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "options_generated"
    assert len(payload["generated_options"]) == 3
    first = payload["generated_options"][0]
    assert first["title"]
    assert first["style_description"]
    assert first["garment_details"]
    assert first["asset_references"][0].startswith("synthetic://design-options/")
    assert first["confidence_metadata"]["mode"] == "deterministic_demo_design_agent"


def test_refinement_adds_new_variation() -> None:
    session = _create_session()
    generated = client.post(f"/v1/body-ai/design-sessions/{session['design_session_id']}/generate", json={"option_count": 1}).json()
    option_id = generated["generated_options"][0]["design_option_id"]

    response = client.post(
        f"/v1/body-ai/design-sessions/{session['design_session_id']}/refine",
        json={"design_option_id": option_id, "refinement_prompt": "make the sleeve softer and more formal"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refined"
    assert len(payload["generated_options"]) == 2
    assert payload["generated_options"][-1]["confidence_metadata"]["source_design_option_id"] == option_id


def test_fitting_preview_returns_beta_result_contract() -> None:
    session = _create_session()
    generated = client.post(f"/v1/body-ai/design-sessions/{session['design_session_id']}/generate", json={"option_count": 1}).json()
    option_id = generated["generated_options"][0]["design_option_id"]

    response = client.post(
        f"/v1/body-ai/design-sessions/{session['design_session_id']}/fitting-preview",
        json={"design_option_id": option_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "fitting_preview_ready"
    result = payload["fitting_results"][0]
    assert result["preview_status"] == "demo_synthetic_preview"
    assert result["preview_asset_references"][0].startswith("synthetic://virtual-fitting/")
    assert result["beta_preview_disclaimer"] == BETA_FITTING_DISCLAIMER
    assert result["confidence_metadata"]["maker_review_required"] is True


def test_production_brief_requires_approval_and_contains_handoff_contract() -> None:
    session = _create_session()
    session_id = session["design_session_id"]
    generated = client.post(f"/v1/body-ai/design-sessions/{session_id}/generate", json={"option_count": 1}).json()
    option_id = generated["generated_options"][0]["design_option_id"]

    before_approval = client.get(f"/v1/body-ai/design-sessions/{session_id}/production-brief")
    assert before_approval.status_code == 400

    client.post(f"/v1/body-ai/design-sessions/{session_id}/fitting-preview", json={"design_option_id": option_id})
    approved = client.post(
        f"/v1/body-ai/design-sessions/{session_id}/approve",
        json={"design_option_id": option_id, "maker_production_notes": "Use light lining and confirm hem allowance."},
    )
    assert approved.status_code == 200

    response = client.get(f"/v1/body-ai/design-sessions/{session_id}/production-brief")

    assert response.status_code == 200
    brief = response.json()
    assert brief["approval_state"] == "approved"
    assert brief["approved_design_option"]["design_option_id"] == option_id
    assert brief["measurement_references"]["measurement_result_id"] == "result_scan_a"
    assert "Use light lining and confirm hem allowance." in brief["maker_notes"]
    assert BETA_FITTING_DISCLAIMER in brief["disclaimers"]
    assert brief["fit_notes"]


def test_measurement_result_contract_remains_compatible_with_design_session() -> None:
    result = _measurement_result()
    payload = result.to_payload()

    response = client.post("/v1/body-ai/design-sessions", json=_create_session_payload(measurement_result=payload))

    assert response.status_code == 201
    body_profile = response.json()["body_profile_snapshot"]
    assert payload["targets"][0]["target"] == "chest"
    assert body_profile["measurement_result_id"] == result.result_id
    assert body_profile["real_world_validated"] is False
    assert body_profile["synthetic_calibrated_only"] is True


def _create_session() -> dict:
    response = client.post("/v1/body-ai/design-sessions", json=_create_session_payload())
    assert response.status_code == 201
    return response.json()


def _create_session_payload(measurement_result: dict | None = None) -> dict:
    measurement_result = measurement_result or _measurement_result().to_payload()
    return {
        "user_id": "user_a",
        "session_reference_id": "fashionapp_session_a",
        "order_id": "order_a",
        "scan_id": "scan_a",
        "measurement_result": measurement_result,
        "preferences": {
            "garment_type": "evening dress",
            "color_palette": ["emerald", "ivory"],
            "style_direction": "modern minimal",
            "fit_preference": "close at bodice with easy skirt",
            "occasion": "wedding guest",
            "fabric_preference": "silk crepe",
            "inspiration_notes": "clean neckline and subtle movement",
            "maker_notes": "Customer prefers modest neckline.",
        },
    }


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
            generated_at="2026-05-31T00:00:00Z",
        ),
        caveats=["synthetic only"],
    )

