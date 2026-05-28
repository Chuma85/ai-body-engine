import json
from pathlib import Path

import pytest

from training.measurements import fashionapp_measurement_mapping as mapping
from training.measurements import production_measurement_package as package_mod


CREATED_AT = "2026-05-27T00:00:00Z"


def test_snake_case_maps_to_camel_case_deterministically() -> None:
    payload = {
        "estimate_cm": 101.2,
        "nested_value": {"final_garment_cm": 109.2},
        "target_rows": [{"confidence_tier": "high_confidence"}],
    }

    first = mapping.camelize_keys(payload)
    second = mapping.camelize_keys(payload)

    assert first == second
    assert first["estimateCm"] == 101.2
    assert first["nestedValue"]["finalGarmentCm"] == 109.2
    assert first["targetRows"][0]["confidenceTier"] == "high_confidence"


def test_customer_response_excludes_maker_ease_allowance() -> None:
    response = mapping.build_customer_measurement_response(_package())

    assert response["role"] == "customer"
    assert not mapping.contains_key(response, "makerEaseAllowanceCm")
    assert not mapping.contains_key(response, "finalGarmentCm")
    assert not mapping.contains_key(response, "makerVerifiedBodyCm")


def test_maker_response_includes_maker_ease_allowance() -> None:
    response = mapping.build_maker_measurement_response(_package())

    assert response["role"] == "maker"
    assert mapping.contains_key(response, "makerEaseAllowanceCm")
    assert mapping.contains_key(response, "finalGarmentCm")


def test_admin_response_includes_audit_references() -> None:
    package = _package()
    response = mapping.build_admin_measurement_response(package, audit_events=_audit_events())

    assert response["role"] == "admin"
    assert response["auditEventIds"] == package["audit_event_ids"]
    assert response["auditEvents"]
    assert response["auditEvents"][0]["eventId"]


def test_synthetic_and_real_world_flags_are_preserved() -> None:
    customer = mapping.build_customer_measurement_response(_package())
    maker = mapping.build_maker_measurement_response(_package())
    admin = mapping.build_admin_measurement_response(_package())

    assert customer["package"]["syntheticCalibratedOnly"] is True
    assert customer["package"]["realWorldValidated"] is False
    assert maker["package"]["syntheticCalibratedOnly"] is True
    assert maker["package"]["realWorldValidated"] is False
    assert admin["metadata"]["syntheticCalibratedOnly"] is True
    assert admin["metadata"]["realWorldValidated"] is False


def test_field_guidance_labels_are_frontend_friendly() -> None:
    customer = mapping.build_customer_measurement_response(_package())
    first_field = customer["fieldGuidance"][0]

    assert "fieldKey" in first_field
    assert "helperText" in first_field
    assert "infoIconText" in first_field
    assert "_" not in first_field["label"]


def test_api_payload_serialization_is_deterministic() -> None:
    response = mapping.build_maker_measurement_response(_package())

    first = mapping.json_dumps_deterministic(response)
    second = mapping.json_dumps_deterministic(response)

    assert first == second
    assert json.loads(first) == response


def test_missing_required_package_fields_fail_clearly() -> None:
    package = dict(_package())
    package.pop("measurement_snapshot_id")

    with pytest.raises(mapping.FashionAppMappingError, match="measurement_snapshot_id"):
        mapping.map_production_package_to_api_payload(package)


def test_sample_artifacts_are_written(tmp_path: Path) -> None:
    paths = mapping.export_sample_mapping(tmp_path, created_at=CREATED_AT)

    for path in paths.values():
        assert Path(path).exists()
    customer = json.loads(Path(paths["sample_customer_measurement_response_json"]).read_text(encoding="utf-8"))
    maker = json.loads(Path(paths["sample_maker_measurement_response_json"]).read_text(encoding="utf-8"))
    admin = json.loads(Path(paths["sample_admin_measurement_response_json"]).read_text(encoding="utf-8"))
    assert not mapping.contains_key(customer, "makerEaseAllowanceCm")
    assert mapping.contains_key(maker, "makerEaseAllowanceCm")
    assert admin["auditEvents"]


def _package() -> dict:
    maker_review = package_mod.sample_locked_maker_review(created_at=CREATED_AT)
    return package_mod.build_production_package(
        maker_review,
        audit_events=_audit_events(),
        package_id="package_4o_test",
        created_at=CREATED_AT,
    )


def _audit_events() -> list[dict]:
    from training.measurements.measurement_audit_trail import sample_audit_data

    events, _ = sample_audit_data()
    return events
