import json
from pathlib import Path

import pytest

from training.measurements import measurement_field_guidance as guidance


def test_customer_guidance_does_not_include_maker_ease_allowance() -> None:
    fields = guidance.list_guidance_for_role("customer")
    keys = {field["field_key"] for field in fields}

    assert "maker_ease_allowance_cm" not in keys
    with pytest.raises(guidance.FieldGuidanceError, match="not visible"):
        guidance.get_field_guidance("maker_ease_allowance_cm", "customer")


def test_maker_guidance_includes_ease_with_clear_helper_text() -> None:
    field = guidance.get_field_guidance("maker_ease_allowance_cm", "maker")

    assert field["required"] is True
    assert "maker-only" in field["info_icon_text"]
    assert "not seam allowance" in field["info_icon_text"]
    assert field["editable_by"] == "maker"


def test_every_interactive_customer_field_has_info_icon_text() -> None:
    for field in guidance.list_guidance_for_role("customer"):
        if field["editable_by"] == "customer":
            assert field["info_icon_text"]
            assert field["helper_text"]


def test_every_interactive_maker_field_has_info_icon_text() -> None:
    for field in guidance.list_guidance_for_role("maker"):
        if field["editable_by"] == "maker":
            assert field["info_icon_text"]
            assert field["helper_text"]


def test_height_guidance_says_user_input_is_required() -> None:
    field = guidance.get_field_guidance("height_cm", "customer")

    assert field["required"] is True
    assert "required" in field["info_icon_text"].lower()
    assert "should not guess height" in field["helper_text"]


def test_fit_preference_guidance_says_it_is_not_a_measurement() -> None:
    field = guidance.get_field_guidance("fit_preference", "customer")

    assert "not a measurement" in field["info_icon_text"]
    assert "not maker ease" in field["info_icon_text"]


def test_final_garment_guidance_explains_body_plus_ease() -> None:
    field = guidance.get_field_guidance("final_garment_cm", "maker")

    assert "selected body measurement + maker ease/allowance" in field["info_icon_text"]


def test_synthetic_calibrated_guidance_explains_real_world_caveat() -> None:
    field = guidance.get_field_guidance("synthetic_calibrated_only", "admin")

    assert "not real-world production validation" in field["info_icon_text"]
    assert field["warning_text"]


def test_export_guidance_json_is_deterministic() -> None:
    first = guidance.export_guidance_json("maker")
    second = guidance.export_guidance_json("maker")

    assert first == second
    payload = json.loads(first)
    assert payload["role"] == "maker"
    assert payload["fields"]


def test_unknown_field_fails_clearly() -> None:
    with pytest.raises(guidance.FieldGuidanceError, match="Unknown measurement guidance field"):
        guidance.get_field_guidance("not_a_field", "customer")


def test_reject_customer_ease_fields_rejects_nested_payload() -> None:
    with pytest.raises(guidance.FieldGuidanceError, match="maker-only ease/allowance"):
        guidance.reject_customer_ease_fields({"measurements": [{"target": "chest", "maker_ease_allowance_cm": 5.0}]})


def test_sample_guidance_artifacts_are_written(tmp_path: Path) -> None:
    paths = guidance.export_sample_guidance(tmp_path)

    for path in paths.values():
        assert Path(path).exists()
    customer_payload = json.loads(Path(paths["customer_field_guidance_json"]).read_text(encoding="utf-8"))
    assert customer_payload["role"] == "customer"
