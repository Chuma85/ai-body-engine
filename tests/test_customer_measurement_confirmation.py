import json
from pathlib import Path

import pytest

from training.measurements import customer_measurement_confirmation as confirmation


def test_customer_confirmation_payload_serializes_deterministically() -> None:
    payload = _payload()

    first = json.dumps(payload, sort_keys=True)
    second = json.dumps(payload, sort_keys=True)

    assert first == second
    assert payload["fit_preference"] == "regular"
    assert payload["synthetic_calibrated_only"] is True
    assert payload["real_world_validated"] is False


def test_required_manual_targets_are_enforced() -> None:
    payload = _payload()

    with pytest.raises(confirmation.CustomerConfirmationError, match="Missing required manual measurement values"):
        confirmation.validate_customer_confirmation_payload(payload, require_complete=True)

    updated = confirmation.apply_customer_measurement_updates(
        payload,
        {
            "height": {"customer_manual_cm": 172.0},
            "inseam": {"customer_manual_cm": 78.0},
            "sleeve": {"customer_manual_cm": 61.0},
            "neck": {"customer_manual_cm": 38.0},
            "chest": {"customer_confirmed_cm": 101.0},
            "waist": {"customer_confirmed_cm": 82.0},
            "hip": {"customer_confirmed_cm": 104.0},
        },
        updated_at="2026-05-27T00:01:00Z",
    )

    confirmation.validate_customer_confirmation_payload(updated, require_complete=True)


def test_require_manual_confirmation_targets_are_not_auto_finalized() -> None:
    payload = _payload()
    updated = confirmation.apply_customer_measurement_updates(
        payload,
        {
            "height": {"customer_manual_cm": 172.0},
            "inseam": {"customer_manual_cm": 78.0},
            "sleeve": {"customer_manual_cm": 61.0},
            "neck": {"customer_manual_cm": 38.0},
        },
        updated_at="2026-05-27T00:01:00Z",
    )

    with pytest.raises(confirmation.CustomerConfirmationError, match="Measurements requiring manual confirmation"):
        confirmation.validate_customer_confirmation_payload(updated, require_complete=True)


def test_fit_preference_is_stored() -> None:
    payload = confirmation.build_customer_confirmation_payload(
        confirmation.sample_snapshot(created_at="2026-05-27T00:00:00Z"),
        fit_preference="relaxed",
        created_at="2026-05-27T00:00:00Z",
    )

    assert payload["fit_preference"] == "relaxed"
    assert {record["fit_preference"] for record in payload["confirmations"]} == {"relaxed"}


def test_ease_allowance_is_not_accepted_in_customer_payload() -> None:
    payload = _payload()
    payload["ease_cm"] = 4.0

    with pytest.raises(confirmation.CustomerConfirmationError, match="must not include maker ease/allowance"):
        confirmation.validate_customer_confirmation_payload(payload)

    clean_payload = _payload()
    with pytest.raises(confirmation.CustomerConfirmationError, match="must not include maker ease/allowance"):
        confirmation.apply_customer_measurement_updates(clean_payload, {"chest": {"customer_confirmed_cm": 101.0, "allowance_cm": 2.0}})


def test_confirmation_status_updates_correctly() -> None:
    payload = _payload()
    updated = confirmation.apply_customer_measurement_updates(
        payload,
        {"height": {"customer_manual_cm": 172.0}, "chest": {"customer_confirmed_cm": 101.0}},
        updated_at="2026-05-27T00:01:00Z",
    )
    by_target = {record["target"]: record for record in updated["confirmations"]}

    assert by_target["height"]["confirmation_status"] == "manual_value_provided"
    assert by_target["chest"]["confirmation_status"] == "customer_confirmed"
    assert by_target["height"]["updated_at"] == "2026-05-27T00:01:00Z"


def test_unrealistic_values_fail_validation() -> None:
    with pytest.raises(confirmation.CustomerConfirmationError, match="outside plausible range"):
        confirmation.apply_customer_measurement_updates(
            _payload(),
            {"height": {"customer_manual_cm": 12.0}},
            updated_at="2026-05-27T00:01:00Z",
        )


def test_synthetic_only_caveat_is_preserved() -> None:
    payload = _payload()

    assert payload["synthetic_calibrated_only"] is True
    assert payload["real_world_validated"] is False
    assert any("Synthetic-calibrated" in caveat for caveat in payload["caveats"])


def test_sample_artifacts_are_written(tmp_path: Path) -> None:
    result = confirmation.export_sample_customer_confirmation(tmp_path, created_at="2026-05-27T00:00:00Z")

    assert Path(result["sample_customer_confirmation_payload_json"]).exists()
    assert Path(result["customer_confirmation_summary_md"]).exists()
    output = json.loads(Path(result["sample_customer_confirmation_payload_json"]).read_text(encoding="utf-8"))
    assert "ease_cm" not in output
    assert "allowance_cm" not in output


def _payload() -> dict:
    return confirmation.build_customer_confirmation_payload(
        confirmation.sample_snapshot(created_at="2026-05-27T00:00:00Z"),
        fit_preference="regular",
        created_at="2026-05-27T00:00:00Z",
    )
