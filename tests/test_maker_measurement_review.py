import json
from pathlib import Path

import pytest

from training.measurements import customer_measurement_confirmation as customer
from training.measurements import maker_measurement_review as maker


def test_maker_review_accepts_ease_allowance() -> None:
    payload = _maker_payload()
    updated = maker.apply_maker_review_updates(
        payload,
        {"chest": {"maker_ease_allowance_cm": 8.0}},
        updated_at="2026-05-27T00:02:00Z",
    )
    chest = _review(updated, "chest")

    assert chest["maker_ease_allowance_cm"] == 8.0
    assert chest["final_garment_cm"] == 109.0
    assert chest["selected_body_measurement_source"] == "customer_confirmed_cm"


def test_customer_payload_rejects_ease_allowance_but_maker_accepts_it() -> None:
    with pytest.raises(customer.CustomerConfirmationError, match="must not include maker ease/allowance"):
        customer.apply_customer_measurement_updates(_customer_payload(), {"chest": {"customer_confirmed_cm": 101.0, "ease_cm": 8.0}})

    updated = maker.apply_maker_review_updates(_maker_payload(), {"chest": {"maker_ease_allowance_cm": 8.0}})

    assert _review(updated, "chest")["maker_ease_allowance_cm"] == 8.0


def test_final_garment_calculation_uses_source_priority() -> None:
    payload = _maker_payload()
    updated = maker.apply_maker_review_updates(
        payload,
        {"chest": {"maker_verified_body_cm": 103.0, "maker_ease_allowance_cm": 7.0}},
        updated_at="2026-05-27T00:02:00Z",
    )
    chest = _review(updated, "chest")

    assert chest["selected_body_measurement_cm"] == 103.0
    assert chest["selected_body_measurement_source"] == "maker_verified_body_cm"
    assert chest["final_garment_cm"] == 110.0


def test_low_confidence_ai_only_measurement_cannot_lock() -> None:
    payload = _maker_payload(with_customer_values=False)
    reviews = []
    for review in payload["reviews"]:
        next_review = dict(review)
        if next_review["target"] == "chest":
            next_review.update(
                {
                    "ai_confidence_tier": "low_confidence",
                    "product_action": "accept_as_ai_estimate",
                    "ai_estimate_cm": 100.0,
                    "source": "ai_geometry_residual",
                    "selected_body_measurement_cm": 100.0,
                    "selected_body_measurement_source": "ai_estimate_cm",
                    "maker_ease_allowance_cm": 5.0,
                    "final_garment_cm": 105.0,
                }
            )
        reviews.append(next_review)
    payload = {**payload, "reviews": reviews}

    with pytest.raises(maker.MakerReviewError, match="Low-confidence AI-only chest"):
        maker.validate_maker_review_payload(payload, require_ready=True)


def test_user_input_or_landmark_targets_cannot_finalize_from_ai_alone() -> None:
    payload = _maker_payload()
    payload = maker.apply_maker_review_updates(
        payload,
        {target: {"maker_ease_allowance_cm": 2.0} for target in customer.CONFIRMABLE_TARGETS},
        updated_at="2026-05-27T00:02:00Z",
    )
    reviews = []
    for review in payload["reviews"]:
        next_review = dict(review)
        if next_review["target"] == "inseam":
            next_review.update(
                {
                    "ai_estimate_cm": 78.0,
                    "ai_confidence_tier": "high_confidence",
                    "product_action": "accept_as_ai_estimate",
                    "source": "landmark_required",
                    "selected_body_measurement_cm": 78.0,
                    "selected_body_measurement_source": "ai_estimate_cm",
                    "maker_ease_allowance_cm": 1.0,
                    "final_garment_cm": 79.0,
                }
            )
        reviews.append(next_review)
    payload = {**payload, "reviews": reviews}

    with pytest.raises(maker.MakerReviewError, match="inseam cannot be finalized from AI alone"):
        maker.validate_maker_review_payload(payload, require_ready=True)


def test_locked_review_cannot_be_edited_without_revision_path() -> None:
    locked = _locked_payload()

    with pytest.raises(maker.MakerReviewError, match="cannot be edited without an explicit revision path"):
        maker.apply_maker_review_updates(locked, {"chest": {"maker_ease_allowance_cm": 12.0}})

    revised = maker.apply_maker_review_updates(
        locked,
        {"chest": {"maker_ease_allowance_cm": 12.0}},
        allow_revision=True,
        updated_at="2026-05-27T00:04:00Z",
    )

    assert revised["locked_at"] is None
    assert _review(revised, "chest")["production_status"] == "revision_requested"


def test_maker_review_serialization_is_deterministic() -> None:
    payload = _maker_payload()

    assert json.dumps(payload, sort_keys=True) == json.dumps(payload, sort_keys=True)


def test_sample_artifacts_are_written(tmp_path: Path) -> None:
    result = maker.export_sample_maker_review(tmp_path, created_at="2026-05-27T00:00:00Z")

    assert Path(result["sample_maker_review_payload_json"]).exists()
    assert Path(result["sample_final_garment_measurements_json"]).exists()
    assert Path(result["maker_review_summary_md"]).exists()
    final = json.loads(Path(result["sample_final_garment_measurements_json"]).read_text(encoding="utf-8"))
    assert "measurements" in final


def _customer_payload(with_values: bool = True) -> dict:
    payload = customer.build_customer_confirmation_payload(
        customer.sample_snapshot(created_at="2026-05-27T00:00:00Z"),
        created_at="2026-05-27T00:00:00Z",
    )
    if not with_values:
        return payload
    return customer.apply_customer_measurement_updates(
        payload,
        {
            "height": {"customer_manual_cm": 172.0},
            "inseam": {"customer_manual_cm": 78.0},
            "sleeve": {"customer_manual_cm": 61.0},
            "neck": {"customer_manual_cm": 38.0},
            "chest": {"customer_confirmed_cm": 101.0},
            "waist": {"customer_confirmed_cm": 82.0},
            "hip": {"customer_confirmed_cm": 104.0},
            "thigh": {"customer_confirmed_cm": 62.0},
            "shoulder": {"customer_manual_cm": 45.0},
            "calf": {"customer_manual_cm": 39.0},
        },
        updated_at="2026-05-27T00:01:00Z",
    )


def _maker_payload(with_customer_values: bool = True) -> dict:
    return maker.build_maker_review_payload(
        _customer_payload(with_values=with_customer_values),
        maker_id="maker_001",
        customer_confirmation_id="confirmation_001",
        created_at="2026-05-27T00:00:00Z",
    )


def _locked_payload() -> dict:
    payload = _maker_payload()
    updates = {target: {"maker_ease_allowance_cm": 2.0} for target in customer.CONFIRMABLE_TARGETS}
    ready = maker.apply_maker_review_updates(payload, updates, updated_at="2026-05-27T00:02:00Z")
    return maker.lock_for_production(ready, maker_id="maker_001", locked_at="2026-05-27T00:03:00Z")


def _review(payload: dict, target: str) -> dict:
    return next(review for review in payload["reviews"] if review["target"] == target)
