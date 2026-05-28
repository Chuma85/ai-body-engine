import json
from pathlib import Path

import pytest

from training.measurements import customer_measurement_confirmation as customer
from training.measurements import maker_measurement_review as maker
from training.measurements import production_measurement_package as package_mod


def test_package_builds_from_valid_snapshot_customer_maker_records() -> None:
    maker_review = _locked_maker_review()
    package = package_mod.build_production_package(maker_review, package_id="package_001", created_at="2026-05-27T00:00:00Z")

    assert package["package_id"] == "package_001"
    assert package["package_status"] == "locked_for_production"
    assert package["readiness_summary"]["ready"] is True
    assert package["targets"][0]["final_garment_cm"] is not None


def test_missing_maker_ease_blocks_ready_for_production() -> None:
    maker_review = _ready_maker_review()
    reviews = [dict(review) for review in maker_review["reviews"]]
    reviews[0]["maker_ease_allowance_cm"] = None
    reviews[0]["final_garment_cm"] = None
    maker_review = {**maker_review, "reviews": reviews, "locked_at": None, "locked_by_maker_id": None}
    package = package_mod.build_production_package(maker_review, package_id="package_001", created_at="2026-05-27T00:00:00Z")

    assert package["readiness_summary"]["ready"] is False
    assert any("missing maker_ease_allowance_cm" in blocker for blocker in package["readiness_summary"]["blockers"])
    with pytest.raises(package_mod.ProductionPackageError, match="not ready"):
        package_mod.validate_production_package(package, require_ready=True)


def test_low_confidence_ai_only_measurement_blocks_readiness() -> None:
    maker_review = _ready_maker_review()
    reviews = []
    for review in maker_review["reviews"]:
        next_review = dict(review)
        if next_review["target"] == "chest":
            next_review.update(
                {
                    "ai_confidence_tier": "low_confidence",
                    "selected_body_measurement_source": "ai_estimate_cm",
                    "selected_body_measurement_cm": 100.0,
                    "maker_ease_allowance_cm": 5.0,
                    "final_garment_cm": 105.0,
                }
            )
        reviews.append(next_review)
    package = package_mod.build_production_package({**maker_review, "reviews": reviews}, package_id="package_001")

    assert any("low-confidence AI-only" in blocker for blocker in package["readiness_summary"]["blockers"])


def test_manual_user_input_targets_cannot_use_ai_only_values() -> None:
    maker_review = _ready_maker_review()
    reviews = []
    for review in maker_review["reviews"]:
        next_review = dict(review)
        if next_review["target"] == "inseam":
            next_review.update(
                {
                    "selected_body_measurement_source": "ai_estimate_cm",
                    "selected_body_measurement_cm": 78.0,
                    "product_action": "require_manual_confirmation",
                    "final_garment_cm": 79.0,
                    "maker_ease_allowance_cm": 1.0,
                }
            )
        reviews.append(next_review)
    package = package_mod.build_production_package({**maker_review, "reviews": reviews}, package_id="package_001")

    assert any("cannot use AI-only value" in blocker for blocker in package["readiness_summary"]["blockers"])


def test_lock_behavior_prevents_double_lock_without_revision_path() -> None:
    package = package_mod.build_production_package(_ready_maker_review(), package_id="package_001")
    locked = package_mod.lock_production_package(package, maker_id="maker_001", locked_at="2026-05-27T00:03:00Z")

    with pytest.raises(package_mod.ProductionPackageError, match="already locked"):
        package_mod.lock_production_package(locked, maker_id="maker_001")

    revised = package_mod.request_package_revision(
        locked,
        requested_by_actor_id="maker_001",
        requested_by_role="maker",
        reason="Adjust chest ease before production.",
        changed_fields=["maker_ease_allowance_cm"],
        previous_values={"maker_ease_allowance_cm": 8.0},
        revised_values={"maker_ease_allowance_cm": 7.0},
        created_at="2026-05-27T00:04:00Z",
    )
    assert revised["package_status"] == "revision_requested"
    assert revised["locked_at"] is None


def test_audit_event_ids_are_attached() -> None:
    package = package_mod.build_production_package(_ready_maker_review(), package_id="package_001")
    locked = package_mod.lock_production_package(package, maker_id="maker_001", locked_at="2026-05-27T00:03:00Z")

    assert package["audit_event_ids"]
    assert any(event_id.startswith("package_locked__") for event_id in locked["audit_event_ids"])


def test_export_json_is_deterministic(tmp_path: Path) -> None:
    package = package_mod.build_production_package(_ready_maker_review(), package_id="package_001", created_at="2026-05-27T00:00:00Z")
    first = json.dumps(package, sort_keys=True)
    second = json.dumps(package, sort_keys=True)
    output = tmp_path / "package.json"
    package_mod.export_package_json(output, package)

    assert first == second
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(first)


def test_readiness_summary_lists_blockers_and_warnings() -> None:
    package = package_mod.build_production_package(_ready_maker_review(), package_id="package_001")
    summary = package_mod.summarize_package_readiness(package)

    assert summary["ready"] is True
    assert any("synthetic-calibrated" in warning for warning in summary["warnings"])


def test_sample_artifacts_are_written(tmp_path: Path) -> None:
    paths = package_mod.export_sample_production_package(tmp_path, created_at="2026-05-27T00:00:00Z")

    for path in paths.values():
        assert Path(path).exists()
    readiness = json.loads(Path(paths["sample_readiness_summary_json"]).read_text(encoding="utf-8"))
    assert readiness["ready"] is True


def _customer_payload() -> dict:
    payload = customer.build_customer_confirmation_payload(
        customer.sample_snapshot(created_at="2026-05-27T00:00:00Z"),
        created_at="2026-05-27T00:00:00Z",
    )
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


def _ready_maker_review() -> dict:
    payload = maker.build_maker_review_payload(
        _customer_payload(),
        maker_id="maker_001",
        customer_confirmation_id="confirmation_001",
        created_at="2026-05-27T00:00:00Z",
    )
    updates = {target: {"maker_ease_allowance_cm": 2.0} for target in customer.CONFIRMABLE_TARGETS}
    return maker.apply_maker_review_updates(payload, updates, updated_at="2026-05-27T00:02:00Z")


def _locked_maker_review() -> dict:
    ready = _ready_maker_review()
    return maker.lock_for_production(ready, maker_id="maker_001", locked_at="2026-05-27T00:03:00Z")
