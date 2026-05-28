import json
from pathlib import Path

import pytest

from training.measurements import measurement_audit_trail as audit


def test_field_change_event_records_old_new_values() -> None:
    event = audit.record_field_change(
        event_id="event_001",
        event_type="customer_measurement_confirmed",
        actor_id="customer_001",
        actor_role="customer",
        field_key="customer_confirmed_cm",
        old_value=100.0,
        new_value=101.5,
        target="chest",
        order_id="order_001",
        created_at="2026-05-27T00:00:00Z",
    )

    assert event["old_value"] == 100.0
    assert event["new_value"] == 101.5
    assert event["event_type"] == "customer_measurement_confirmed"


def test_customer_cannot_create_maker_ease_event() -> None:
    with pytest.raises(audit.AuditTrailError, match="Customer actors cannot create"):
        audit.record_field_change(
            event_type="maker_ease_allowance_updated",
            actor_id="customer_001",
            actor_role="customer",
            field_key="maker_ease_allowance_cm",
            old_value=5.0,
            new_value=8.0,
        )


def test_locked_measurement_change_requires_revision_reason() -> None:
    with pytest.raises(audit.AuditTrailError, match="requires a revision reason"):
        audit.record_field_change(
            event_type="production_measurement_revised",
            actor_id="maker_001",
            actor_role="maker",
            field_key="final_garment_cm",
            old_value=113.2,
            new_value=112.2,
            locked=True,
        )


def test_revision_request_preserves_previous_and_revised_values() -> None:
    revision = audit.record_revision_request(
        revision_id="revision_001",
        maker_review_id="review_001",
        requested_by_actor_id="maker_001",
        requested_by_role="maker",
        reason="Customer requested a fit correction.",
        changed_fields=["maker_ease_allowance_cm", "final_garment_cm"],
        previous_values={"maker_ease_allowance_cm": 8.0, "final_garment_cm": 113.2},
        revised_values={"maker_ease_allowance_cm": 7.0, "final_garment_cm": 112.2},
        created_at="2026-05-27T00:04:00Z",
    )
    events = audit.audit_events_for_revision(revision, order_id="order_001")

    assert revision["previous_values"]["final_garment_cm"] == 113.2
    assert revision["revised_values"]["final_garment_cm"] == 112.2
    assert {event["event_type"] for event in events} == {
        "production_measurement_revision_requested",
        "production_measurement_revised",
    }


def test_list_events_by_order_returns_deterministic_ordering() -> None:
    events = [
        audit.create_audit_event(
            event_id="event_b",
            event_type="admin_review_note_added",
            actor_id="admin",
            actor_role="admin",
            field_key="notes",
            old_value=None,
            new_value="b",
            order_id="order_001",
            created_at="2026-05-27T00:00:02Z",
        ),
        audit.create_audit_event(
            event_id="event_a",
            event_type="admin_review_note_added",
            actor_id="admin",
            actor_role="admin",
            field_key="notes",
            old_value=None,
            new_value="a",
            order_id="order_001",
            created_at="2026-05-27T00:00:01Z",
        ),
    ]

    assert [event["event_id"] for event in audit.list_events_for_order(events, "order_001")] == ["event_a", "event_b"]


def test_system_event_uses_actor_role_system() -> None:
    event = audit.create_audit_event(
        event_id="system_event",
        event_type="ai_snapshot_created",
        actor_id="body_ai_system",
        actor_role="system",
        field_key="measurement_snapshot",
        old_value=None,
        new_value="snapshot_001",
    )

    assert event["actor_role"] == "system"
    audit.validate_audit_event(event)


def test_audit_event_serialization_is_deterministic() -> None:
    event = audit.create_audit_event(
        event_id="event_001",
        event_type="quality_flag_added",
        actor_id="system",
        actor_role="system",
        field_key="quality_flags",
        old_value=[],
        new_value=["calibration_risk"],
        created_at="2026-05-27T00:00:00Z",
    )

    assert json.dumps(event, sort_keys=True) == json.dumps(event, sort_keys=True)


def test_invalid_actor_role_fails_clearly() -> None:
    with pytest.raises(audit.AuditTrailError, match="Invalid actor role"):
        audit.create_audit_event(
            event_type="admin_review_note_added",
            actor_id="someone",
            actor_role="tailor",
            field_key="notes",
            old_value=None,
            new_value="note",
        )


def test_audit_event_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    event = audit.create_audit_event(
        event_id="event_001",
        event_type="admin_review_note_added",
        actor_id="admin",
        actor_role="admin",
        field_key="notes",
        old_value=None,
        new_value="note",
        created_at="2026-05-27T00:00:00Z",
    )
    audit.append_audit_event(path, event)

    assert audit.list_audit_events(path) == [event]


def test_sample_artifacts_are_written(tmp_path: Path) -> None:
    paths = audit.export_sample_audit_trail(tmp_path)

    for path in paths.values():
        assert Path(path).exists()
    events = json.loads(Path(paths["sample_audit_events_json"]).read_text(encoding="utf-8"))
    revisions = json.loads(Path(paths["sample_revision_history_json"]).read_text(encoding="utf-8"))
    assert events
    assert revisions
