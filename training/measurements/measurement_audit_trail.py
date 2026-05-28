from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from training.measurements.measurement_result_schema import write_json

DEFAULT_OUTPUT_DIR = "artifacts/phase_4m_measurement_audit_trail"
SAMPLE_AUDIT_EVENTS_JSON = "sample_audit_events.json"
SAMPLE_REVISION_HISTORY_JSON = "sample_revision_history.json"
AUDIT_SUMMARY_MD = "audit_trail_summary.md"


class AuditTrailError(ValueError):
    """Raised when audit events or revisions are invalid."""


class AuditEventType(str, Enum):
    AI_SNAPSHOT_CREATED = "ai_snapshot_created"
    CUSTOMER_MEASUREMENT_CONFIRMED = "customer_measurement_confirmed"
    CUSTOMER_MANUAL_MEASUREMENT_UPDATED = "customer_manual_measurement_updated"
    FIT_PREFERENCE_UPDATED = "fit_preference_updated"
    MAKER_BODY_MEASUREMENT_VERIFIED = "maker_body_measurement_verified"
    MAKER_EASE_ALLOWANCE_UPDATED = "maker_ease_allowance_updated"
    FINAL_GARMENT_MEASUREMENT_CALCULATED = "final_garment_measurement_calculated"
    PRODUCTION_MEASUREMENTS_LOCKED = "production_measurements_locked"
    PRODUCTION_MEASUREMENT_REVISION_REQUESTED = "production_measurement_revision_requested"
    PRODUCTION_MEASUREMENT_REVISED = "production_measurement_revised"
    QUALITY_FLAG_ADDED = "quality_flag_added"
    ADMIN_REVIEW_NOTE_ADDED = "admin_review_note_added"


class ActorRole(str, Enum):
    CUSTOMER = "customer"
    MAKER = "maker"
    ADMIN = "admin"
    SYSTEM = "system"


class RevisionStatus(str, Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass(frozen=True)
class MeasurementAuditEvent:
    event_id: str
    event_type: AuditEventType
    measurement_snapshot_id: str | None
    customer_confirmation_id: str | None
    maker_review_id: str | None
    order_id: str | None
    user_id: str | None
    actor_id: str
    actor_role: ActorRole
    target: str | None
    field_key: str
    old_value: Any
    new_value: Any
    reason: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=lambda: utc_now())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        payload["actor_role"] = self.actor_role.value
        return payload


@dataclass(frozen=True)
class MeasurementRevisionHistory:
    revision_id: str
    maker_review_id: str
    requested_by_actor_id: str
    requested_by_role: ActorRole
    reason: str
    changed_fields: list[str]
    previous_values: dict[str, Any]
    revised_values: dict[str, Any]
    created_at: str = field(default_factory=lambda: utc_now())
    approved_by_admin_id: str | None = None
    status: RevisionStatus = RevisionStatus.REQUESTED

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested_by_role"] = self.requested_by_role.value
        payload["status"] = self.status.value
        return payload


def create_audit_event(
    event_type: str | AuditEventType,
    actor_id: str,
    actor_role: str | ActorRole,
    field_key: str,
    old_value: Any = None,
    new_value: Any = None,
    measurement_snapshot_id: str | None = None,
    customer_confirmation_id: str | None = None,
    maker_review_id: str | None = None,
    order_id: str | None = None,
    user_id: str | None = None,
    target: str | None = None,
    reason: str | None = None,
    notes: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    event = MeasurementAuditEvent(
        event_id=event_id or f"audit_event_{uuid4().hex}",
        event_type=coerce_event_type(event_type),
        measurement_snapshot_id=measurement_snapshot_id,
        customer_confirmation_id=customer_confirmation_id,
        maker_review_id=maker_review_id,
        order_id=order_id,
        user_id=user_id,
        actor_id=actor_id,
        actor_role=coerce_actor_role(actor_role),
        target=target,
        field_key=field_key,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        notes=notes,
        created_at=created_at or utc_now(),
        metadata=metadata or {},
    ).to_payload()
    validate_audit_event(event)
    return event


def record_field_change(
    field_key: str,
    old_value: Any,
    new_value: Any,
    actor_id: str,
    actor_role: str | ActorRole,
    event_type: str | AuditEventType,
    locked: bool = False,
    reason: str | None = None,
    **context: Any,
) -> dict[str, Any]:
    if locked and field_key in {"final_garment_cm", "maker_ease_allowance_cm"} and not reason:
        raise AuditTrailError(f"Changing locked {field_key} requires a revision reason.")
    return create_audit_event(
        event_type=event_type,
        actor_id=actor_id,
        actor_role=actor_role,
        field_key=field_key,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        **context,
    )


def record_lock_event(
    maker_review_id: str,
    actor_id: str,
    order_id: str | None = None,
    measurement_snapshot_id: str | None = None,
    created_at: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    return create_audit_event(
        event_id=event_id,
        event_type=AuditEventType.PRODUCTION_MEASUREMENTS_LOCKED,
        actor_id=actor_id,
        actor_role=ActorRole.MAKER,
        field_key="production_status",
        old_value="ready_for_production",
        new_value="locked_for_production",
        maker_review_id=maker_review_id,
        order_id=order_id,
        measurement_snapshot_id=measurement_snapshot_id,
        created_at=created_at,
        notes="Production measurements locked by maker.",
    )


def record_revision_request(
    maker_review_id: str,
    requested_by_actor_id: str,
    requested_by_role: str | ActorRole,
    reason: str,
    changed_fields: list[str],
    previous_values: dict[str, Any],
    revised_values: dict[str, Any],
    created_at: str | None = None,
    approved_by_admin_id: str | None = None,
    status: str | RevisionStatus = RevisionStatus.REQUESTED,
    revision_id: str | None = None,
) -> dict[str, Any]:
    if not reason:
        raise AuditTrailError("Revision reason is required.")
    if not changed_fields:
        raise AuditTrailError("Revision changed_fields cannot be empty.")
    revision = MeasurementRevisionHistory(
        revision_id=revision_id or f"measurement_revision_{uuid4().hex}",
        maker_review_id=maker_review_id,
        requested_by_actor_id=requested_by_actor_id,
        requested_by_role=coerce_actor_role(requested_by_role),
        reason=reason,
        changed_fields=list(changed_fields),
        previous_values=dict(previous_values),
        revised_values=dict(revised_values),
        created_at=created_at or utc_now(),
        approved_by_admin_id=approved_by_admin_id,
        status=coerce_revision_status(status),
    ).to_payload()
    validate_revision_history(revision)
    return revision


def audit_events_for_revision(revision: dict[str, Any], order_id: str | None = None) -> list[dict[str, Any]]:
    validate_revision_history(revision)
    events = [
        create_audit_event(
            event_type=AuditEventType.PRODUCTION_MEASUREMENT_REVISION_REQUESTED,
            event_id=f"{revision['revision_id']}__requested",
            actor_id=revision["requested_by_actor_id"],
            actor_role=revision["requested_by_role"],
            field_key="revision_request",
            old_value=None,
            new_value=revision["revision_id"],
            maker_review_id=revision["maker_review_id"],
            order_id=order_id,
            reason=revision["reason"],
            created_at=revision["created_at"],
            metadata={"changed_fields": revision["changed_fields"]},
        )
    ]
    for field in revision["changed_fields"]:
        events.append(
            record_field_change(
                event_id=f"{revision['revision_id']}__{field}",
                field_key=field,
                old_value=revision["previous_values"].get(field),
                new_value=revision["revised_values"].get(field),
                actor_id=revision["requested_by_actor_id"],
                actor_role=revision["requested_by_role"],
                event_type=AuditEventType.PRODUCTION_MEASUREMENT_REVISED,
                locked=True,
                reason=revision["reason"],
                maker_review_id=revision["maker_review_id"],
                order_id=order_id,
                created_at=revision["created_at"],
                metadata={"revision_id": revision["revision_id"]},
            )
        )
    return events


def list_events_for_order(events: list[dict[str, Any]], order_id: str) -> list[dict[str, Any]]:
    return sort_events([event for event in events if event.get("order_id") == order_id])


def list_events_for_snapshot(events: list[dict[str, Any]], measurement_snapshot_id: str) -> list[dict[str, Any]]:
    return sort_events([event for event in events if event.get("measurement_snapshot_id") == measurement_snapshot_id])


def list_events_for_maker_review(events: list[dict[str, Any]], maker_review_id: str) -> list[dict[str, Any]]:
    return sort_events([event for event in events if event.get("maker_review_id") == maker_review_id])


def validate_audit_event(event: dict[str, Any]) -> None:
    required = {
        "event_id",
        "event_type",
        "actor_id",
        "actor_role",
        "field_key",
        "old_value",
        "new_value",
        "created_at",
        "metadata",
    }
    missing = sorted(required - set(event))
    if missing:
        raise AuditTrailError(f"Audit event is missing required fields: {', '.join(missing)}")
    event_type = coerce_event_type(event["event_type"])
    actor_role = coerce_actor_role(event["actor_role"])
    if not event["actor_id"]:
        raise AuditTrailError("actor_id is required for audit events.")
    if actor_role != ActorRole.SYSTEM and not event["actor_id"]:
        raise AuditTrailError("actor_id is required for user-driven audit events.")
    if actor_role == ActorRole.CUSTOMER and event_type == AuditEventType.MAKER_EASE_ALLOWANCE_UPDATED:
        raise AuditTrailError("Customer actors cannot create maker_ease_allowance_updated events.")
    if event["field_key"] in {"final_garment_cm", "maker_ease_allowance_cm"} and event_type == AuditEventType.PRODUCTION_MEASUREMENT_REVISED:
        if not event.get("reason"):
            raise AuditTrailError(f"Revision reason is required when changing locked {event['field_key']}.")


def validate_revision_history(revision: dict[str, Any]) -> None:
    required = {
        "revision_id",
        "maker_review_id",
        "requested_by_actor_id",
        "requested_by_role",
        "reason",
        "changed_fields",
        "previous_values",
        "revised_values",
        "created_at",
        "approved_by_admin_id",
        "status",
    }
    missing = sorted(required - set(revision))
    if missing:
        raise AuditTrailError(f"Revision history is missing required fields: {', '.join(missing)}")
    coerce_actor_role(revision["requested_by_role"])
    coerce_revision_status(revision["status"])
    if not revision["requested_by_actor_id"]:
        raise AuditTrailError("requested_by_actor_id is required.")
    if not revision["reason"]:
        raise AuditTrailError("Revision reason is required.")
    if not revision["changed_fields"]:
        raise AuditTrailError("Revision changed_fields cannot be empty.")


def save_audit_events(path: str | Path, events: list[dict[str, Any]]) -> None:
    for event in events:
        validate_audit_event(event)
    write_json(Path(path), sort_events(events))


def load_audit_events(path: str | Path) -> list[dict[str, Any]]:
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(f"Audit event file does not exist: {event_path}")
    try:
        events = json.loads(event_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditTrailError(f"Audit event file is not valid JSON: {event_path}") from exc
    if not isinstance(events, list):
        raise AuditTrailError("Audit event file must contain a list of events.")
    for event in events:
        validate_audit_event(event)
    return sort_events(events)


def append_audit_event(path: str | Path, event: dict[str, Any]) -> None:
    validate_audit_event(event)
    events = load_audit_events(path) if Path(path).exists() else []
    events.append(event)
    save_audit_events(path, events)


def list_audit_events(path: str | Path) -> list[dict[str, Any]]:
    return load_audit_events(path)


def save_revision_history(path: str | Path, revisions: list[dict[str, Any]]) -> None:
    for revision in revisions:
        validate_revision_history(revision)
    write_json(Path(path), sorted(revisions, key=lambda item: (item["created_at"], item["revision_id"])))


def sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda event: (str(event["created_at"]), str(event["event_id"])))


def coerce_event_type(value: str | AuditEventType) -> AuditEventType:
    if isinstance(value, AuditEventType):
        return value
    try:
        return AuditEventType(str(value))
    except ValueError as exc:
        raise AuditTrailError(f"Invalid audit event type: {value}") from exc


def coerce_actor_role(value: str | ActorRole) -> ActorRole:
    if isinstance(value, ActorRole):
        return value
    try:
        return ActorRole(str(value))
    except ValueError as exc:
        raise AuditTrailError(f"Invalid actor role: {value}") from exc


def coerce_revision_status(value: str | RevisionStatus) -> RevisionStatus:
    if isinstance(value, RevisionStatus):
        return value
    try:
        return RevisionStatus(str(value))
    except ValueError as exc:
        raise AuditTrailError(f"Invalid revision status: {value}") from exc


def export_sample_audit_trail(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    events, revisions = sample_audit_data()
    event_path = output_path / SAMPLE_AUDIT_EVENTS_JSON
    revision_path = output_path / SAMPLE_REVISION_HISTORY_JSON
    summary_path = output_path / AUDIT_SUMMARY_MD
    save_audit_events(event_path, events)
    save_revision_history(revision_path, revisions)
    summary_path.write_text(format_audit_summary(events, revisions), encoding="utf-8")
    return {
        "sample_audit_events_json": str(event_path),
        "sample_revision_history_json": str(revision_path),
        "audit_trail_summary_md": str(summary_path),
    }


def sample_audit_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    created = "2026-05-27T00:00:00Z"
    snapshot_id = "sample_snapshot_phase_4j"
    maker_review_id = "maker_review_sample_snapshot_phase_4j"
    order_id = "demo_order"
    events = [
        create_audit_event(
            event_id="audit_001",
            event_type=AuditEventType.AI_SNAPSHOT_CREATED,
            actor_id="body_ai_system",
            actor_role=ActorRole.SYSTEM,
            field_key="measurement_snapshot",
            old_value=None,
            new_value=snapshot_id,
            measurement_snapshot_id=snapshot_id,
            order_id=order_id,
            created_at=created,
        ),
        record_field_change(
            event_id="audit_002",
            event_type=AuditEventType.CUSTOMER_MEASUREMENT_CONFIRMED,
            actor_id="demo_user",
            actor_role=ActorRole.CUSTOMER,
            field_key="customer_confirmed_cm",
            old_value=None,
            new_value=105.2,
            measurement_snapshot_id=snapshot_id,
            order_id=order_id,
            user_id="demo_user",
            target="chest",
            created_at="2026-05-27T00:01:00Z",
        ),
        record_field_change(
            event_id="audit_003",
            event_type=AuditEventType.MAKER_EASE_ALLOWANCE_UPDATED,
            actor_id="demo_maker",
            actor_role=ActorRole.MAKER,
            field_key="maker_ease_allowance_cm",
            old_value=None,
            new_value=8.0,
            maker_review_id=maker_review_id,
            order_id=order_id,
            target="chest",
            created_at="2026-05-27T00:02:00Z",
        ),
        record_field_change(
            event_id="audit_004",
            event_type=AuditEventType.FINAL_GARMENT_MEASUREMENT_CALCULATED,
            actor_id="body_ai_system",
            actor_role=ActorRole.SYSTEM,
            field_key="final_garment_cm",
            old_value=None,
            new_value=113.2,
            maker_review_id=maker_review_id,
            order_id=order_id,
            target="chest",
            created_at="2026-05-27T00:02:30Z",
        ),
        record_lock_event(
            event_id="audit_005",
            maker_review_id=maker_review_id,
            actor_id="demo_maker",
            order_id=order_id,
            measurement_snapshot_id=snapshot_id,
            created_at="2026-05-27T00:03:00Z",
        ),
    ]
    revision = record_revision_request(
        revision_id="revision_001",
        maker_review_id=maker_review_id,
        requested_by_actor_id="demo_maker",
        requested_by_role=ActorRole.MAKER,
        reason="Customer requested chest ease adjustment before cutting.",
        changed_fields=["maker_ease_allowance_cm", "final_garment_cm"],
        previous_values={"maker_ease_allowance_cm": 8.0, "final_garment_cm": 113.2},
        revised_values={"maker_ease_allowance_cm": 7.0, "final_garment_cm": 112.2},
        created_at="2026-05-27T00:04:00Z",
    )
    events.extend(audit_events_for_revision(revision, order_id=order_id))
    return sort_events(events), [revision]


def format_audit_summary(events: list[dict[str, Any]], revisions: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Phase 4M Measurement Audit Trail",
            "",
            f"Audit events: `{len(events)}`",
            f"Revision records: `{len(revisions)}`",
            "",
            "Audit events record AI snapshot creation, customer confirmation, maker ease updates, final garment calculations, production locks, and revisions.",
            "Locked production measurements require a revision reason before changing maker ease/allowance or final garment values.",
            "",
        ]
    )


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export sample measurement audit trail artifacts.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = export_sample_audit_trail(args.output)
    print(f"Audit events: {paths['sample_audit_events_json']}")
    print(f"Revision history: {paths['sample_revision_history_json']}")
    print(f"Summary: {paths['audit_trail_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
