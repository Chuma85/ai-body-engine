# Phase 4M Measurement Audit Trail

Phase 4M adds measurement audit events and revision history for the Body AI, customer confirmation, and maker review flow. It creates a durable record of meaningful measurement changes for accountability, admin review, production history, and dispute handling.

No model was trained, no data was generated, and no Blender render was run.

## Module

The audit layer lives in:

- `training/measurements/measurement_audit_trail.py`

Primary helpers:

- `create_audit_event(...)`
- `record_field_change(...)`
- `record_lock_event(...)`
- `record_revision_request(...)`
- `audit_events_for_revision(...)`
- `list_events_for_order(...)`
- `list_events_for_snapshot(...)`
- `list_events_for_maker_review(...)`
- `validate_audit_event(...)`
- `save_audit_events(...)`
- `load_audit_events(...)`
- `append_audit_event(...)`

## Why Audit Trails Are Needed

Measurement decisions affect customer fit, maker production, and dispute resolution. The audit trail records:

- AI snapshot creation,
- customer confirmations,
- customer manual measurement updates,
- fit preference updates,
- maker body verification,
- maker-only ease/allowance updates,
- final garment measurement calculations,
- production locks,
- revision requests,
- applied revisions,
- quality/admin review notes.

This makes it possible to answer who changed what, when it changed, what the previous value was, and why the change happened.

## Actor Roles

Supported actor roles are:

- `customer`
- `maker`
- `admin`
- `system`

User-driven changes require an `actor_id` and `actor_role`. System-generated events explicitly use `actor_role = system`.

Customers cannot create `maker_ease_allowance_updated` audit events. Ease and allowance remain maker-only.

## Revision History

Revision records include:

- `revision_id`
- `maker_review_id`
- `requested_by_actor_id`
- `requested_by_role`
- `reason`
- `changed_fields`
- `previous_values`
- `revised_values`
- `created_at`
- `approved_by_admin_id`
- `status`

Supported revision statuses are:

- `requested`
- `approved`
- `rejected`
- `applied`

Revision records preserve both previous and revised values so later audit or dispute review can see exactly what changed.

## Locked Production Protection

Locked production measurements cannot be changed silently.

Changing locked `final_garment_cm` or `maker_ease_allowance_cm` requires a revision reason. The audit helper records revision-request events and field-level revision events so the production history remains explicit.

## Local Persistence

Phase 4M includes local JSON persistence:

- `save_audit_events(...)`
- `load_audit_events(...)`
- `append_audit_event(...)`
- `list_audit_events(...)`

Loaded events are validated and returned in deterministic order by `created_at` and `event_id`.

## Local Artifacts

Sample artifacts were written under:

- `artifacts/phase_4m_measurement_audit_trail/sample_audit_events.json`
- `artifacts/phase_4m_measurement_audit_trail/sample_revision_history.json`
- `artifacts/phase_4m_measurement_audit_trail/audit_trail_summary.md`

These generated artifacts are local and are not committed.

## Workflow Connections

The audit trail connects the prior workflow layers:

- Phase 4I snapshots identify the persisted Body AI measurement record.
- Phase 4J customer confirmation events record customer body measurement and fit preference updates.
- Phase 4K maker review events record maker verification, maker-only ease/allowance, final garment calculation, and production locking.

Together, these records show how a synthetic-calibrated AI estimate became a customer-confirmed body measurement and then a maker-reviewed garment measurement.

## Production Caveat

This audit trail improves accountability, but it does not make the measurement model real-world validated. The upstream Body AI outputs remain synthetic-calibrated only until real tape-measured validation exists.
