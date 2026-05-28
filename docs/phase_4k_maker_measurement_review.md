# Phase 4K Maker Measurement Review

Phase 4K adds a maker-facing measurement review workflow. It consumes the Phase 4J customer confirmation payload, lets makers apply maker-only ease/allowance, calculates final garment measurements, and supports production locking with an explicit revision path.

No model was trained, no data was generated, and no Blender render was run.

## Module

The workflow lives in:

- `training/measurements/maker_measurement_review.py`

Primary functions:

- `build_maker_review_payload(...)`
- `apply_maker_review_updates(...)`
- `lock_for_production(...)`
- `build_final_garment_measurements(...)`
- `validate_maker_review_payload(...)`

## Maker Responsibilities

Makers are responsible for:

- reviewing customer-confirmed body measurements,
- reviewing AI confidence and interval context,
- verifying measurements when needed,
- applying garment-specific ease/allowance,
- calculating final garment measurements,
- locking production measurements only when ready,
- requesting revisions when locked measurements must change.

## Body Measurement Versus Garment Measurement

Body measurements describe the customer. Garment measurements describe the item being made.

Phase 4K uses:

`final_garment_cm = selected_body_measurement_cm + maker_ease_allowance_cm`

The selected body measurement source follows this priority:

1. `maker_verified_body_cm`
2. `customer_confirmed_cm`
3. `customer_manual_cm`
4. `ai_estimate_cm`, only when the action permits an assisted AI estimate

The selected source is preserved as `selected_body_measurement_source`.

## Maker-Only Ease And Allowance

`maker_ease_allowance_cm` is accepted only in maker review payloads. Customer confirmation payloads still reject ease/allowance fields.

Ease and allowance are maker-only because they depend on:

- garment type,
- fabric and stretch,
- construction method,
- intended silhouette,
- fit preference,
- maker judgment.

Customers provide body measurements and fit preference. Makers convert those body measurements into garment measurements.

## Production Statuses

Supported statuses are:

- `draft`
- `awaiting_customer_confirmation`
- `awaiting_maker_review`
- `ready_for_production`
- `locked_for_production`
- `revision_requested`

Before production lock:

- `maker_id` is required.
- `maker_ease_allowance_cm` is required for every target.
- `final_garment_cm` must be positive.
- low-confidence AI-only measurements require maker verification or manual confirmation.
- `height`, `inseam`, `sleeve`, and `neck` cannot be finalized from AI alone.

After production lock, final garment values and ease cannot be edited without the explicit revision path.

## Local Artifacts

Sample artifacts were written under:

- `artifacts/phase_4k_maker_measurement_review/sample_maker_review_payload.json`
- `artifacts/phase_4k_maker_measurement_review/sample_final_garment_measurements.json`
- `artifacts/phase_4k_maker_measurement_review/maker_review_summary.md`

These generated artifacts are local and are not committed.

## Audit And Dispute Review

The maker review payload preserves:

- source body measurement,
- selected body measurement source,
- maker ease/allowance,
- final garment measurement,
- maker ID,
- lock timestamp,
- lock maker ID,
- status and notes.

This supports later dispute review by showing exactly how a customer body measurement became a garment measurement.

## Production Caveat

The upstream Body AI estimates remain synthetic-calibrated only and are not real-world production validation. Maker review is a product safety layer, not a substitute for real-world calibration or professional judgment.
