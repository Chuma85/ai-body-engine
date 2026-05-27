# Phase 4I Measurement Snapshot Persistence

Phase 4I adds local JSON snapshot persistence for Body AI measurement results. This turns a Phase 4H inference result into a durable record that can be saved, loaded, listed, validated, and later mapped into FashionApp customer/order workflows.

No model was trained, no data was generated, and no Blender render was run.

## Module

The snapshot store lives in:

- `training/measurements/measurement_snapshot_store.py`

Primary functions:

- `create_snapshot(...)`
- `save_snapshot(...)`
- `load_snapshot(...)`
- `list_snapshots(...)`
- `validate_snapshot(...)`
- `validate_measurement_result_payload(...)`

## Snapshot Contents

Each snapshot stores:

- `snapshot_id`
- `scan_id`
- `user_id`
- `order_id`
- `front_image_path`
- `side_image_path`
- `height_cm`
- `measurement_result`
- `model_version`
- `pipeline_version`
- `calibration_version`
- `synthetic_calibrated_only`
- `real_world_validated`
- `created_at`
- `updated_at`

The nested `measurement_result` is the Phase 4G result contract produced by the Phase 4H inference wrapper.

## Guardrails

The store validates snapshots before writing and after loading:

- `scan_id` is required.
- `measurement_result` is required.
- target order must match the Phase 4G contract.
- target estimates must sit inside their intervals when present.
- snapshot metadata must match measurement-result metadata.
- `synthetic_calibrated_only` must remain `true`.
- `real_world_validated` must remain `false`.
- corrupt JSON fails clearly.

This keeps the local snapshot format honest while the model remains synthetic-calibrated only.

## Local Artifacts

Sample artifacts were written under:

- `artifacts/phase_4i_measurement_snapshots/sample_snapshot.json`
- `artifacts/phase_4i_measurement_snapshots/snapshot_store_summary.md`

These generated artifacts are local and are not committed.

## FashionApp Mapping

Later, FashionApp can map this JSON shape into database records such as:

- measurement snapshots by `scan_id`
- customer measurement history by `user_id`
- order-specific measurement snapshots by `order_id`
- target-level review states for maker/customer confirmation
- audit events for AI estimates, manual overrides, and maker verification

The snapshot separates body measurements from garment construction. Maker-only ease, allowance, style preference, and fit decisions should be stored downstream as garment/order decisions, not as replacements for the body measurement record.

## Review Workflows

Snapshots support:

- customer review of AI estimates and uncertainty intervals,
- maker review of target-level product actions,
- manual confirmation for medium/low confidence targets,
- retake or tape-measure requests when required,
- dispute/audit trails showing the model version, calibration version, inputs, and result payload used at the time.

## Production Caveat

These snapshots are not real-world production measurement proof. They preserve the existing Phase 4G/4H caveats:

- synthetic-calibrated labels only,
- no tape-measured human validation yet,
- manual confirmation required before custom garment cutting decisions.

## Recommendation

Next phase should define the API or database persistence boundary that saves these snapshots as first-class records. The same validation rules should run before any snapshot is accepted into a FashionApp customer or order workflow.
