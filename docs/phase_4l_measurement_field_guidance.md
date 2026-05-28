# Phase 4L Measurement Field Guidance Metadata

Phase 4L adds reusable role-aware field guidance metadata for future FashionApp measurement screens. It does not build UI screens. It gives the future UI the labels, placeholders, helper text, tooltip text, warnings, validation hints, role visibility, and edit permissions needed to render measurement fields safely.

No model was trained, no data was generated, and no Blender render was run.

## Module

The guidance catalog lives in:

- `training/measurements/measurement_field_guidance.py`

Primary helpers:

- `get_field_guidance(field_key, role)`
- `list_guidance_for_role(role)`
- `validate_role_visibility(field_key, role)`
- `export_guidance_json(role)`
- `reject_customer_ease_fields(payload)`

## Why Guidance Metadata Matters

Measurement fields are easy to misunderstand. Every customer, maker, and admin-facing field needs an info icon or helper explanation so users know:

- whether a value is a body measurement or garment measurement,
- whether it is AI-estimated, manually entered, or system generated,
- whether it should be confirmed before production,
- whether a field is editable,
- what units and validation hints apply,
- what caveats should be shown.

## Customer Responsibilities

Customer guidance covers:

- `height_cm`
- `chest_cm`
- `waist_cm`
- `hip_cm`
- `thigh_cm`
- `shoulder_cm`
- `calf_cm`
- `inseam_cm`
- `sleeve_cm`
- `neck_cm`
- `fit_preference`
- `customer_notes`
- `customer_confirmed_cm`
- `customer_manual_cm`

Customer guidance repeatedly states that customer values are body measurements, not garment measurements. Customers should not add extra room, ease, allowance, or comfort margin.

`height_cm` is required because AI should not guess height from photos. `inseam_cm`, `sleeve_cm`, and `neck_cm` explain that tape/manual or landmark entry may be required. `thigh_cm` includes the Phase 4F undercoverage caution.

## Maker Responsibilities

Maker guidance covers:

- `maker_verified_body_cm`
- `maker_ease_allowance_cm`
- `final_garment_cm`
- `selected_body_measurement_source`
- `maker_notes`
- `production_status`
- `revision_reason`
- `locked_for_production`
- `locked_at`
- `locked_by_maker_id`

Maker guidance explains that `maker_verified_body_cm` is the maker-confirmed body measurement, while `maker_ease_allowance_cm` is extra room added for fit, design, comfort, fabric, and construction.

`final_garment_cm` is explained as:

`selected body measurement + maker ease/allowance`

## Body Measurement Versus Garment Measurement

Body measurements describe the customer. Garment measurements describe the item being made.

Customers provide body measurements and fit preference. Makers convert those body measurements into garment measurements. This preserves a clean audit trail and avoids mixing body data with garment construction decisions.

## Ease/Allowance Versus Seam Allowance

Ease/allowance is extra room added to a body measurement for fit, comfort, fabric, style, and garment design.

It is not seam allowance. Seam allowance is a construction margin added to pattern pieces for sewing. The guidance explicitly states that maker ease/allowance is maker-only and not seam allowance.

## Admin/Internal Guidance

Admin/internal guidance covers:

- `measurement_snapshot_id`
- `model_version`
- `pipeline_version`
- `calibration_version`
- `synthetic_calibrated_only`
- `real_world_validated`
- `quality_flags`
- `confidence_tier`
- `estimated_error_cm`
- `interval_low_cm`
- `interval_high_cm`
- `product_action`

These fields explain model and calibration lineage, uncertainty intervals, product-risk tiers, and synthetic-only validation caveats.

`synthetic_calibrated_only` guidance states that the result is validated on synthetic calibrated labels, not real tape measurements. `real_world_validated` guidance says it must remain false until real-world validation exists. `confidence_tier` is described as a product-risk/action tier, not a guarantee.

## Local Artifacts

Sample exports were written under:

- `artifacts/phase_4l_measurement_field_guidance/customer_field_guidance.json`
- `artifacts/phase_4l_measurement_field_guidance/maker_field_guidance.json`
- `artifacts/phase_4l_measurement_field_guidance/admin_field_guidance.json`
- `artifacts/phase_4l_measurement_field_guidance/measurement_field_guidance_summary.md`

These generated artifacts are local and are not committed.

## UI Rendering Recommendation

Future app UI should use this metadata to render:

- field label,
- unit,
- placeholder,
- helper text,
- info icon tooltip,
- warning text,
- required marker,
- validation hint,
- role-specific edit permissions.

Customer screens must not request maker ease/allowance. Maker screens may show maker-only ease/allowance and final garment calculations.
