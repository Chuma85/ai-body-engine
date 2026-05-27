# Phase 4G Measurement Result Contract

Phase 4G defines a stable Body AI measurement result contract for future FashionApp/mobile integration. It does not train a model, render new data, or claim real-world production readiness.

## Contract Module

The schema lives in:

- `training/measurements/measurement_result_schema.py`

It defines:

- `MeasurementResult`
- `MeasurementTargetResult`
- `MeasurementInterval`
- `MeasurementConfidence`
- `MeasurementQualityFlag`
- `MeasurementProductAction`
- `MeasurementModelMetadata`

The module also includes an exporter that builds a JSON-ready sample payload from the Phase 4D residual predictions plus Phase 4E/4F confidence and uncertainty policies.

## Supported Targets

The contract always emits targets in a stable order:

1. `chest`
2. `waist`
3. `hip`
4. `thigh`
5. `shoulder`
6. `calf`
7. `height`
8. `inseam`
9. `neck`
10. `sleeve`

Phase 4D currently provides AI geometry + residual predictions for `chest`, `waist`, `hip`, and `thigh`. Other targets are included so the app receives a complete result shape even when a measurement must be supplied manually or reviewed by a maker.

## Target Fields

Each target result includes:

- `target`
- `estimate_cm`
- `interval.low_cm`
- `interval.high_cm`
- `interval.estimated_error_cm`
- `confidence_tier`
- `product_action`
- `geometry_estimate_cm`
- `residual_correction_cm`
- `source`
- `quality_flags`
- `notes`

AI predictions require a confidence tier and a valid interval where:

`interval_low_cm <= estimate_cm <= interval_high_cm`

## Sources

Supported source values are:

- `ai_geometry_residual`
- `ai_geometry_only`
- `manual_user_input_required`
- `manual_maker_verified`
- `landmark_required`
- `unavailable`

In the current Phase 4G sample payload:

- `chest`, `waist`, `hip`, and `thigh` use `ai_geometry_residual`.
- `height` uses `manual_user_input_required`.
- `inseam`, `neck`, and `sleeve` use `landmark_required`.
- `shoulder` and `calf` are marked `unavailable` for this specific Phase 4D residual pipeline.

## Product Actions

Supported action values are:

- `accept_as_ai_estimate`
- `require_manual_confirmation`
- `request_retake_or_tape_measurement`
- `user_input_required`
- `maker_review_required`

These actions are product-risk actions, not a promise of production measurement accuracy. The app should display AI estimates as estimates, preserve the uncertainty interval, and require confirmation before custom garment production.

## Metadata

Every result includes model and calibration metadata:

- `model_version`
- `pipeline_version`
- `calibration_version`
- `training_dataset_id`
- `readiness_level`
- `synthetic_calibrated_only`
- `real_world_validated`
- `generated_at`

`real_world_validated` defaults to `false`. The current readiness level is `synthetic_calibrated_research`.

## Phase 4F Caveat

Phase 4F achieved 0.9025 overall test interval coverage on synthetic calibrated labels. However, `thigh_cm` remained undercovered at 0.8400. Phase 4G therefore adds a thigh calibration-risk note and `calibration_risk` quality flag for thigh AI predictions.

## Sample Artifacts

The exporter wrote local sample artifacts under:

- `artifacts/phase_4g_measurement_result_contract/sample_measurement_result.json`
- `artifacts/phase_4g_measurement_result_contract/measurement_schema_summary.md`

These generated artifacts are local and are not committed.

## FashionApp Integration Notes

The mobile app should consume the target-level payload and show:

- final estimate,
- uncertainty interval,
- confidence tier,
- product action,
- geometry/residual explanation fields where available,
- quality flags and caveats.

Maker-only ease, allowance, style fit, and garment construction adjustments should remain downstream of this body measurement contract. The body estimate should describe the person, not the garment.

## Recommendation

Next phase should wire this contract into a service boundary or API-style payload test so FashionApp can consume the same result shape without depending on training artifact internals. Real-world tape-measured validation is still required before using any AI output as a final production cutting measurement.
