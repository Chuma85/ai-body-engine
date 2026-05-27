# Phase 4H Body AI Inference Wrapper

Phase 4H adds a thin inference/service wrapper that produces the Phase 4G Body AI measurement result contract. This prepares the engine for later FashionApp/mobile integration without training a new model or claiming production readiness.

## Module

The wrapper lives in:

- `training/measurements/body_ai_inference.py`

Primary entry points:

- `BodyAIMeasurementService.predict(...)`
- `run_body_ai_measurement(...)`
- `to_dict(result)`
- `to_json(result)`
- `save_result_json(result, path)`

## Inputs

The wrapper accepts:

- `scan_id`
- `user_id` optional
- `order_id` optional
- `front_image_path`
- `side_image_path`
- `height_cm` optional
- `model_version` optional
- `pipeline_version` optional

Front and side images are required and must exist locally as PNG or JPEG files. Missing or invalid image paths fail clearly before any result is produced.

`height_cm` is optional. If it is missing, the result includes a `height` target with `user_input_required`. If it is provided, height is included in the payload as user-input-sourced rather than AI-estimated.

## Output

Output is a Phase 4G `MeasurementResult` payload with:

- target estimates where available,
- uncertainty intervals,
- confidence tiers,
- product actions,
- geometry estimate and residual correction fields,
- quality flags,
- model/calibration metadata,
- synthetic-only caveats.

The local sample artifact was written to:

- `artifacts/phase_4h_body_ai_inference_wrapper/sample_inference_result.json`
- `artifacts/phase_4h_body_ai_inference_wrapper/inference_wrapper_summary.md`

Generated artifacts are local and are not committed.

## Current Target Behavior

Current deterministic local/demo mode uses Phase 4D/4F residual prediction artifacts:

- `chest`, `waist`, `hip`, and `thigh`: `ai_geometry_residual`
- `height`: `manual_user_input_required`
- `inseam`, `neck`, and `sleeve`: `landmark_required`
- `shoulder` and `calf`: `unavailable` in the Phase 4D residual pipeline, with `maker_review_required`

`thigh` carries the Phase 4F calibration-risk note because synthetic interval coverage was 0.8400.

## Demo Mode

This is not a production model server yet. The wrapper currently runs in deterministic local/demo mode using existing synthetic calibrated-label artifacts:

- Phase 4D residual predictions
- Phase 4E confidence policy
- Phase 4F uncertainty intervals
- Phase 4G result schema

This lets the app team integrate against the stable contract while the real inference backend is still being packaged.

Example:

```powershell
python -m training.measurements.body_ai_inference --scan-id sample_000007 --front-image data/synthetic/phase_3t/images/front/sample_000001_front.png --side-image data/synthetic/phase_3t/images/side/sample_000001_side.png --height-cm 172 --output artifacts/phase_4h_body_ai_inference_wrapper
```

## Production Caveats

The current best geometry + residual pipeline is synthetic-calibrated only:

- Test MAE: 1.6422 cm against synthetic calibrated labels
- Overall synthetic interval coverage: 0.9025
- `real_world_validated`: `false`
- `synthetic_calibrated_only`: `true`

This is not proof of real-world tape-measure accuracy. The payload should be shown as an AI estimate with uncertainty and appropriate manual confirmation actions.

## FashionApp Integration Recommendation

The next integration phase should expose this wrapper through a small API boundary that accepts uploaded front/side images and optional user height, then returns the exact Phase 4G payload. FashionApp should not depend on training artifact CSVs directly.

Maker-only ease, style allowance, and final garment decisions should remain downstream of this body measurement result.
