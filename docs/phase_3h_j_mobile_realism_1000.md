# Phase 3H-J Mobile Realism Workflow Checkpoint

Phase 3H-J infrastructure is implemented, but the full Phase 3H-J dataset is not currently restored and this checkpoint must not be treated as a completed dataset phase.

This phase adds conservative mobile-camera realism controls on top of the Phase 3H-I safe-framed coupled Blender workflow. The code supports deterministic jitter for camera distance, camera height, body yaw, lighting, background tone, and phone framing while preserving synthetic label traceability.

This phase does not claim real-world validation. Generated labels remain synthetic, and `real_world_validated=false` is preserved.

## Current Dataset State

Expected final dataset path:

```text
data/synthetic/phase_3h_j_mobile_realism_1000
```

Expected final structure:

```text
labels.csv
metadata.json
images/front
images/side
images/back
```

Current state:

- Phase 3H-J infrastructure is implemented.
- The full dataset generation was interrupted.
- The expected final dataset path is currently missing: `data/synthetic/phase_3h_j_mobile_realism_1000`.
- `labels.csv`, `metadata.json`, and `images/front`, `images/side`, `images/back` are not currently present at the final dataset path.
- Prior artifact reports exist, but they are not enough to treat the dataset as restored.
- The next required step is a clean full regeneration or resumable generation that preserves the final merged dataset.

Generated dataset files are ignored/local and should not be committed.

## Prior Artifact Reports

The following reports exist from a prior successful run:

```text
artifacts/phase_3h_j_mobile_realism_1000_audit/audit_report.json
artifacts/phase_3h_j_mobile_realism_label_visual_correlation/correlation_report.json
artifacts/phase_3h_j_mobile_realism_blend_baseline/metrics.json
```

These reports are useful evidence for debugging and comparison, but they do not replace the missing final merged dataset. Phase 3H-J should not be marked complete until the final dataset path exists and can be validated in no-render mode.

Prior report metrics included:

- Reported labels: `1000`
- Reported PNG images: `3000`
- Strict audit: passed
- Audit warnings/errors/flagged: `0/0/0`
- Best benchmark model: `ridge`
- Overall mean MAE: `1.9239 cm`

## Realism Settings

Phase 3H-J keeps the Phase 3H-I coupled shape-key labels and safe framing, then adds deterministic per-sample/per-view mobile capture jitter:

```json
{
  "mobile_realism": true,
  "safe_framing_scale": 1.34,
  "distance_jitter": 0.015,
  "camera_height_jitter": 0.015,
  "body_rotation_jitter": 2.0,
  "lighting_jitter": 0.08,
  "background_jitter": 0.04,
  "phone_framing_jitter": 0.006
}
```

Metadata requirements for a future completed full dataset:

- `mobile_realism=true`
- `variation_source=shape_keys_safe_range_plus_mobile_realism`
- `label_generation_mode=shape_key_coupled_synthetic`
- `label_formula_version=shape_key_coupled_synthetic_v2_wide_safe_range`
- `synthetic_labels=true`
- `real_world_validated=false`

## Recovery Commands

Smoke:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --samples 25 --batch-size 25 --smoke --force
```

Full generation:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --force --batch-size 250
```

No-render validation after the final merged dataset exists:

```bash
python scripts/verify_phase_3h_j_mobile_realism_1000.py --no-render
```

`--no-render` and `--reuse-existing` never call Blender and never regenerate PNGs. They require the final merged dataset path to exist first; if the dataset is missing, the verifier fails clearly instead of attempting recovery.

## Recovery Requirements

Before Phase 3H-J can be treated as complete:

- The final path `data/synthetic/phase_3h_j_mobile_realism_1000` must exist.
- `labels.csv` must contain `1000` rows.
- `metadata.json` must exist.
- `images/front`, `images/side`, and `images/back` must exist.
- The dataset must contain `3000` PNGs.
- The clipping audit must report `0` clipped views.
- Strict audit, correlation, and benchmark must pass from the final merged dataset.
- `python scripts/verify_phase_3h_j_mobile_realism_1000.py --no-render` must pass.

## Benchmark Context From Prior Reports

Prior report correlation by target:

- `height_cm`: `0.3558` via `front_projection_column_height_std`
- `chest_cm`: `0.3564` via `mean_projection_column_height_mean`
- `waist_cm`: `0.2595` via `mean_raw_mask_area_ratio`
- `hip_cm`: `0.4305` via `back_crop_offset_y`
- `shoulder_cm`: `0.4995` via `front_neck_width_ratio`
- `inseam_cm`: `0.3630` via `front_torso_area_ratio`

Prior report MAE by target:

- `height_cm`: `3.2723`
- `chest_cm`: `1.7146`
- `waist_cm`: `1.9691`
- `hip_cm`: `1.7190`
- `shoulder_cm`: `0.9750`
- `inseam_cm`: `1.8934`

Prior report comparison to Phase 3H-I:

- Phase 3H-I overall mean MAE: `1.7486 cm`
- Phase 3H-J report overall mean MAE: `1.9239 cm`
- Delta: `+0.1753 cm`

These values should be refreshed after a clean restored or regenerated final dataset is validated.

## Limitations

- This is an infrastructure checkpoint, not a completed dataset checkpoint.
- The final merged full dataset is missing at the time of this commit.
- Existing reports alone are not sufficient to prove the current dataset is usable.
- This remains synthetic-only Blender data and is not real-world validated.
- Mobile realism is conservative: no distracting room objects, clothing, camera blur, device sensor noise, or real user capture artifacts yet.
