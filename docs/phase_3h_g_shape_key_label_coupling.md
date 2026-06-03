# Phase 3H-G Shape-Key Label Coupling

Phase 3H-F showed that measurement labels in the 250-sample Blender dataset were weakly correlated with deterministic visual features. That means the rendered body changes and generated labels were not tightly coupled enough for useful model training.

Phase 3H-G updates the Blender dataset generation workflow so labels are derived from the same sampled shape-key values used during rendering.

## Label Generation Mode

```text
shape_key_coupled_synthetic
```

Formula version:

```text
shape_key_coupled_synthetic_v1
```

The formula is synthetic-only. It is a calibrated deterministic mapping from shape-key values to interpretable body factors and then to measurement labels. It is not real mesh circumference extraction and is not real-world validated.

## How Shape Keys Drive Labels

For each sample, the render script stores the actual shape-key values used for the body render. Those values are transformed into body factors:

- `height_factor`
- `chest_factor`
- `waist_factor`
- `hip_factor`
- `shoulder_factor`
- `inseam_factor`
- `torso_width_factor`
- `leg_length_factor`

The measurement labels are then generated from a base synthetic profile plus those factors. Related measurements share common signals, so label co-variation is more realistic:

- `height_cm` and `inseam_cm` share height and leg-length factors.
- `chest_cm`, `waist_cm`, `hip_cm`, and `shoulder_cm` share torso-width signal while retaining target-specific factors.

## New Label Traceability Fields

`labels.csv` keeps the existing required columns and adds:

```text
label_generation_mode
height_factor
chest_factor
waist_factor
hip_factor
shoulder_factor
inseam_factor
torso_width_factor
leg_length_factor
shape_key_values_json
body_shape_profile_id
```

`shape_key_values_json` is the per-sample trace of the rendered shape-key values.

## Metadata

`metadata.json` now includes:

- `label_generation_mode`
- `label_formula_version`
- `body_factor_definitions`
- `base_measurement_profile`
- `shape_key_to_factor_mapping`
- `measurement_formula_summary`
- `deterministic_seed`
- `synthetic_labels=true`
- `real_world_validated=false`

## Smoke Verification

Run:

```powershell
python scripts\verify_phase_3h_g_shape_key_label_coupling.py
```

This generates:

```text
data/synthetic/phase_3h_g_coupled_smoke
```

with 25 samples and seed 42, runs the strict Blender dataset audit, runs the Phase 3H-F visual correlation audit, and verifies strong factor-to-label correlations.

## Before Scaling

Before generating 1000+ samples, rerun the smoke verifier and inspect:

- factor-to-label correlations should be strong for all targets,
- visual correlations should improve versus Phase 3H-F,
- no target should depend on random labels independent of the rendered body,
- outputs must remain `synthetic_labels=true` and `real_world_validated=false`.

This phase improves synthetic label consistency. It still does not establish production tailoring accuracy.
