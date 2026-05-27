# Phase 3X Measurement Band Diagnostics

Phase 3X focused on the four silhouette-learnable targets that remained weak after Phase 3W:

- `chest_cm`
- `waist_cm`
- `hip_cm`
- `thigh_cm`

The goal was to test whether better localized vertical measurement bands could improve these targets without generating new data or training a larger CNN.

## What Changed

Added a new additive band feature extractor:

- Module: `training/features/measurement_band_features.py`
- Version: `silhouette_geometry_v6_bands`

The v6 extractor samples multiple plausible vertical bands around each target zone and computes:

- front width at band
- side width/depth at band
- front/side ratio at band
- local area around band
- local contour slope around band
- normalized and raw variants where useful

Added an audit/benchmark script:

- `training/experiments/audit_measurement_bands.py`

Artifacts were written under:

- `artifacts/phase_3x_measurement_band_diagnostics/`

## Best Band Correlations

| Target | Best Band | Center Y | Best Feature | Role | Abs Corr |
| --- | --- | ---: | --- | --- | ---: |
| chest_cm | `chest_band_03_y40` | 0.40 | `chest_band_03_y40_norm_width_depth_product` | combined | 0.8514 |
| waist_cm | `waist_band_01_y46` | 0.46 | `waist_band_01_y46_side_norm_width_ratio` | side | 0.8238 |
| hip_cm | `hip_band_03_y68` | 0.68 | `hip_band_03_y68_norm_width_depth_product` | combined | 0.7477 |
| thigh_cm | `thigh_band_00_y68` | 0.68 | `thigh_band_00_y68_norm_width_depth_product` | combined | 0.6293 |

The best vertical locations are plausible:

- Chest signal appears lower in the chest/upper torso zone than the initial rough band.
- Waist signal is strongest in side depth near the waist zone.
- Hip and thigh both peak around the pelvis/upper-leg transition, suggesting the current synthetic geometry may not cleanly separate hip from upper-thigh silhouette.

## Benchmark Results

Benchmarks compared:

- `v5_existing`
- `v6_bands`
- `v5_plus_v6`

with:

- Ridge
- ElasticNet
- RandomForest
- GradientBoosting

| Run | Test Group MAE | Gate | Worst Target |
| --- | ---: | --- | --- |
| v6_bands + RandomForest | 6.1061 | research_only | hip_cm |
| v5_plus_v6 + RandomForest | 6.1120 | research_only | hip_cm |
| v5_existing + RandomForest | 6.1560 | research_only | hip_cm |
| v6_bands + ElasticNet | 6.1583 | research_only | hip_cm |
| v6_bands + Ridge | 6.2098 | research_only | hip_cm |
| v5_existing + GradientBoosting | 6.2413 | research_only | hip_cm |

The best Phase 3X four-target result was `v6_bands + RandomForest` at `6.1061` MAE. This did not beat the relevant Phase 3W target baselines, and no target moved below 5 cm.

## Per-Target Results

| Target | Best Phase 3X Run | Best Phase 3X MAE | Beats Phase 3W Target Baseline |
| --- | --- | ---: | --- |
| chest_cm | v6_bands + RandomForest | 5.6254 | false |
| waist_cm | v5_existing + GradientBoosting | 5.8929 | false |
| hip_cm | v6_bands + RandomForest | 6.4855 | false |
| thigh_cm | v5_existing + RandomForest | 6.0389 | false |

All four targets remain research-only.

## Interpretation

Phase 3X suggests that fixed vertical band localization is not the primary bottleneck.

Why:

- Band correlations are strong for chest, waist, and hip.
- The selected bands are anatomically plausible.
- Adding v6 band features did not improve prediction MAE.
- Combining v5 and v6 features also did not help.

This means the model can see some label-related silhouette signal, but the signal is not enough to produce assisted-range predictions. Likely bottlenecks are:

- synthetic label-to-geometry alignment is still noisy for torso/hip/thigh,
- hip and thigh geometry overlap too much in the current silhouette representation,
- raw scale/camera cues remain useful but unstable,
- the current renderer may not deform local body regions strongly or independently enough for the labels.

## Visual Diagnostics

Contact sheets were generated for low/mid/high:

- chest
- waist
- hip
- thigh

They are local artifacts under `artifacts/phase_3x_measurement_band_diagnostics/contact_sheets/` and are intentionally not committed.

## Recommendation

The next phase should inspect and improve synthetic geometry realism and label-to-mesh deformation for chest, waist, hip, and thigh rather than adding more band features.

Recommended next work:

- Add renderer-side measurement probes or mesh cross-section checks for chest/waist/hip/thigh.
- Verify that increasing a label visibly changes the corresponding body region while leaving unrelated regions reasonably stable.
- Consider adding explicit measurement-derived mesh annotations or landmarks for torso and upper-leg regions.
- Keep Phase 3W `raw_scale_camera + GradientBoosting` as the best silhouette-group candidate for now.
