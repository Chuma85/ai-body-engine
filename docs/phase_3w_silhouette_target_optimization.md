# Phase 3W Silhouette Target Optimization

Phase 3W trained and evaluated models only on the silhouette-learnable targets identified in Phase 3V:

- `chest_cm`
- `waist_cm`
- `hip_cm`
- `thigh_cm`
- `shoulder_cm`
- `calf_cm`

The weak-signal/manual targets were excluded from training and scoring in this phase:

- `height_cm`
- `weight_kg`
- `inseam_cm`
- `sleeve_cm`
- `neck_cm`

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Feature extractor: `silhouette_geometry_v5_hybrid`
- Baseline to beat: Phase 3V silhouette-learnable group MAE `5.3132`
- Output artifacts: `artifacts/phase_3w_silhouette_target_optimization/`

## Models Compared

The benchmark compared multi-output silhouette-group models and target-specific models for:

- `raw_scale_camera + Ridge`
- `raw_scale_camera + ElasticNet`
- `raw_scale_camera + RandomForest`
- `raw_scale_camera + GradientBoosting`
- `normalized_shape + Ridge`
- `combined_hybrid_without_area_ratios + RandomForest`
- `selected_low_drift_features + ElasticNet`
- existing Phase 3T dual-branch CNN metrics filtered to the six silhouette targets

## Results

| Run | Mode | Test Group MAE | Gate | Worst Target |
| --- | --- | ---: | --- | --- |
| raw_scale_camera + GradientBoosting | multi-output | 5.2379 | research_only | hip_cm |
| raw_scale_camera + GradientBoosting | target-specific | 5.2379 | research_only | hip_cm |
| raw_scale_camera + Ridge | multi-output | 5.3132 | research_only | hip_cm |
| raw_scale_camera + ElasticNet | multi-output | 5.3152 | research_only | hip_cm |
| combined_hybrid_without_area_ratios + RandomForest | multi-output | 5.3242 | research_only | hip_cm |
| raw_scale_camera + RandomForest | target-specific | 5.3354 | research_only | hip_cm |
| selected_low_drift_features + ElasticNet | multi-output | 5.4624 | research_only | hip_cm |
| normalized_shape + Ridge | multi-output | 5.6119 | research_only | hip_cm |
| Phase 3T dual-branch CNN, filtered | existing CNN | 6.9760 | research_only | waist_cm |

The best result was `raw_scale_camera + GradientBoosting` at group MAE `5.2379`. This beats the Phase 3V silhouette-group benchmark by `0.0753`, but it remains above the 5 cm research-only threshold.

Target-specific training did not materially improve over multi-output training for the current model families. For these sklearn-style models, the outputs are effectively target-independent internally, so the main value of this phase is target-group filtering and per-target model selection rather than a large target-specific gain.

## Best Per Target

| Target | Best Run | Test MAE | Gate |
| --- | --- | ---: | --- |
| chest_cm | raw_scale_camera + Ridge | 5.4918 | research_only |
| waist_cm | raw_scale_camera + GradientBoosting | 5.6725 | research_only |
| hip_cm | raw_scale_camera + GradientBoosting | 6.1704 | research_only |
| thigh_cm | raw_scale_camera + ElasticNet | 5.7530 | research_only |
| shoulder_cm | combined_hybrid_without_area_ratios + RandomForest | 3.1375 | assisted_manual_confirmation |
| calf_cm | combined_hybrid_without_area_ratios + RandomForest | 4.3017 | assisted_manual_confirmation |

No silhouette target reached the 1-3 cm stronger-candidate range. Shoulder and calf reached the 3-5 cm assisted/manual-confirmation range. Chest, waist, hip, and thigh remain research-only.

## Error Analysis

For the best group model, `raw_scale_camera + GradientBoosting`:

| Target | MAE | Bias | Small MAE | Large MAE |
| --- | ---: | ---: | ---: | ---: |
| chest_cm | 5.9950 | -0.4916 | 5.9627 | 5.8789 |
| waist_cm | 5.6725 | 0.1448 | 5.2661 | 5.5586 |
| hip_cm | 6.1704 | 0.0352 | 6.7647 | 6.8957 |
| thigh_cm | 5.8954 | 1.1521 | 6.6830 | 5.6325 |
| shoulder_cm | 3.2998 | 0.0952 | 4.3448 | 3.0341 |
| calf_cm | 4.3946 | 0.5946 | 4.5417 | 4.8161 |

Hip is the weakest silhouette target and remains the main drag on the group average. Shoulder and calf are the closest to practical assisted-measurement usefulness.

## Promotion Gates

Gates used in this phase:

- `>5 cm`: research-only
- `3-5 cm`: assisted/manual confirmation candidate
- `1-3 cm`: stronger production candidate, pending real-world validation

Phase 3W does not promote the silhouette model to assisted measurement as a group. It does identify two targets, shoulder and calf, as assisted/manual-confirmation candidates.

## Recommendation

The next phase should focus on improving chest, waist, hip, and thigh signal rather than training another all-target model. Good options:

- Add target-specific geometric features for hip/thigh/waist from side and front silhouettes.
- Add measurement-aware synthetic deformation checks so labels visibly alter the mesh at the correct body regions.
- Train or calibrate a hybrid strategy where shoulder/calf can be assisted outputs, while torso/hip/thigh remain research-only until they enter the 3-5 cm range.

The current best global benchmark remains Phase 3L clean Ridge, but the best silhouette-only candidate is now Phase 3W `raw_scale_camera + GradientBoosting` at group MAE `5.2379`.
