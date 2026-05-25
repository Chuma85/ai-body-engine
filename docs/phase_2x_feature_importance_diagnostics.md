# Phase 2X Feature Importance Diagnostics

Dataset: `data/synthetic/phase_2v`

Experiment: `artifacts/experiments/phase_2v_image_features`

Model: regular ridge image-feature baseline

Feature count: 195

Phase 2W hardest targets by percent MAE:

```text
weight_kg, waist_cm, neck_cm, shoulder_cm, calf_cm
```

## Command

```powershell
python -m training.experiments.analyze_feature_importance --experiment artifacts/experiments/phase_2v_image_features --dataset data/synthetic/phase_2v --output artifacts/analysis/phase_2x_feature_importance
```

The analyzer writes:

- `summary.json`
- `report.md`
- `per_target_top_features.csv`
- `feature_group_summary.csv`

## Hard Target Feature Groups

| Target | Most Important Feature Groups |
| --- | --- |
| weight_kg | cross-view ratios, band width profiles, height-normalized ratios |
| waist_cm | band width profiles, body band ratios, cross-view ratios |
| neck_cm | cross-view ratios, height-normalized ratios, band width profiles |
| shoulder_cm | band width profiles, height-normalized ratios, body band ratios |
| calf_cm | height-normalized ratios, cross-view ratios, band width profiles |

## Top Absolute Coefficient Features

| Target | Top Features |
| --- | --- |
| weight_kg | `side_shoulder_right_extent_ratio`, `side_hip_right_extent_ratio`, `front_to_side_bbox_width_ratio`, `front_upper_chest_left_extent_ratio`, `side_median_row_width_ratio` |
| waist_cm | `side_mid_torso_left_extent_ratio`, `side_max_row_width_ratio`, `side_median_row_width_ratio`, `side_mid_torso_width_ratio`, `side_mid_torso_right_extent_ratio` |
| neck_cm | `front_max_row_width_ratio`, `side_calf_right_extent_ratio`, `front_foreground_area_ratio`, `front_mean_row_width_ratio`, `side_calf_to_height_ratio` |
| shoulder_cm | `side_median_row_width_ratio`, `side_max_row_width_ratio`, `side_head_asymmetry_ratio`, `side_head_center_x_ratio`, `side_head_left_extent_ratio` |
| calf_cm | `front_max_row_width_ratio`, `side_max_row_width_ratio`, `side_hip_right_extent_ratio`, `front_neck_left_extent_ratio`, `front_neck_width_ratio` |

These coefficient rankings should be treated as ridge-model diagnostics, not causal explanations. Several top coefficients have weak direct feature-target correlations.

## Signal Warnings

Low-correlation warnings were emitted for every Phase 2W hard target:

| Target | Max Absolute Feature Correlation |
| --- | ---: |
| weight_kg | 0.104 |
| waist_cm | 0.135 |
| shoulder_cm | 0.102 |
| neck_cm | 0.052 |
| calf_cm | 0.114 |

This is the most important Phase 2X finding: the current silhouette feature set does not contain strong one-feature signal for the hardest measurements. Ridge is combining many weak signals rather than relying on clear target-specific evidence.

## Near-Constant Features

Near-constant features were detected, including fixed image dimensions and ankle/arm-span fields:

```text
front_image_width_px, front_image_height_px, front_arm_span_to_torso_ratio,
front_left_right_extension_delta, front_ankle_width_ratio,
front_ankle_left_extent_ratio, front_ankle_right_extent_ratio,
front_ankle_center_x_ratio, front_ankle_asymmetry_ratio,
side_image_width_px, ...
```

These features add little or no predictive signal in the current dataset and can be pruned or replaced in a future feature pass.

Repeated coefficient patterns were also detected for 17 groups, which is consistent with duplicated or highly redundant silhouette measurements.

## Global Feature Groups

| Feature Group | Feature Count | Mean Abs Coef | Max Abs Corr |
| --- | ---: | ---: | ---: |
| cross_view_ratio | 3 | 0.7753 | 0.1010 |
| height_normalized_ratio | 4 | 0.5753 | 0.1416 |
| band_width_profile | 32 | 0.5218 | 0.1487 |
| body_band_ratio | 8 | 0.4135 | 0.0904 |
| band_position_asymmetry | 104 | 0.3823 | 0.1471 |
| area | 8 | 0.2850 | 0.1397 |
| bbox_scale_position | 22 | 0.1864 | 0.1468 |
| arm_span_extension | 10 | 0.1619 | 0.1482 |

## Recommendation

Do not switch away from the regular ridge baseline yet. Phase 2X suggests the bottleneck is weak feature signal, not only model choice.

Next phase should improve or replace the silhouette feature set before introducing a deeper image model:

- remove near-constant and redundant features
- add stronger part-aware measurements for neck, shoulder, calf, and waist
- add more direct body-scale or volume proxy features for weight
- inspect whether controlled mesh deformation actually changes rendered silhouettes enough for labels
- consider landmark or segmentation-derived features as an intermediate step before CNN training

If those feature improvements still leave weak correlations, that would be a stronger signal to move toward a learned image representation.
