# Phase 2Y Advanced Silhouette Geometry Features

Dataset: `data/synthetic/phase_2v`

Baseline experiment: `artifacts/experiments/phase_2v_image_features`

Phase 2Y experiment: `artifacts/experiments/phase_2y_geometry_features`

Model: regular ridge image-feature baseline

## Feature Changes

Phase 2Y added deterministic silhouette geometry features focused on the hard targets from Phase 2W/2X:

- weight-oriented area and volume proxies
- waist minimum torso width/depth proxies and torso ratios
- shoulder peak, slope, and shoulder-to-body ratios
- neck/head transition and neck-to-shoulder ratios
- calf/lower-leg width, depth, and calf-to-thigh/ankle ratios

Feature extractor version: `silhouette_geometry_v2`

| Version | Feature Count |
| --- | ---: |
| Phase 2V / 2X | 195 |
| Phase 2Y | 266 |

## Commands

```powershell
python -m training.experiments.run_image_feature_experiment --dataset data/synthetic/phase_2v --output artifacts/experiments/phase_2y_geometry_features --model ridge
python -m training.experiments.analyze_feature_importance --experiment artifacts/experiments/phase_2y_geometry_features --dataset data/synthetic/phase_2v --output artifacts/analysis/phase_2y_feature_importance
python -m training.analyze_baseline_errors --runs artifacts/experiments/phase_2v_image_features artifacts/experiments/phase_2y_geometry_features --output artifacts/analysis/phase_2y_comparison
```

## Overall Result

| Experiment | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V ridge | 9.0564 | 9.4635 | 9.4726 |
| Phase 2Y geometry ridge | 8.9090 | 9.5597 | 9.6182 |

Phase 2Y worsened test MAE by `0.1456`. The lower train MAE with worse validation/test MAE suggests the extra geometry features added some overfit or redundant signal rather than stronger generalizable target signal.

## Per-Target Test MAE

| Target | Phase 2V | Phase 2Y | Change |
| --- | ---: | ---: | ---: |
| height_cm | 14.3656 | 14.5833 | -0.2177 |
| weight_kg | 16.0824 | 16.1897 | -0.1073 |
| chest_cm | 11.6222 | 11.8970 | -0.2748 |
| waist_cm | 13.3796 | 13.5408 | -0.1612 |
| hip_cm | 10.4903 | 10.5716 | -0.0813 |
| shoulder_cm | 6.5135 | 6.6317 | -0.1182 |
| inseam_cm | 7.1232 | 7.1335 | -0.0103 |
| sleeve_cm | 6.0523 | 6.1200 | -0.0677 |
| neck_cm | 5.3283 | 5.2932 | +0.0350 |
| thigh_cm | 7.8219 | 8.3536 | -0.5317 |
| calf_cm | 5.4193 | 5.4861 | -0.0668 |

Only `neck_cm` improved, and the gain was small. The new lower-leg features did not improve `calf_cm`; the new area/volume proxies did not improve `weight_kg`; the new torso features did not improve `waist_cm`.

## Hard Target Correlations

Phase 2X hard-target max absolute feature correlations:

| Target | Phase 2X |
| --- | ---: |
| weight_kg | 0.104 |
| waist_cm | 0.135 |
| shoulder_cm | 0.102 |
| neck_cm | 0.052 |
| calf_cm | 0.114 |

Phase 2Y hard-target max absolute feature correlations:

| Target | Phase 2Y |
| --- | ---: |
| weight_kg | 0.104 |
| waist_cm | 0.135 |
| shoulder_cm | 0.102 |
| neck_cm | 0.070 |
| calf_cm | 0.114 |

The geometry pass modestly improved the best direct correlation for `neck_cm`, but the other hard targets remained essentially unchanged. All hard targets still emitted low-correlation warnings below the `0.20` threshold.

## Feature Importance Notes

Top hard-target feature groups in the Phase 2Y ridge model:

| Target | Top Feature Groups |
| --- | --- |
| weight_kg | cross-view ratios, lower-leg geometry, integrated profiles |
| waist_cm | cross-view ratios, lower-leg geometry, shoulder geometry |
| neck_cm | cross-view ratios, integrated profiles, shoulder geometry |
| shoulder_cm | integrated profiles, height-normalized ratios, lower-leg geometry |
| calf_cm | area-scale ratios, lower-leg geometry, height-normalized ratios |

Near-constant features are still present, especially fixed image dimensions and ankle/calf-to-ankle fields. Repeated coefficient patterns increased to 26 groups, which is consistent with redundant geometry measurements in the current silhouette representation.

## Recommendation

Do not replace the Phase 2V regular ridge baseline with the Phase 2Y geometry feature set as the default benchmark. Keep the new extractor and tests as useful diagnostic infrastructure, but treat this result as evidence that simple handcrafted silhouette geometry is not enough for the remaining weak targets.

The next phase should focus on improving signal quality rather than adding more ratios:

- prune near-constant and duplicate features before rerunning ridge
- inspect whether label variation is visibly expressed in rendered silhouettes
- add landmark/keypoint or part-segmentation style measurements if staying lightweight
- consider a small learned image representation once feature pruning confirms handcrafted geometry is saturated
