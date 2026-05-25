# Phase 2AA Feature-Selected Ridge Baseline

Dataset: `data/synthetic/phase_2v`

Experiment: `artifacts/experiments/phase_2aa_feature_selected_ridge`

Model: per-target feature-selected ridge regression

Feature ranking method: absolute Pearson correlation on the train split only

Feature count grid: `10`, `25`, `50`, `100`, `all`

## Commands

```powershell
python -m training.experiments.run_feature_selected_ridge_experiment --dataset data/synthetic/phase_2v --output artifacts/experiments/phase_2aa_feature_selected_ridge
python -m training.analyze_baseline_errors --runs artifacts/experiments/phase_2v_image_features artifacts/experiments/phase_2aa_feature_selected_ridge --output artifacts/analysis/phase_2aa_comparison
python -m training.experiments.register_baseline_results --runs artifacts/experiments/phase_2v_image_features artifacts/experiments/phase_2w_target_tuned_ridge artifacts/experiments/phase_2y_geometry_features artifacts/experiments/phase_2aa_feature_selected_ridge --output artifacts/analysis/phase_2aa_registry
```

## Overall Result

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V regular ridge | 9.0564 | 9.4635 | 9.4726 |
| Phase 2AA feature-selected ridge | 9.2781 | 9.2727 | 9.5483 |

Phase 2AA improved validation MAE but worsened test MAE by `0.0756`, so it does not become the new best baseline.

## Selected Feature Counts

| Target | Selected Features |
| --- | ---: |
| height_cm | 266 |
| weight_kg | 10 |
| chest_cm | 10 |
| waist_cm | 25 |
| hip_cm | 10 |
| shoulder_cm | 100 |
| inseam_cm | 10 |
| sleeve_cm | 50 |
| neck_cm | 25 |
| thigh_cm | 10 |
| calf_cm | 100 |

The selected counts show that validation preferred smaller feature subsets for many targets, but `height_cm` still preferred all available features.

## Per-Target Test Comparison

| Target | Phase 2V | Phase 2AA | Result |
| --- | ---: | ---: | --- |
| height_cm | 14.3656 | 14.5833 | regressed |
| weight_kg | 16.0824 | 16.3977 | regressed |
| chest_cm | 11.6222 | 11.6406 | regressed |
| waist_cm | 13.3796 | 13.0179 | improved |
| hip_cm | 10.4903 | 10.9208 | regressed |
| shoulder_cm | 6.5135 | 6.4039 | improved |
| inseam_cm | 7.1232 | 7.1136 | improved |
| sleeve_cm | 6.0523 | 6.0826 | regressed |
| neck_cm | 5.3283 | 5.1900 | improved |
| thigh_cm | 7.8219 | 8.2180 | regressed |
| calf_cm | 5.4193 | 5.4624 | regressed |

Improved targets:

- `waist_cm`: +0.3617 MAE
- `neck_cm`: +0.1382 MAE
- `shoulder_cm`: +0.1096 MAE
- `inseam_cm`: +0.0096 MAE

Largest regressions:

- `hip_cm`: -0.4304 MAE
- `thigh_cm`: -0.3961 MAE
- `weight_kg`: -0.3153 MAE

## Registry Result

The optional Phase 2AA registry still selects `phase_2v_image_features` as current best:

| Rank | Run | Test MAE |
| --- | --- | ---: |
| 1 | `phase_2v_image_features` | 9.4726 |
| 2 | `phase_2w_target_tuned_ridge` | 9.4875 |
| 3 | `phase_2aa_feature_selected_ridge` | 9.5483 |
| 4 | `phase_2y_geometry_features` | 9.6182 |

Per-target wins in the registry:

| Run | Target Wins |
| --- | ---: |
| `phase_2v_image_features` | 5 |
| `phase_2w_target_tuned_ridge` | 2 |
| `phase_2aa_feature_selected_ridge` | 4 |
| `phase_2y_geometry_features` | 0 |

Phase 2AA is interesting because it wins four targets, but it does not win the overall benchmark.

## Recommendation

Keep Phase 2V regular ridge as the current best non-deep baseline.

Phase 2AA suggests feature selection can help target-specific measurements like `waist_cm`, `neck_cm`, and `shoulder_cm`, but it is not enough to improve overall test performance. This is a good stopping point for handcrafted non-deep baselines.

Next phase should move to a Phase 3A deep image model scaffold while keeping Phase 2V regular ridge as the benchmark to beat.
