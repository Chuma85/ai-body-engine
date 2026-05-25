# Phase 2W Target Diagnostics And Tuning

Dataset: `data/synthetic/phase_2v`

Split: 800 train / 100 val / 100 test

Feature count: 195 front/side silhouette features

## Baseline

Phase 2V ridge baseline:

| Split | Overall MAE |
| --- | ---: |
| train | 9.0564 |
| val | 9.4635 |
| test | 9.4726 |

## Target Diagnostics

Diagnostics command:

```powershell
python -m training.experiments.analyze_target_diagnostics --experiment artifacts/experiments/phase_2v_image_features --output artifacts/analysis/phase_2w_target_diagnostics
```

Test diagnostics:

| Target | MAE | Mean True | MAE % | Bias | Under | Over | Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| height_cm | 14.3656 | 176.1480 | 8.1554 | 1.4356 | 42 | 58 | 0.1303 |
| weight_kg | 16.0824 | 81.1000 | 19.8303 | 2.8019 | 38 | 62 | 0.2350 |
| chest_cm | 11.6222 | 102.7730 | 11.3086 | 0.9761 | 47 | 53 | 0.2677 |
| waist_cm | 13.3796 | 83.8600 | 15.9547 | 1.1172 | 39 | 61 | 0.0786 |
| hip_cm | 10.4903 | 104.0900 | 10.0781 | -0.5530 | 47 | 53 | 0.3014 |
| shoulder_cm | 6.5135 | 48.4080 | 13.4554 | 0.8336 | 50 | 50 | -0.0851 |
| inseam_cm | 7.1232 | 79.2920 | 8.9836 | 0.8767 | 46 | 54 | 0.0683 |
| sleeve_cm | 6.0523 | 62.1870 | 9.7324 | 0.5368 | 50 | 50 | 0.0970 |
| neck_cm | 5.3283 | 39.5520 | 13.4715 | 0.2763 | 44 | 56 | -0.0153 |
| thigh_cm | 7.8219 | 60.2450 | 12.9836 | -0.4405 | 47 | 53 | 0.2954 |
| calf_cm | 5.4193 | 40.8540 | 13.2651 | 0.3841 | 44 | 56 | 0.1499 |

Easiest targets by test MAE percent: `height_cm`, `inseam_cm`, `sleeve_cm`, `hip_cm`, `chest_cm`.

Hardest targets by test MAE percent: `weight_kg`, `waist_cm`, `neck_cm`, `shoulder_cm`, `calf_cm`.

The correlation values are generally weak, which suggests the current silhouette features are not strongly ordering samples by true measurement value for several targets.

## Target-Tuned Ridge

Target-tuned command:

```powershell
python -m training.experiments.run_target_tuned_image_feature_experiment --dataset data/synthetic/phase_2v --output artifacts/experiments/phase_2w_target_tuned_ridge
```

Alpha grid: `0.1`, `1.0`, `10.0`, `30.0`, `100.0`

Selected alphas:

| Target | Alpha |
| --- | ---: |
| height_cm | 30.0 |
| weight_kg | 100.0 |
| chest_cm | 100.0 |
| waist_cm | 100.0 |
| hip_cm | 100.0 |
| shoulder_cm | 100.0 |
| inseam_cm | 100.0 |
| sleeve_cm | 100.0 |
| neck_cm | 100.0 |
| thigh_cm | 100.0 |
| calf_cm | 30.0 |

Target-tuned ridge result:

| Split | Overall MAE |
| --- | ---: |
| train | 9.1331 |
| val | 9.3728 |
| test | 9.4875 |

## Comparison

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V ridge | 9.0564 | 9.4635 | 9.4726 |
| Phase 2W target-tuned ridge | 9.1331 | 9.3728 | 9.4875 |

Target tuning improved validation MAE but slightly regressed test MAE by `0.0149`, so it is not a clear overall improvement.

Targets improved on test:

| Target | MAE Gain |
| --- | ---: |
| waist_cm | 0.1993 |
| shoulder_cm | 0.0739 |
| neck_cm | 0.0654 |
| height_cm | 0.0359 |
| calf_cm | 0.0066 |

Targets that regressed most on test:

| Target | Regression |
| --- | ---: |
| hip_cm | 0.1800 |
| thigh_cm | 0.1328 |
| weight_kg | 0.1155 |

## Recommendation

Keep the regular ridge baseline as the default benchmark. Target-specific alpha tuning is useful diagnostically, but it did not improve overall test MAE. The next phase should focus on feature/data alignment: inspect `worst_samples_test.csv`, improve scale and profile features for `weight_kg`, `height_cm`, and `waist_cm`, and consider richer pose or landmark signals before moving to a deeper image model.
