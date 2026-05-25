# Phase 2Z Current Best Baseline

Dataset: `data/synthetic/phase_2v`

Registry output: `artifacts/analysis/phase_2z_baseline_registry`

## Command

```powershell
python -m training.experiments.register_baseline_results --runs artifacts/experiments/phase_2v_image_features artifacts/experiments/phase_2w_target_tuned_ridge artifacts/experiments/phase_2y_geometry_features --output artifacts/analysis/phase_2z_baseline_registry
```

The registry writes:

- `summary.json`
- `report.md`

## Compared Runs

| Rank | Run | Model | Feature Version | Feature Count | Train MAE | Val MAE | Test MAE | Delta vs Best |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `phase_2v_image_features` | regular ridge | `phase_2p` | 195 | 9.0564 | 9.4635 | 9.4726 | 0.0000 |
| 2 | `phase_2w_target_tuned_ridge` | target-tuned ridge | `phase_2p` | 195 | 9.1331 | 9.3728 | 9.4875 | 0.0149 |
| 3 | `phase_2y_geometry_features` | regular ridge | `silhouette_geometry_v2` | 266 | 8.9090 | 9.5597 | 9.6182 | 0.1456 |

## Current Best Baseline

Current best: `phase_2v_image_features`

Current best test MAE: `9.4726`

This remains the selected baseline because it has the lowest same-dataset test MAE among the recent lightweight image-feature experiments. It also wins the most per-target comparisons: 6 of 11 targets.

## Why Phase 2W Is Not Selected

Phase 2W target-tuned ridge improved validation MAE and won 5 of 11 targets, but it slightly worsened test MAE:

```text
9.4726 -> 9.4875
```

The gap is small, but the default baseline should stay with the best test result unless a later same-dataset run clearly beats it.

## Why Phase 2Y Is Not Selected

Phase 2Y added advanced geometry features and increased the feature count from 195 to 266, but it worsened test MAE:

```text
9.4726 -> 9.6182
```

The Phase 2Y feature work is still useful diagnostic infrastructure, especially for feature importance and future pruning, but it should not replace the current benchmark. The lower train MAE with worse validation/test MAE suggests the added handcrafted geometry features introduced redundancy or overfit rather than stronger generalizable signal.

## Per-Target Winners

| Target | Best Run | Best Test MAE |
| --- | --- | ---: |
| height_cm | `phase_2w_target_tuned_ridge` | 14.3297 |
| weight_kg | `phase_2v_image_features` | 16.0824 |
| chest_cm | `phase_2v_image_features` | 11.6222 |
| waist_cm | `phase_2w_target_tuned_ridge` | 13.1803 |
| hip_cm | `phase_2v_image_features` | 10.4903 |
| shoulder_cm | `phase_2w_target_tuned_ridge` | 6.4396 |
| inseam_cm | `phase_2v_image_features` | 7.1232 |
| sleeve_cm | `phase_2v_image_features` | 6.0523 |
| neck_cm | `phase_2w_target_tuned_ridge` | 5.2628 |
| thigh_cm | `phase_2v_image_features` | 7.8219 |
| calf_cm | `phase_2w_target_tuned_ridge` | 5.4127 |

Target win counts:

| Run | Target Wins |
| --- | ---: |
| `phase_2v_image_features` | 6 |
| `phase_2w_target_tuned_ridge` | 5 |
| `phase_2y_geometry_features` | 0 |

## Known Hardest Targets

For the current best Phase 2V regular ridge baseline, the hardest targets by test MAE are:

| Target | Test MAE |
| --- | ---: |
| weight_kg | 16.0824 |
| height_cm | 14.3656 |
| waist_cm | 13.3796 |
| chest_cm | 11.6222 |
| hip_cm | 10.4903 |

These remain the main targets to watch in the next modeling pass.

## Recommendation

Keep `phase_2v_image_features` as the current benchmark and keep regular ridge as the default lightweight baseline.

Next phase should not promote target tuning or Phase 2Y geometry by default. A better next step is to prune near-constant/redundant features, inspect whether rendered silhouettes visibly encode the hard labels, or add a small learned image representation once the lightweight feature path is clearly saturated.
