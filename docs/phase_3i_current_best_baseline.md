# Phase 3I Current Best Baseline

## Summary

Phase 3I updates the baseline registry interpretation after the Phase 3H realism-enabled dataset benchmark. The official current best benchmark is now the Phase 3H ridge image-feature baseline.

No new images were generated and no models were retrained in this phase.

## Registry Runs

Registry command:

```powershell
python -m training.experiments.register_baseline_results --runs artifacts/experiments/phase_2v_image_features artifacts/experiments/phase_3h_ridge_image_features artifacts/deep/phase_3h_dual_branch_augmented --output artifacts/analysis/phase_3i_baseline_registry
```

Ranked results:

| Rank | Run | Dataset | Model | Train MAE | Val MAE | Test MAE | Delta vs Best |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `phase_3h_ridge_image_features` | `data/synthetic/phase_3h` | ridge | 6.1539 | 6.6519 | 6.8022 | 0.0000 |
| 2 | `phase_3h_dual_branch_augmented` | `data/synthetic/phase_3h` | front_side_cnn | 8.6169 | 8.7007 | 8.9501 | 2.1479 |
| 3 | `phase_2v_image_features` | `data/synthetic/phase_2v` | ridge | 9.0564 | 9.4635 | 9.4726 | 2.6704 |

## Current Best

Previous current best:

- Run: `phase_2v_image_features`
- Dataset: `data/synthetic/phase_2v`
- Test MAE: `9.4726`

New current best:

- Run: `phase_3h_ridge_image_features`
- Dataset: `data/synthetic/phase_3h`
- Test MAE: `6.8022`

Improvement:

- Absolute MAE improvement: `2.6704`
- Relative improvement: about `28.2%`

## Why Phase 3H Won

The key change was realism-enabled rendering, not a new model family. Phase 3H used the same ridge image-feature baseline pattern, but the dataset came from Phase 3G render-realism controls:

- background brightness/color variation
- lighting strength variation
- safe camera jitter
- render resolution override
- skin-tone brightness variation

This appears to have made the silhouette/image-feature signal more useful. The ridge baseline improved on 9 of 11 measurement targets in the registry comparison.

The augmented dual-branch CNN also improved compared with earlier CNN runs, but it still trails Phase 3H ridge:

- Phase 3F augmented dual CNN test MAE: `9.5734`
- Phase 3H augmented dual CNN test MAE: `8.9501`
- Phase 3H ridge test MAE: `6.8022`

## Per-Target Notes

Phase 3H ridge is the per-target winner for:

- `height_cm`
- `weight_kg`
- `chest_cm`
- `waist_cm`
- `hip_cm`
- `shoulder_cm`
- `neck_cm`
- `thigh_cm`
- `calf_cm`

Phase 2V ridge still wins:

- `inseam_cm`
- `sleeve_cm`

Hardest current-best targets remain:

- `weight_kg`
- `height_cm`
- `inseam_cm`

## Recommendation

Use `artifacts/experiments/phase_3h_ridge_image_features` as the current recommended baseline for future comparisons. Future work should compare against Phase 3H ridge test MAE `6.8022`.

The next phase should either update any project-facing baseline registry docs to reference Phase 3H as current best, or run a stricter ablation with independent parameter RNG and render-realism RNG to isolate whether the gain comes from realism controls alone or from the new sampled dataset distribution.
