# Phase 3M Current Best Baseline

Phase 3M updates the project benchmark anchor after the controlled Phase 3L same-body clean vs realism experiment.

## Registry Result

The local baseline registry was refreshed with:

- `artifacts/experiments/phase_3l_clean_ridge`
- `artifacts/experiments/phase_3h_ridge_image_features`
- `artifacts/experiments/phase_3l_realism_ridge`
- `artifacts/deep/phase_3h_dual_branch_augmented`
- `artifacts/experiments/phase_2v_image_features`

Registry ranking by test MAE:

| Rank | Run | Test MAE | Delta vs Best |
| ---: | --- | ---: | ---: |
| 1 | Phase 3L clean ridge | 6.5780 | 0.0000 |
| 2 | Phase 3H ridge | 6.8022 | 0.2243 |
| 3 | Phase 3L realism ridge | 6.9717 | 0.3937 |
| 4 | Phase 3H augmented dual-branch CNN | 8.9501 | 2.3722 |
| 5 | Phase 2V ridge | 9.4726 | 2.8946 |

## Current Best

Previous official best:

- Phase 3H ridge image-feature baseline
- test MAE: 6.8022

New official best:

- Phase 3L clean ridge image-feature baseline
- dataset: `data/synthetic/phase_3l_clean`
- test MAE: 6.5780

Improvement over Phase 3H:

- 0.2243 MAE

## Why This Comparison Is Trustworthy

Phase 3L used the Phase 3K RNG isolation work to hold body generation fixed while changing render appearance. The clean and realism datasets had matching labels:

```text
MeasurementRowsMatch: True
MismatchCount: 0
```

That means the Phase 3L clean-vs-realism result is not explained by a hidden body-measurement distribution shift between those two datasets. The only intended difference was render appearance.

The same-body A/B result was:

- Clean ridge test MAE: 6.5780
- Realism ridge test MAE: 6.9717
- Clean beat realism by 0.3937 MAE

This shows that the current silhouette-feature ridge baseline benefits more from the cleaner Phase 3L render than from the realism controls used in the matched Phase 3L run.

## Recommended Baseline

Use `phase_3l_clean_ridge` as the current recommended benchmark for future comparisons:

```text
artifacts/experiments/phase_3l_clean_ridge
```

Future model or data experiments should compare against test MAE 6.5780 unless the registry is updated by a later phase.

## Recommendation

Next, run targeted ablations rather than broad realism changes. The most useful follow-up is to isolate which factor drove the Phase 3L clean improvement:

- clean 640 x 896 resolution versus older image sizing
- body seed/body distribution effects
- camera jitter versus fixed camera
- background and lighting variation separately

The current evidence suggests that controlled clean rendering plus stable body labels is the strongest baseline for the handcrafted silhouette feature model.
