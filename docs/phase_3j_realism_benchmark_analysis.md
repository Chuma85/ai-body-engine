# Phase 3J Realism Benchmark Analysis

## Summary

Phase 3J analyzed why the Phase 3H realism-enabled dataset improved the current benchmark so much. The analysis used existing artifacts only:

- `artifacts/experiments/phase_2v_image_features`
- `artifacts/experiments/phase_3h_ridge_image_features`
- `artifacts/deep/phase_3h_dual_branch_augmented`

No images were generated and no models were trained in this phase.

## Overall Results

| Run | Train MAE | Val MAE | Test MAE | Train-Test Gap |
| --- | ---: | ---: | ---: | ---: |
| Phase 2V ridge | 9.0564 | 9.4635 | 9.4726 | 0.4163 |
| Phase 3H ridge | 6.1539 | 6.6519 | 6.8022 | 0.6483 |
| Phase 3H augmented dual CNN | 8.6169 | 8.7007 | 8.9501 | 0.3332 |

Phase 3H ridge improved over Phase 2V ridge by `2.6704` test MAE. That is about a `28.2%` relative improvement.

Phase 3H CNN improved over earlier CNN runs, but it still trails Phase 3H ridge by `2.1479` test MAE.

## Top Improved Targets

The largest Phase 2V ridge to Phase 3H ridge gains were:

| Target | Phase 2V Ridge | Phase 3H Ridge | Improvement | Improvement % |
| --- | ---: | ---: | ---: | ---: |
| waist_cm | 13.3796 | 6.1769 | 7.2028 | 53.8338 |
| chest_cm | 11.6222 | 5.5905 | 6.0316 | 51.8977 |
| height_cm | 14.3656 | 9.9378 | 4.4279 | 30.8226 |
| hip_cm | 10.4903 | 6.4411 | 4.0493 | 38.5999 |
| shoulder_cm | 6.5135 | 3.4405 | 3.0731 | 47.1797 |

Realism appears to have helped the silhouette/image-feature pipeline most on torso and upper-body measurements where lighting/background/material variation likely produced cleaner, more discriminative foreground geometry.

## Hardest Remaining Targets

The hardest Phase 3H ridge targets are:

| Target | Phase 3H Ridge Test MAE |
| --- | ---: |
| weight_kg | 13.6076 |
| height_cm | 9.9378 |
| inseam_cm | 7.2978 |
| sleeve_cm | 6.9328 |
| hip_cm | 6.4411 |

`inseam_cm` and `sleeve_cm` are notable because Phase 3H ridge slightly regressed versus Phase 2V ridge on those targets. They may need more explicit limb-length features, pose constraints, or label/render calibration.

## CNN Gap

The CNN underperforms ridge globally on the Phase 3H split:

- CNN trails ridge on 9 of 11 targets.
- CNN only beats or roughly matches ridge on `inseam_cm` and `sleeve_cm`.

Largest CNN gaps versus Phase 3H ridge:

| Target | Phase 3H Ridge | Phase 3H CNN | CNN Gap |
| --- | ---: | ---: | ---: |
| height_cm | 9.9378 | 14.9291 | 4.9913 |
| waist_cm | 6.1769 | 10.9143 | 4.7374 |
| chest_cm | 5.5905 | 10.1310 | 4.5405 |
| hip_cm | 6.4411 | 9.9782 | 3.5371 |
| thigh_cm | 6.1156 | 8.5730 | 2.4575 |

This suggests the current CNN is not extracting the strong geometric cues that the handcrafted silhouette features capture.

## Likely Explanation

Realism-enabled rendering likely helped because it changed the pixel distribution in a way that made the silhouette extractor more informative:

- darker/varied backgrounds improve foreground contrast
- lighting/material variation creates more stable body boundaries across profiles
- camera/scale jitter may reduce overfitting to one exact crop
- the 640x896 render setting still preserves enough body detail while changing feature sampling

There is one caveat from Phase 3H: the current renderer uses one RNG stream for both body parameters and render realism. The improvement is real for the Phase 3H dataset, but a strict ablation should separate body-parameter RNG from realism RNG to isolate the render-realism effect.

## Recommendation

Keep Phase 3H ridge as the current benchmark. The next best step is a small data/renderer refinement phase:

- separate body-parameter RNG from render-realism RNG
- regenerate a matched-control realism ablation if needed
- preserve the Phase 3H realism controls
- add targeted limb-length features or calibration checks for `inseam_cm` and `sleeve_cm`

For deep learning, do not scale CNN training blindly yet. The current CNN trails ridge broadly, so a future CNN phase should focus on stronger geometry-aware architecture/training rather than just more epochs.
