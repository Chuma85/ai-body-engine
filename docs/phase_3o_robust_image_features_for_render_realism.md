# Phase 3O Robust Image Features For Render Realism

Phase 3O diagnosed render-realism sensitivity in the silhouette feature pipeline and added a more robust foreground mask extractor before starting any larger dataset or CNN phase.

## What Changed

The image feature extractor now uses `silhouette_geometry_v3`.

Main changes:

- RGB image loading for feature extraction.
- Border-estimated RGB background color.
- Foreground mask from color distance to the estimated background.
- Grayscale fallback retained for simple fixtures and edge cases.
- Mask sanity checks for:
  - missing foreground
  - tiny/partial foreground
  - over-thresholded masks
- Focused tests for bright/dark backgrounds, material brightness changes, small framing shifts, and invalid masks.

The goal was to reduce sensitivity to background brightness/color, lighting strength, and skin/material brightness while preserving geometry-derived features.

## Feature Drift Diagnostics

Added:

```text
training/experiments/analyze_feature_drift.py
```

The analyzer compares extracted features across same-body datasets and writes:

- `summary.json`
- `report.md`
- `feature_drift.csv`

Phase 3O feature drift was run on all six Phase 3N datasets using the new v3 extractor.

Largest observed drift by ablation:

| Ablation | Top Drift Features |
| --- | --- |
| `background_only` | `front_bbox_width_px` 0.0533, `side_bbox_width_px` 0.0467, `side_bbox_height_px` 0.0333 |
| `lighting_only` | `front_bbox_width_px` 0.0467, `side_bbox_width_px` 0.0433, `side_bbox_height_px` 0.0300 |
| `camera_jitter_only` | `side_bbox_height_px` 17.0233, `front_bbox_height_px` 16.5967, `front_bbox_width_px` 10.7100 |
| `skin_tone_only` | `side_bbox_width_px` 0.0567, `front_bbox_width_px` 0.0533, `side_bbox_height_px` 0.0367 |
| `combined_realism` | `side_bbox_height_px` 17.0167, `front_bbox_height_px` 16.6100, `front_bbox_width_px` 10.7000 |

Interpretation: background, lighting, and material brightness drift is now very small. Camera jitter remains the main source of feature drift because it intentionally changes framing and measured pixel geometry.

## Ridge Benchmark Rerun

The Phase 3N ridge artifacts were preserved. Phase 3O wrote new artifacts under `artifacts/experiments/phase_3o_*_ridge`.

| Ablation | Train MAE | Val MAE | Test MAE | Delta vs Clean | Effect |
| --- | ---: | ---: | ---: | ---: | --- |
| `clean_baseline` | 5.6346 | 7.5638 | 7.7216 | 0.0000 | matched |
| `background_only` | 5.6727 | 7.6671 | 7.8063 | 0.0847 | hurt |
| `lighting_only` | 5.6301 | 7.7002 | 7.8066 | 0.0850 | hurt |
| `camera_jitter_only` | 5.9731 | 7.3297 | 7.3958 | -0.3258 | helped |
| `skin_tone_only` | 5.6589 | 7.5924 | 8.1903 | 0.4687 | hurt |
| `combined_realism` | 6.0665 | 7.3902 | 7.5491 | -0.1725 | helped |

Per-target details are written to:

```text
artifacts/analysis/phase_3o_render_ablation/per_target_results.csv
```

For `combined_realism`, the largest target improvements versus clean were:

- `hip_cm`: -1.2518 MAE
- `chest_cm`: -0.9573 MAE
- `neck_cm`: -0.7354 MAE
- `waist_cm`: -0.6941 MAE
- `calf_cm`: -0.5999 MAE

Regressions remained on:

- `sleeve_cm`: +1.2920 MAE
- `thigh_cm`: +1.2336 MAE
- `height_cm`: +0.4416 MAE
- `weight_kg`: +0.2622 MAE

## Before And After

Phase 3N used the previous `silhouette_geometry_v2` extractor. Phase 3O used `silhouette_geometry_v3`.

| Ablation | Phase 3N Test MAE | Phase 3O Test MAE | Delta |
| --- | ---: | ---: | ---: |
| `clean_baseline` | 7.4657 | 7.7216 | +0.2559 |
| `background_only` | 7.6893 | 7.8063 | +0.1170 |
| `lighting_only` | 7.4462 | 7.8066 | +0.3604 |
| `camera_jitter_only` | 7.2621 | 7.3958 | +0.1337 |
| `skin_tone_only` | 7.5653 | 8.1903 | +0.6251 |
| `combined_realism` | 7.3070 | 7.5491 | +0.2422 |

Absolute 300-sample test MAE worsened with v3, but render-control sensitivity became clearer: background and lighting no longer cause large feature drift, while camera/framing remains the dominant source of geometry drift. Combined realism now improves over the v3 clean baseline, but it still does not beat the full 1000-sample Phase 3L clean benchmark.

## Current Best

The current official best remains:

```text
Phase 3L clean ridge test MAE: 6.5780
```

No Phase 3O 300-sample run beats this benchmark.

## Recommendation

Do not scale a broad realism dataset yet. The next phase should focus on camera/framing normalization:

- convert pixel-size bbox features into more height-normalized geometry features
- reduce dependence on raw `bbox_width_px` and `bbox_height_px`
- rerun the 300-sample ablation after removing or down-weighting raw pixel-scale features
- only then scale a promising camera-jitter or combined-realism variant

The v3 mask extractor is more robust to color and brightness changes, but the ridge feature set still carries raw framing-sensitive features that camera jitter can move substantially.
