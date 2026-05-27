# Phase 3Q Hybrid Scale-Aware Silhouette Features

Phase 3Q tested a hybrid feature extractor after Phase 3P showed that full canonical body-mask normalization reduced camera/framing drift but removed useful measurement signal.

## What Changed

The image feature extractor is now:

```text
silhouette_geometry_v5_hybrid
```

It keeps the Phase 3P normalized/canonical body-mask shape features and adds explicit raw scale/camera cues from the pre-normalized mask.

New raw scale/camera features are emitted for both front and side views:

- `raw_image_width_px`
- `raw_image_height_px`
- `raw_bbox_width_px`
- `raw_bbox_height_px`
- `raw_mask_area_px`
- `raw_bbox_aspect_ratio`
- `raw_bbox_width_ratio`
- `raw_bbox_height_ratio`
- `raw_mask_area_ratio`
- `normalization_scale_factor`
- `crop_offset_x`
- `crop_offset_y`
- `crop_offset_x_ratio`
- `crop_offset_y_ratio`

The existing cross-view bbox and area ratios now use raw pre-normalized scale cues, while the normalized shape and cross-view geometry proxies remain available.

Feature count changed from 266 in Phase 3P to 294 in Phase 3Q.

## Feature Drift Groups

The feature drift analyzer now groups features into:

| Group | Meaning |
| --- | --- |
| `normalized_shape` | Canonical mask geometry and band/profile features |
| `raw_scale_camera` | Raw bbox, mask area, crop offset, normalization scale, and raw front/side scale ratios |
| `combined_hybrid` | Cross-view normalized geometry/volume proxies |

On the Phase 3N same-body ablations, normalized-shape drift stayed small, but raw scale/camera drift was intentionally large when camera jitter changed the apparent body size and crop.

Camera jitter feature drift:

| Feature Group | Mean Abs Drift | Top Feature |
| --- | ---: | --- |
| `raw_scale_camera` | 216.3881 | `front_raw_mask_area_px` |
| `normalized_shape` | 0.0053 | `side_bbox_width_px` |
| `combined_hybrid` | 0.0005 | `front_side_waist_width_depth_proxy` |

Combined realism feature drift:

| Feature Group | Mean Abs Drift | Top Feature |
| --- | ---: | --- |
| `raw_scale_camera` | 216.1524 | `front_raw_mask_area_px` |
| `normalized_shape` | 0.0053 | `side_bbox_width_px` |
| `combined_hybrid` | 0.0005 | `front_side_waist_width_depth_proxy` |

Interpretation: Phase 3Q successfully separates stable normalized shape signal from volatile raw scale/framing signal, but the raw cue block remains sensitive to camera jitter.

## Ridge Benchmark

Phase 3Q reran ridge experiments on the existing 300-sample Phase 3N ablation datasets.

| Ablation | Train MAE | Val MAE | Test MAE | Delta vs Clean | Effect |
| --- | ---: | ---: | ---: | ---: | --- |
| `clean_baseline` | 5.5888 | 8.0850 | 8.0674 | 0.0000 | matched |
| `background_only` | 5.5513 | 7.7664 | 7.8957 | -0.1717 | helped |
| `lighting_only` | 5.5060 | 8.0924 | 7.8636 | -0.2038 | helped |
| `camera_jitter_only` | 5.4137 | 7.9167 | 8.3506 | +0.2832 | hurt |
| `skin_tone_only` | 5.5921 | 7.8284 | 7.9762 | -0.0912 | helped |
| `combined_realism` | 5.3133 | 7.8534 | 8.5426 | +0.4752 | hurt |

Best Phase 3Q ablation:

```text
lighting_only test MAE: 7.8636
```

Worst Phase 3Q ablation:

```text
combined_realism test MAE: 8.5426
```

## Comparison With Phase 3O And Phase 3P

| Ablation | Phase 3O Test MAE | Phase 3P Test MAE | Phase 3Q Test MAE | Q - P | Q - O |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean_baseline` | 7.7216 | 7.9802 | 8.0674 | +0.0872 | +0.3458 |
| `background_only` | 7.8063 | 7.9221 | 7.8957 | -0.0264 | +0.0894 |
| `lighting_only` | 7.8066 | 7.8601 | 7.8636 | +0.0034 | +0.0570 |
| `camera_jitter_only` | 7.3958 | 8.0253 | 8.3506 | +0.3253 | +0.9548 |
| `skin_tone_only` | 8.1903 | 7.9118 | 7.9762 | +0.0643 | -0.2142 |
| `combined_realism` | 7.5491 | 8.3309 | 8.5426 | +0.2118 | +0.9935 |

The hybrid cue block modestly helped background-only versus Phase 3P, and skin-tone-only remained better than Phase 3O, but the camera-jitter and combined-realism runs worsened.

## Per-Target Notes

For `camera_jitter_only`, Phase 3Q helped versus Phase 3Q clean on:

- `height_cm`: -1.4706 MAE
- `inseam_cm`: -3.0984 MAE
- `chest_cm`: -0.3393 MAE
- `sleeve_cm`: -0.0752 MAE

But it hurt substantially on:

- `weight_kg`: +3.3523 MAE
- `waist_cm`: +1.8469 MAE
- `hip_cm`: +0.5855 MAE
- `neck_cm`: +0.6263 MAE
- `thigh_cm`: +0.7695 MAE

For `combined_realism`, only `inseam_cm` improved versus clean. Most torso and mass-related targets regressed, led by `weight_kg`, `waist_cm`, `neck_cm`, and `shoulder_cm`.

## Current Best

No Phase 3Q run beats the current best benchmark:

```text
Phase 3L clean ridge test MAE: 6.5780
```

## Recommendation

Do not scale Phase 3Q as-is. The hybrid extractor proves that scale cues can be separated and audited, but raw pixel area/height cues are too sensitive to camera jitter when used directly by ridge regression.

The next phase should keep the v5 diagnostics but try a more constrained scale strategy:

- standardize raw scale cues by known render/camera metadata where available
- add ratio-only scale cues instead of large absolute pixel counts
- optionally remove or down-weight `raw_mask_area_px` and crop-offset pixel features
- compare a normalized-shape-only model against a curated scale-cue subset
- preserve Phase 3L clean ridge as the benchmark anchor until a full same-body run beats 6.5780 test MAE

Phase 3Q is useful infrastructure, but it is not a new best baseline.
