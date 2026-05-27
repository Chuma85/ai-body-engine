# Phase 3P Camera Framing Normalization

Phase 3P added canonical body-mask normalization to test whether silhouette features could become less sensitive to camera/framing jitter before generating any larger dataset.

## What Changed

The image feature extractor is now:

```text
silhouette_geometry_v4
```

After foreground detection, the extractor now:

- computes the body bounding box
- rejects empty, tiny, over-thresholded, or boundary-truncated masks
- crops to the body bounding box
- preserves aspect ratio
- scales body height to a canonical 220 px
- centers the body on a 256 x 256 canonical mask canvas
- extracts the existing geometry features from the normalized mask

The feature drift analyzer now supports raw-vs-normalized comparison with:

```text
--include-raw-comparison
```

This matters for mobile phone capture because a person can be shifted, padded, or slightly zoomed differently between photos. The feature pipeline should not treat small framing differences as body-shape changes.

## Feature Drift Result

Phase 3P reran feature drift on the six Phase 3N same-body ablations.

For camera jitter:

| Comparison | Top Drift Feature | Mean Abs Drift |
| --- | --- | ---: |
| Raw features | `side_bbox_height_px` | 17.0233 |
| Normalized features | `side_bbox_width_px` | 0.1800 |

For combined realism:

| Comparison | Top Drift Feature | Mean Abs Drift |
| --- | --- | ---: |
| Raw features | `side_bbox_height_px` | 17.0167 |
| Normalized features | `side_bbox_width_px` | 0.1800 |

The normalization did what it was designed to do: raw pixel-height drift from camera framing was largely removed. Remaining drift is mostly from width/ratio features after mask resampling and small segmentation differences.

## Ridge Benchmark

Phase 3P reran ridge experiments on the existing 300-sample Phase 3N datasets using the v4 extractor.

| Ablation | Train MAE | Val MAE | Test MAE | Delta vs Clean | Effect |
| --- | ---: | ---: | ---: | ---: | --- |
| `clean_baseline` | 5.6809 | 8.1442 | 7.9802 | 0.0000 | matched |
| `background_only` | 5.6467 | 7.7686 | 7.9221 | -0.0581 | helped |
| `lighting_only` | 5.5891 | 8.1408 | 7.8601 | -0.1200 | helped |
| `camera_jitter_only` | 5.6227 | 8.1102 | 8.0253 | +0.0451 | hurt |
| `skin_tone_only` | 5.6809 | 7.8671 | 7.9118 | -0.0683 | helped |
| `combined_realism` | 5.5155 | 7.9712 | 8.3309 | +0.3507 | hurt |

Best Phase 3P ablation:

```text
lighting_only test MAE: 7.8601
```

Worst Phase 3P ablation:

```text
combined_realism test MAE: 8.3309
```

No Phase 3P run beats the current best benchmark:

```text
Phase 3L clean ridge test MAE: 6.5780
```

## Phase 3O To Phase 3P

| Ablation | Phase 3O Test MAE | Phase 3P Test MAE | Delta |
| --- | ---: | ---: | ---: |
| `clean_baseline` | 7.7216 | 7.9802 | +0.2586 |
| `background_only` | 7.8063 | 7.9221 | +0.1158 |
| `lighting_only` | 7.8066 | 7.8601 | +0.0535 |
| `camera_jitter_only` | 7.3958 | 8.0253 | +0.6295 |
| `skin_tone_only` | 8.1903 | 7.9118 | -0.2785 |
| `combined_realism` | 7.5491 | 8.3309 | +0.7817 |

The important result is mixed: camera/framing feature drift improved, but predictive performance worsened for most ablations. Canonical height normalization appears to remove useful apparent-scale signal that the ridge baseline was using for targets such as height, weight, waist, and combined body-size proxies.

## Per-Target Notes

For `camera_jitter_only`, normalization helped:

- `height_cm`: -2.8596 MAE versus Phase 3P clean
- `inseam_cm`: -2.7123 MAE versus Phase 3P clean

But it hurt:

- `weight_kg`: +2.8365 MAE
- `waist_cm`: +2.1537 MAE
- `calf_cm`: +0.8196 MAE
- `neck_cm`: +0.4884 MAE

For `combined_realism`, normalization helped `height_cm` and `inseam_cm`, but hurt `weight_kg`, `waist_cm`, `calf_cm`, `neck_cm`, `shoulder_cm`, and `sleeve_cm`.

## Recommendation

Do not scale Phase 3P as-is. The next phase should keep the canonical mask path but make scale handling explicit:

- retain canonical normalized shape features for framing stability
- add separate scale/context features from the original image, such as raw body height ratio, raw body width ratio, padding ratios, and camera-normalized scale proxies
- avoid making `bbox_height_px` constant for all samples without replacing the lost body-scale signal
- compare a hybrid feature set against Phase 3O and Phase 3L before generating larger datasets

Phase 3P confirms that camera normalization is useful diagnostically, but full canonical height normalization alone is too aggressive for the current ridge baseline.
