# Phase 3T-B: Optional Back-View Synthetic Generation

## Purpose

Phase 3T-B goal: Add optional back-view synthetic generation to the synthetic body dataset pipeline.

- Front + side remains the minimum scan set.
- Front + side + back becomes the enhanced scan set.
- Back view supports future body morphology, back shoulder shape, waist/hip contour, posture/asymmetry, garment back-fit analysis, maker review, and mannequin/fitting quality work.
- Back view is optional and must not be required by current model, app, or training flows.
- This phase does not claim real-world accuracy improvement.

## Dataset Structure

Enhanced datasets may use:

```text
data/synthetic/phase_3t/
  images/
    front/
    side/
    back/
  labels/
    labels.csv
  manifest.csv
```

Minimum datasets may continue to omit `images/back/`.

## Naming Convention

All views use the same sample id:

```text
sample_000001_front.png
sample_000001_side.png
sample_000001_back.png
```

The back file is generated only when back view is enabled.

## Same-Sample Consistency

Front, side, and back images for the same `sample_id` must be generated from the same body sample.

The views must share:

- body parameters
- morphology and body shape profile
- pose variation
- skin tone
- lighting profile
- render seed stream
- label row

Back view must not be generated as an unrelated body variation.

## Metadata/Label Expectations

Generated labels and manifests should include:

- `front_image_path`
- `side_image_path`
- `back_image_path`
- `has_front`
- `has_side`
- `has_back`
- `capture_views`
- `minimum_scan_views=front,side`
- `enhanced_scan_views=front,side,back`

For minimum datasets:

- `has_back=false`
- `back_image_path` can be empty
- `capture_views=front,side`

For enhanced datasets:

- `has_back=true`
- `back_image_path` points to `images/back/{sample_id}_back.png`
- `capture_views=front,side,back`

## Model/Input Expectations

- Back view is optional.
- Current front/side training remains compatible.
- Pipelines should not fail when back view is missing.
- Future models can use back view as an additional channel or view.
- Enhanced pipelines can detect back availability through `has_back`, `back_image_path`, and `capture_views`.

## Beta/App Expectations

- Mobile scan can continue with front + side as the required capture set.
- Back view can be offered later as an optional enhanced scan.
- Maker review may use back-view summaries later for back-fit and shoulder-sensitive garments.
- No real-world accuracy claim should be made until the enhanced flow is validated against real capture and measurement outcomes.

## Implementation Notes

- The lightweight Python generator supports `include_back_view=True` and the CLI flag `--include-back-view`.
- The Blender renderer continues to require `front` and `side` in `views`.
- The Blender renderer supports `views: ["front", "side", "back"]` for enhanced generation.
- `synthetic/blender/configs/phase_3t_optional_back_view_config.example.json` is a small smoke config for explicit enhanced generation.
- `synthetic.validate_synthetic_dataset.validate_dataset(..., require_back=True)` validates enhanced datasets.
- Without `require_back=True`, missing back view remains a warning for legacy/minimum datasets.

## Safety

This phase changes synthetic generation and metadata only. It does not retrain models, does not make back view mandatory, does not commit generated image batches, and does not claim real-world accuracy improvement.
