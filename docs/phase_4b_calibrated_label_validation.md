# Phase 4B Calibrated Label Validation

Phase 4B validates the Phase 4A geometry-calibrated labels for chest, waist, hip, and thigh. The goal is to decide whether these labels are suitable as synthetic training targets while clearly separating synthetic benchmark success from real-world production readiness.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Samples: 1000
- Phase 4A artifacts: `artifacts/phase_4a_geometry_calibrated_labels/`
- Feature extractor: `silhouette_geometry_v5_hybrid`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered and no larger dataset was generated.

## Outputs

Local artifacts were written under `artifacts/phase_4b_calibrated_label_validation/`:

- `calibrated_label_validation.json`
- `calibrated_label_validation.csv`
- `calibrated_label_validation.md`
- `calibration_delta_summary.csv`
- `proxy_leakage_risk.md`
- `promotion_gate_summary.md`

These artifacts are generated locally and are not committed.

## Label Realism

The calibrated labels stayed inside broad plausible synthetic measurement ranges and remained monotonic against their best localized geometry proxies.

| Target | Min | Max | Mean | Std | Outliers | Best Geometry Corr | Monotonic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| chest_cm | 73.59 | 129.89 | 103.01 | 12.12 | 0 | 0.9732 | true |
| waist_cm | 50.24 | 123.78 | 84.57 | 14.62 | 0 | 0.9331 | true |
| hip_cm | 77.40 | 129.55 | 103.03 | 11.31 | 0 | 0.9397 | true |
| thigh_cm | 42.57 | 72.88 | 59.36 | 6.61 | 0 | 0.9077 | true |

This supports the Phase 4A finding that geometry calibration makes the synthetic labels much more consistent with the rendered silhouette.

## Calibration Deltas

| Target | Mean Abs Delta | P90 Abs Delta | Max Abs Delta | Ambiguous Minus Clean Abs Delta |
| --- | ---: | ---: | ---: | ---: |
| chest_cm | 5.5232 | 11.2279 | 22.1040 | 1.9917 |
| waist_cm | 6.0478 | 12.3605 | 26.1564 | 3.5914 |
| hip_cm | 6.8801 | 14.5250 | 29.8470 | 3.9497 |
| thigh_cm | 5.8316 | 11.5480 | 22.0163 | 2.0270 |

Corrections are larger on Phase 3Z ambiguous samples, especially for waist and hip. This reinforces the conclusion that formula-label collisions were limiting the model.

## Holdout Stability

Best calibrated-label run from Phase 4A:

- Run: `calibrated_labels__raw_scale_camera__target_specific__gradient_boosting`
- Train MAE: 1.1766
- Val MAE: 1.6936
- Test MAE: 1.6777
- Train/test gap: 0.5011

The split behavior is stable enough for a synthetic diagnostic. The train/test gap is not suspiciously tiny, but it is still a synthetic holdout using labels derived from geometry proxies.

## Proxy Leakage Risk

Risk level: high.

The calibrated labels are created from rendered geometry proxies. The strongest model also uses image features that encode similar geometry. Removing direct geometry-size features causes performance to collapse:

| Feature Config | Model | Feature Count | Train MAE | Val MAE | Test MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| raw_scale_camera | GradientBoosting | 31 | 1.1766 | 1.6936 | 1.6777 |
| raw_scale_camera | Ridge | 31 | 1.7302 | 1.6572 | 1.8101 |
| normalized_shape | GradientBoosting | 250 | 1.3554 | 2.0573 | 1.9037 |
| normalized_shape | Ridge | 250 | 1.7501 | 1.9806 | 1.9124 |
| raw_scale_camera_without_direct_proxies | Ridge | 14 | 5.9141 | 5.6173 | 6.3277 |
| raw_scale_camera_without_direct_proxies | GradientBoosting | 14 | 4.7967 | 5.6379 | 6.6041 |

Interpretation:

- The Phase 4A improvement is real for synthetic label consistency.
- It also has circularity risk: the model may be learning geometry-proxy formulas rather than a robust real-world measurement concept.
- Normalized-shape features still perform well, which is encouraging, but direct size/scale proxies are clearly carrying much of the signal.

## Promotion Gates

Synthetic gate:

- `synthetic_calibrated_strong_candidate`

Real-world gate:

- `requires_real_world_calibration_before_production`

Product behavior:

- Show chest, waist, hip, and thigh as AI estimates with confidence.
- Require manual confirmation before custom garment production.
- Do not use calibrated synthetic predictions as sole cutting instructions.

Gate definition:

- 1-3 cm on synthetic calibrated labels: strong synthetic candidate, not production proof.
- 3-5 cm: assisted/manual-confirmation candidate.
- >5 cm: research-only.

## Recommendation

Use geometry-calibrated labels as the official synthetic training target candidate for chest, waist, hip, and thigh, but do not claim production readiness.

The next phase should move geometry measurement into the renderer:

- compute mesh or silhouette measurement probes during rendering,
- save sampled formula labels and measured geometry labels side by side,
- train against measured labels for silhouette-learnable targets,
- validate against real-world or independently measured data before promotion.

The most important risk to manage next is circularity: a model trained on geometry-derived labels can look excellent on synthetic data while still needing real-world calibration and independent measurement validation.
