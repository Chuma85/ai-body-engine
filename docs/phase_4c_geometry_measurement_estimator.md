# Phase 4C Explainable Geometry Measurement Estimator

Phase 4C adds a transparent front/side geometry estimator for chest, waist, hip, and thigh. The goal is to compare an auditable measurement formula against the Phase 4A geometry-calibrated ML model.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Calibrated labels: `artifacts/phase_4a_geometry_calibrated_labels/calibrated_labels.csv`
- Phase 4A benchmark summary: `artifacts/phase_4a_geometry_calibrated_labels/calibrated_benchmark_results.json`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered, no larger dataset was generated, and no CNN was trained.

## Estimator

The estimator uses visible front/side geometry:

- target-specific vertical body band
- front band width
- side band depth
- height-derived scale factor
- ellipse/circumference proxy
- local front/side area proxy

It fits simple affine calibration coefficients on the train split only, then evaluates on validation and test. Each prediction row includes intermediate geometry values and quality flags.

The estimator is implemented in:

- `training/measurements/geometry_measurement_estimator.py`

## Artifacts

Local artifacts were written under `artifacts/phase_4c_geometry_measurement_estimator/`:

- `estimator_results.json`
- `estimator_results.csv`
- `per_target_estimator_results.csv`
- `calibration_coefficients.json`
- `estimator_vs_ml_summary.md`
- `failure_cases.csv`

These artifacts are generated locally and are not committed.

## Results

Direct estimator MAE:

| Split | MAE vs Calibrated Labels | MAE vs Original Formula Labels |
| --- | ---: | ---: |
| train | 4.1188 | 7.4928 |
| val | 4.0690 | 7.0250 |
| test | 4.3130 | 7.3844 |

Per-target test MAE against calibrated labels:

| Target | Test MAE | Gate |
| --- | ---: | --- |
| chest_cm | 6.5748 | research-only |
| waist_cm | 5.0903 | research-only |
| hip_cm | 3.4260 | assisted/manual-confirmation |
| thigh_cm | 2.1609 | stronger synthetic candidate |

Quality flags:

- `ok`: 1000 samples

## ML Comparison

| Approach | Test Group MAE |
| --- | ---: |
| Original formula-label ML baseline | 5.9333 |
| Direct geometry estimator | 4.3130 |
| Geometry-calibrated ML model | 1.6777 |

The direct estimator improves over the original formula-label baseline, but it does not match the calibrated-label GradientBoosting model. The ML model still adds meaningful residual correction beyond the simple ellipse/width/depth formula.

## Interpretation

The geometry estimator is useful because it is explainable:

- every estimate can expose front width, side depth, scale factor, ellipse proxy, and calibration coefficients,
- poor masks or unstable geometry can be flagged,
- predictions can be audited sample by sample.

However, chest and waist remain weak for the direct estimator. That suggests a single fixed-band ellipse approximation is too simple for upper-torso/waist geometry. The calibrated ML model likely benefits from nonlinear combinations of raw scale and shape features.

## Product Guidance

Do not claim production-ready tape accuracy.

Recommended behavior:

- Use the direct estimator as an auditable baseline and debugging tool.
- Use calibrated ML as a synthetic benchmark, not as real-world proof.
- Surface these measurements as AI estimates with confidence.
- Require manual confirmation before custom garment production.
- Do not use either estimator as sole cutting instruction until validated against real tape measurements.

## Recommendation

Next phase should combine transparency and residual correction:

- keep the explainable geometry estimator as the baseline,
- train residual models that predict correction from estimator output plus silhouette features,
- validate against renderer-side measured labels,
- eventually calibrate on real tape-measured examples.

The Phase 4C result says ML adds value, but the best path is probably an explainable geometry estimator plus a small learned residual, not an opaque end-to-end model alone.
