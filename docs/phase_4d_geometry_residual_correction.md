# Phase 4D Geometry Residual Correction

Phase 4D adds a hybrid measurement pipeline:

1. direct explainable geometry estimator,
2. learned target-specific residual correction,
3. final estimate = geometry estimate + predicted residual.

The goal is to keep measurement predictions auditable while testing whether a small learned residual can close the gap to the Phase 4A calibrated-label ML model.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Calibrated labels: `artifacts/phase_4a_geometry_calibrated_labels/calibrated_labels.csv`
- Phase 4A benchmark: `artifacts/phase_4a_geometry_calibrated_labels/calibrated_benchmark_results.json`
- Geometry estimator: `training/measurements/geometry_measurement_estimator.py`
- Residual correction module: `training/measurements/residual_correction.py`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered, no larger dataset was generated, and no CNN was trained.

## Method

For each target:

- compute `geometry_estimate_cm` from the Phase 4C geometry estimator,
- compute `residual_cm = calibrated_label_cm - geometry_estimate_cm`,
- train residual models on the train split only,
- predict `predicted_residual_cm` on train/val/test,
- compute `final_estimate_cm = geometry_estimate_cm + predicted_residual_cm`.

Models compared:

- Ridge
- ElasticNet
- RandomForest
- GradientBoosting

Each prediction row records:

- geometry estimate,
- calibrated label,
- true residual,
- predicted residual,
- final estimate,
- absolute error,
- model name,
- geometry quality flags,
- confidence flags.

## Artifacts

Local artifacts were written under `artifacts/phase_4d_residual_correction/`:

- `residual_training_summary.json`
- `residual_training_summary.csv`
- `residual_benchmark_results.json`
- `residual_benchmark_results.csv`
- `per_target_residual_results.csv`
- `residual_distribution.csv`
- `estimator_plus_residual_summary.md`

These artifacts are generated locally and are not committed.

## Benchmark Results

| Approach | Test Group MAE |
| --- | ---: |
| Direct geometry estimator | 4.3130 |
| Phase 4A calibrated ML model | 1.6777 |
| Geometry + residual GradientBoosting | 1.6422 |

Residual model comparison:

| Residual Model | Train MAE | Val MAE | Test MAE | Beats Direct | Beats Phase 4A ML |
| --- | ---: | ---: | ---: | --- | --- |
| GradientBoosting | 1.0712 | 1.6568 | 1.6422 | true | true |
| Ridge | 1.6598 | 1.6599 | 1.8016 | true | false |
| ElasticNet | 1.6846 | 1.6844 | 1.8271 | true | false |
| RandomForest | 0.9566 | 1.7673 | 1.8666 | true | false |

Best run:

- `geometry_plus_residual__gradient_boosting`
- Test group MAE: 1.6422

Per-target test MAE for the best residual run:

| Target | Test MAE | Gate |
| --- | ---: | --- |
| chest_cm | 1.4692 | stronger synthetic candidate |
| waist_cm | 1.7856 | stronger synthetic candidate |
| hip_cm | 1.7754 | stronger synthetic candidate |
| thigh_cm | 1.5385 | stronger synthetic candidate |

## Residual Stability

The direct estimator residuals are not tiny for every target, especially chest and waist:

| Target | Test Residual Mean | Test Residual Std | Test Mean Abs Residual | Test P90 Abs Residual | Large Residual Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| chest_cm | -0.7307 | 7.6790 | 6.5748 | 12.3958 | 22 |
| waist_cm | -0.1722 | 6.2882 | 5.0903 | 10.9679 | 14 |
| hip_cm | -0.1690 | 4.4452 | 3.4260 | 7.3523 | 2 |
| thigh_cm | 0.3240 | 2.7307 | 2.1609 | 4.4978 | 0 |

For the best GradientBoosting residual run, confidence flags were:

- `ok`: 3760 target predictions
- `large_residual_correction`: 240 target predictions

This means residual correction is useful, but not always small. Large corrections should lower confidence.

## Interpretation

The hybrid model slightly beats the Phase 4A calibrated ML benchmark while preserving per-sample geometry explanations.

Important nuance:

- The geometry estimator alone is auditable but too simple for chest and waist.
- The residual model adds nonlinear correction and substantially improves accuracy.
- The residual model still uses synthetic geometry-calibrated labels, so this remains a synthetic benchmark.
- Large residual corrections indicate cases where the transparent estimator is not enough.

## Product Guidance

Do not claim real-world production tape accuracy.

Recommended behavior:

- Use the geometry estimate as the auditable base measurement.
- Use residual correction as an AI adjustment with confidence flags.
- Show low confidence when residual correction is large.
- Require manual confirmation before custom garment production.
- Do not use these predictions as sole cutting instructions until real-world calibration is complete.

## Recommendation

Next phase should make this hybrid pipeline first-class:

- persist geometry estimate, residual correction, final estimate, and confidence flags in a reusable output schema,
- add renderer-side measured labels so calibrated labels are generated during rendering,
- validate the hybrid estimator on independent measured data,
- keep the transparent geometry estimator as the audit baseline for every future model.
