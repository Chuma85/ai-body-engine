# Phase 4F Measurement Uncertainty Calibration

Phase 4F adds calibrated uncertainty estimates and prediction intervals for geometry + residual body measurements.

## Inputs

- Phase 4D predictions: `artifacts/phase_4d_residual_correction/residual_training_summary.csv`
- Run evaluated: `geometry_plus_residual__gradient_boosting`
- Confidence policy: Phase 4E confidence tiers
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered, no new data was generated, and no CNN was trained.

## Why Confidence Tiers Are Not Enough

Phase 4E confidence tiers are useful as product action gates, but they do not provide a measurement range. Phase 4F adds an explicit estimated error interval:

- `estimated_error_cm`
- `prediction_interval_low_cm`
- `prediction_interval_high_cm`

These intervals let the product say, for example, "estimated waist is 82 cm, likely within plus/minus N cm" instead of only saying high/medium/low confidence.

## Calibration Method

Uncertainty is calibrated from train and validation rows only.

For each prediction, Phase 4F uses empirical absolute-error quantiles:

1. target + confidence-tier p90 error if enough calibration rows exist,
2. target p90 fallback,
3. confidence-tier p90 fallback,
4. global p90 fallback.

A conservative safety multiplier is applied because train/validation p90 was optimistic relative to test:

- base safety multiplier: 1.35
- high confidence tier multiplier: 1.0
- medium confidence tier multiplier: 1.1
- low confidence tier multiplier: 1.25

Test data is used only for evaluation, not fitting thresholds.

## Artifacts

Local artifacts were written under `artifacts/phase_4f_measurement_uncertainty_calibration/`:

- `uncertainty_policy.json`
- `uncertainty_eval_results.json`
- `uncertainty_eval_results.csv`
- `per_target_uncertainty_summary.csv`
- `coverage_summary.md`
- `product_action_policy.md`

These artifacts are generated locally and are not committed.

## Test Coverage

| Target | Confidence | Count | Coverage | MAE | P90 Error | Mean Estimated Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all_targets | all_confidence | 400 | 0.9025 | 1.6422 | 3.3645 | 3.4144 |
| all_targets | high_confidence | 240 | 0.8708 | 1.6619 | 3.3650 | 3.2644 |
| all_targets | medium_confidence | 107 | 0.9720 | 1.4210 | 3.0715 | 3.5047 |
| all_targets | low_confidence | 53 | 0.9057 | 1.9992 | 4.2146 | 3.9113 |

The calibrated intervals reach about 90% overall test coverage. Low-confidence predictions receive wider intervals, as intended.

## Per-Target Coverage

| Target | Count | Coverage | MAE | P80 Error | P90 Error | Mean Estimated Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chest_cm | 100 | 0.9000 | 1.4692 | 2.5441 | 3.1433 | 3.1730 |
| waist_cm | 100 | 0.9800 | 1.7856 | 2.9073 | 3.3852 | 4.0350 |
| hip_cm | 100 | 0.8900 | 1.7754 | 3.0118 | 3.6521 | 3.4679 |
| thigh_cm | 100 | 0.8400 | 1.5385 | 2.5046 | 3.2020 | 2.9817 |

Thigh remains undercovered relative to the intended p90-style interval. This is a useful warning: target-specific interval calibration still needs more data and real-world validation.

## Product Action Policy

Recommended mapping:

- high confidence and interval <= 3 cm: `accept_as_ai_estimate`
- medium confidence or interval > 3 cm: `require_manual_confirmation`
- low confidence or interval > 5 cm: `request_retake_or_tape_measurement`

These actions remain conservative because the calibration is synthetic-only.

## Interpretation

Phase 4F upgrades confidence from simple tiers to actionable measurement-risk estimates. Each prediction can now carry:

- final measurement estimate,
- estimated error range,
- prediction interval,
- confidence tier,
- product action.

This is still not real-world production readiness. The intervals are calibrated against synthetic geometry-calibrated labels, not tape-measured humans.

## Recommendation

Next phase should integrate the measurement output contract:

- final estimate,
- interval low/high,
- estimated error,
- confidence tier,
- product action,
- geometry and residual explanation fields.

Before production use, interval calibration must be repeated with real tape-measured validation data.
