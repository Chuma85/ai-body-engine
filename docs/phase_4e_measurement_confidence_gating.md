# Phase 4E Measurement Confidence Gating

Phase 4E adds confidence tiers and product-safe actions for geometry + residual measurement predictions.

## Inputs

- Phase 4D predictions: `artifacts/phase_4d_residual_correction/residual_training_summary.csv`
- Run evaluated: `geometry_plus_residual__gradient_boosting`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered, no new data was generated, and no CNN was trained.

## Policy

The confidence policy uses:

- geometry estimate,
- predicted residual correction,
- absolute residual size,
- residual size relative to final measurement,
- geometry quality flags,
- existing residual confidence flags,
- target name and model family.

Initial thresholds:

| Tier | Criteria | Product Action |
| --- | --- | --- |
| high confidence | small residual correction and clean geometry | `accept_as_ai_estimate` |
| medium confidence | moderate residual correction or minor concern | `require_manual_confirmation` |
| low confidence | large residual correction, bad geometry, unstable mask, or out-of-range prediction | `request_retake_or_tape_measurement` |

Thresholds used:

- high max absolute residual: 4.0 cm
- high max relative residual: 4.5%
- medium max absolute residual: 8.0 cm
- medium max relative residual: 9.0%

## Artifacts

Local artifacts were written under `artifacts/phase_4e_measurement_confidence_gating/`:

- `confidence_policy.json`
- `confidence_eval_results.json`
- `confidence_eval_results.csv`
- `per_target_confidence_summary.csv`
- `confidence_gate_summary.md`

These artifacts are generated locally and are not committed.

## Results

All splits:

| Tier | Action | Count | Percent | MAE | P90 Error | Mean Residual Correction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| high_confidence | accept_as_ai_estimate | 2338 | 58.5 | 1.1893 | 2.5455 | 1.5779 |
| medium_confidence | require_manual_confirmation | 1146 | 28.6 | 1.1695 | 2.4668 | 5.1405 |
| low_confidence | request_retake_or_tape_measurement | 516 | 12.9 | 1.2140 | 2.6274 | 10.1620 |

Test split:

| Tier | Action | Count | Percent | MAE | P90 Error | Mean Residual Correction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| high_confidence | accept_as_ai_estimate | 240 | 60.0 | 1.6619 | 3.3650 | 1.5386 |
| medium_confidence | require_manual_confirmation | 107 | 26.8 | 1.4210 | 3.0715 | 5.3130 |
| low_confidence | request_retake_or_tape_measurement | 53 | 13.2 | 1.9992 | 4.2146 | 9.9502 |

The low-confidence test bucket has the highest MAE and p90 error. That makes the policy directionally useful for product safety.

Important nuance: medium confidence has slightly lower MAE than high confidence on this synthetic split. These gates should be interpreted as intervention and risk tiers, not a perfectly calibrated probability of error.

## Per-Target Test Summary

| Target | High MAE | Medium MAE | Low MAE |
| --- | ---: | ---: | ---: |
| chest_cm | 1.2548 | 1.3878 | 1.8888 |
| waist_cm | 1.7100 | 1.6685 | 2.0855 |
| hip_cm | 1.8114 | 1.5269 | 2.8878 |
| thigh_cm | 1.6627 | 0.9155 | 1.0545 |

Low confidence is most useful for chest, waist, and hip. Thigh has very few low-confidence examples, so its bucket is not very informative yet.

## Product Behavior

Recommended mapping:

- `high_confidence`: show as an AI estimate.
- `medium_confidence`: show as an AI estimate but require manual confirmation.
- `low_confidence`: request a retake or tape measurement.

Even high-confidence predictions are not production-ready tape measurements. For custom garment production, manual confirmation is still required until real-world calibration exists.

## Interpretation

Confidence gating is necessary because Phase 4D still had large residual-correction cases. A large residual means the transparent geometry estimate and learned correction disagree meaningfully, which should reduce trust even if synthetic MAE remains low.

The policy does what we need for the next product workflow step:

- preserves auditable geometry + residual outputs,
- converts predictions into user-facing action tiers,
- routes large correction cases away from silent acceptance,
- avoids claiming real-world production readiness.

## Recommendation

Next phase should integrate this confidence schema with the measurement output contract:

- include geometry estimate, residual correction, final estimate, confidence tier, and product action in one prediction object,
- keep low-confidence retake/manual-measurement paths explicit,
- calibrate thresholds on real tape-measured data before using them for production tailoring decisions.
