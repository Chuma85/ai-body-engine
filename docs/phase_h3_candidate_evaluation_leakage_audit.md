# Phase H.3 Candidate Evaluation And Leakage Audit

Phase H.3 evaluates a candidate Body AI measurement model produced by Phase H.2. It compares the candidate against a verified-dataset baseline, audits leakage and split integrity, analyzes confidence calibration, and writes a conservative promotion recommendation.

This phase does not promote a model, replace the production model, change live inference, or modify production API responses.

## Evaluation Entry Point

```powershell
python -m training.evaluate_candidate_model --dataset <verified-dataset-root> --candidate-model <candidate-output>/model.json --output <evaluation-output>
```

The evaluator loads:

- the verified dataset through `VerifiedMeasurementDatasetLoader`
- the candidate `model.json`
- the candidate `trainingConfig`
- the candidate registry when available

It rebuilds the deterministic train/validation/test split from the candidate training configuration and evaluates the held-out test split for the promotion recommendation.

## Output Artifacts

The evaluation output directory contains:

- `candidate_evaluation_metrics.json`
- `leakage_audit.json`
- `split_audit.json`
- `candidate_evaluation_report.md`

## Metrics

The evaluator reports mean absolute error in centimeters for:

- overall MAE
- per-measurement MAE
- chest
- waist
- hip
- shoulder
- sleeve
- inseam
- neck

The comparison includes:

- baseline MAE
- candidate MAE
- absolute improvement
- percentage improvement
- regressions

The default baseline is a verified train-split mean estimator. It is a conservative comparison baseline for the verified dataset and does not represent a production promotion target.

## Confidence Calibration

The evaluator groups held-out test errors by exported confidence tier:

- high confidence actual error
- medium confidence actual error
- low confidence actual error
- unknown confidence actual error

This is descriptive calibration analysis only. It does not recalibrate the model or alter production confidence behavior.

## Leakage Audit

The leakage audit checks for:

- final approved measurements appearing as input features
- customer, maker, or final measurement lineage appearing as input features
- correction-delta features
- correction deltas or other features that exactly match a target
- near-perfect feature-target correlations
- suspiciously low held-out errors

Correction deltas are a known Phase H.2 risk surface. They can be useful for analysis, but if they encode the target or are not available at intended inference time, they are leakage. Any high-risk leakage finding blocks promotion.

## Split Integrity Audit

The split audit checks for duplicate identifiers crossing train/validation/test:

- `profileId`
- `scanSessionId`
- `orderId`

It also writes the deterministic split indices and split counts so the evaluation can be reproduced.

## Compatibility Disclaimer

Phase H.3 preserves the H.2 compatibility-mode disclaimer:

- `pixelsConsumed=false`
- front, side, and back image references are validated
- the back image is accepted but not pixel-weighted
- image-learning is not yet implemented

This means the candidate is a metadata/correction compatibility candidate, not an image+metadata model.

## Promotion Gate Logic

The evaluator writes a recommendation but does not promote. Possible decisions:

- `promote`
- `do_not_promote`
- `needs_more_data`
- `leakage_risk`
- `regression_detected`

Conservative rules:

- If leakage risk is detected, the recommendation is not `promote`.
- If important measurements regress, the recommendation is not `promote`.
- If test data is insufficient, the recommendation is `needs_more_data`.
- If split integrity fails, promotion is blocked.

## Known Limitations

- The baseline is a verified train-split mean estimator, not a live production inference artifact.
- The candidate still does not consume image pixels.
- Correction-delta features require H.3/H.4 decisions before any deployment-like benchmark.
- Small test splits can make MAE and confidence calibration unstable.
- Promotion requires a future explicit phase with independent evaluation and production integration review.

## Next Step Toward Image And Metadata Training

Before training a stronger image+metadata model:

- normalize verified front/side/back image assets into a stable tensor or feature pipeline
- decide whether correction deltas are training-only labels, evaluation diagnostics, or disallowed inference features
- enforce subject/order-level holdout splits before training
- calibrate confidence on an independent verified test set
- keep candidate and production artifact registries separate until a promotion phase is explicitly authorized

## Verification

```powershell
python -m pytest tests/test_candidate_model_training.py
python -m pytest tests/test_verified_measurement_dataset.py
python -m pytest
git diff --check
```
