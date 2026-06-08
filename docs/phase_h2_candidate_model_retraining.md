# Phase H.2 Candidate Model Retraining

Phase H.2 trains a candidate Body AI measurement model from verified FashionApp datasets ingested by Phase H.1.

This phase does not replace the production model, change live inference, promote models, or modify production API responses.

## Candidate Training Flow

```text
Verified Dataset Version
-> Train Candidate Model
-> Training Metrics
-> Candidate Artifact
-> Ready For Evaluation
```

The training entrypoint is:

```powershell
python -m training.train_candidate_model --dataset <verified-dataset-root> --dataset-version v1 --output artifacts/candidates/<run-name>
```

The trainer loads `VerifiedMeasurementDatasetLoader`, filters to one explicit dataset version, builds deterministic train/validation/test splits, trains a ridge-style candidate model, writes metrics and config artifacts, and updates a candidate-only registry entry.

## Inputs

The candidate trainer validates and records:

- front image references
- side image references
- back image references
- pose metadata summaries
- validation metadata summaries
- final approved measurements
- correction deltas

In Phase H.2 compatibility mode, image references are validated for existence but image pixels are not consumed as model features. This is intentional because verified real-world image assets may not yet be normalized enough for the existing silhouette image feature extractor.

The active compatibility feature pipeline uses:

- numeric pose metadata features
- numeric validation metadata features
- numeric correction delta features
- final approved measurements as supervised targets

The model artifact explicitly records `imageUsage.pixelsConsumed=false`.

## Candidate Artifact Structure

The output directory contains:

- `model.json`
- `training_config.json`
- `training_metrics.json`
- `candidate_model_registry.json`
- `candidate_training_report.md`

`model.json` stores:

- `modelVersion`, such as `candidate_model_v1`
- `datasetVersion`, such as `v1`
- `trainingTimestamp`
- `recordCount`
- model coefficients and feature normalization values
- `trainingConfig`
- `trainingMetrics`
- `candidateOnly=true`
- `isProduction=false`

`candidate_model_registry.json` stores candidate entries with:

- `candidateStatus=ready_for_evaluation`
- `productionStatus=not_production`
- `productionModelUpdated=false`
- artifact paths for model, config, and metrics

The registry is not a promotion mechanism. It is only a local candidate inventory for evaluation.

## Training Metrics

`training_metrics.json` reports mean absolute error in centimeters:

- overall MAE by split
- per-measurement MAE by split
- train/validation/test record counts
- target columns

Default Phase H.2 targets:

- `chest_cm`
- `waist_cm`
- `hip_cm`
- `shoulder_cm`
- `sleeve_cm`
- `inseam_cm`
- `neck_cm`

## Reproducible Configuration

`training_config.json` records:

- dataset root
- records file
- dataset version
- model version
- random seed
- split policy
- ridge alpha
- target columns
- feature names
- compatibility-mode feature policy

Using the same dataset, dataset version, model version, random seed, split sizes, and ridge alpha reproduces the same candidate artifact.

## Known Limitations

- Image pixels are not used in Phase H.2 compatibility mode.
- The compatibility model can learn from correction deltas, so H.3 must evaluate whether those deltas are available at the intended inference time or should be removed for a deployment-like benchmark.
- No production inference code loads candidate artifacts.
- No API response shape changes are made.
- No promotion gate exists in this phase.
- The candidate registry does not compare against production or prior real-world candidates.
- Small verified datasets may produce unstable train/validation/test metrics.
- Consent, retention, duplicate-subject, and holdout-group policies remain prerequisites before using larger real-world training data.

## H.3 Evaluation Prerequisites

Before H.3 evaluation, confirm:

- candidate artifact exists and is candidate-only
- metrics were generated for the required targets
- the evaluation set is independent from training records
- correction-delta features are evaluated for leakage risk
- image-feature readiness is decided separately from metadata compatibility mode
- production inference and API behavior remain unchanged
- any promotion path requires an explicit later phase

## Verification

Focused candidate tests:

```powershell
python -m pytest tests/test_candidate_model_training.py
```

Full regression:

```powershell
python -m pytest
```

Whitespace check:

```powershell
git diff --check
```
