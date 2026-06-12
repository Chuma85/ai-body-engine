# Model Lifecycle Platform

The AI Body lifecycle platform connects approved dataset registry entries to reviewable model lifecycle records.

It does not automatically train, promote, or replace production models.

## Flow

```text
Dataset Registry
Training Queue
Training Manifest
Training Run Registry
Model Registry
Evaluation Gate
Promotion Decision
Production Model Tracking
Rollback
```

## Registry Files

Lifecycle records live under:

```text
model_lifecycle/
  training_queue.json
  training_runs.json
  model_registry.json
  promotion_decisions.json
  production_models.json
  registry_dashboard.json
  training_manifest.json
```

Dataset readiness is read from:

```text
dataset_registry/datasets.json
```

Eligible datasets must have either:

- `status: approved_for_training`
- `validation_status: validated`
- legacy-compatible `ready_for_training`

## Guarantees

- Training queue creation does not start training.
- Training manifest generation requires an approved queue item.
- Training run registration starts as `pending` unless explicitly set otherwise.
- Completing a training run creates a model with `status: evaluation_pending`.
- Evaluation report generation does not promote the model.
- Promotion approval does not replace production.
- Production activation requires an explicit promotion call with an approved decision.
- Rollback is explicit and recorded.

## Dashboard Data

`registry_dashboard.json` exposes:

- production model
- evaluation candidates
- training queue
- recent training runs
- archived models

Use:

```powershell
python scripts\list_training_queue.py
python scripts\list_training_runs.py
python scripts\list_models.py
python scripts\list_candidates.py
```
