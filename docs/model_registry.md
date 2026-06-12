# Model Registry

The model registry stores candidate, approved, production, and archived model records.

It is metadata-only. It does not load weights, run inference, or update API behavior.

## Model Fields

Each model tracks:

- `model_version`
- `model_type`
- `parent_model_version`
- `training_run_id`
- `training_dataset_versions`
- `created_at`
- `status`
- `lineage`

## Statuses

- `development`
- `evaluation_pending`
- `approved`
- `production`
- `archived`

## Lineage

Lineage connects:

```text
production model
candidate model
training run
dataset versions
source exports
evaluation reports
```

The lineage payload stores:

- parent model
- training datasets
- training manifests
- training runs
- evaluation reports

## Training Run Registry

Training runs track:

- `training_run_id`
- `dataset_version`
- `model_version`
- `model_base_version`
- `start_time`
- `end_time`
- `duration`
- `status`
- `metrics`

Training run statuses are:

- `pending`
- `running`
- `completed`
- `failed`
- `cancelled`

When a run is completed, the resulting model is registered as `evaluation_pending`.
