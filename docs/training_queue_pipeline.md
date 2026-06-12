# Training Queue Pipeline

The training queue starts from a validated dataset registry entry and records intent to train without running training.

## Queue Fields

Each queue record tracks:

- `queue_id`
- `dataset_version`
- `source_export_id`
- `source_dataset_registry_id`
- `model_base_version`
- `created_at`
- `queued_by`
- `notes`
- `status`
- `audit`

## Statuses

- `pending`
- `approved_for_training`
- `training`
- `completed`
- `failed`
- `cancelled`
- `archived`

## Required Review Step

Training manifest generation requires queue status `approved_for_training`.

This prevents a newly ingested dataset from silently becoming a training run.

## Training Manifest

`training_manifest.json` includes:

- `dataset_version`
- `source_registry_entry`
- `model_base_version`
- `schema_version`
- `lineage`
- `training_parameters`
- `generated_timestamp`
- `training_execution.auto_train: false`

The manifest is an auditable handoff to a future explicit training runner.
