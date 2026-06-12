# Real-World Dataset Ingestion

This ingestion path accepts reviewed FashionApp field-data exports as training candidate packages.

It does not train models, evaluate model candidates, promote models, or overwrite model artifacts.

## Incoming Package

Place a package at:

```text
data/real_world/incoming/<dataset_version>/
```

Required files:

- `dataset_export_manifest.json`
- `records.json`
- image files referenced by each record

The manifest must declare:

- `export_id`
- `dataset_version`
- `export_timestamp`
- `exported_by_admin_id`
- `session_count`
- `image_count`
- `measurement_count`
- `consent_version`
- `measurement_schema_version`
- `image_quality_schema_version`
- `app_version`
- `approved_only: true`

The loader accepts both snake_case and camelCase keys so FashionApp exports can be passed through directly.

## Validation

The ingestion validator fails if:

- the manifest is missing
- `records.json` is missing
- any image file is missing
- consent metadata is missing or not granted
- `approved_only` is not true
- schema versions are unsupported
- rejected records are included
- front, side, or back views are missing
- required measurements are missing
- record lineage is missing

Supported schema versions:

- `field-measurements-v1`
- `image-quality-v1`

## Processed Dataset

Successful ingestion creates:

```text
data/real_world/processed/<dataset_version>/
```

Output files:

- `images/`
- `labels.csv`
- `metadata.json`
- `quality_scores.csv`
- `lineage.json`

## Dataset Registry

Ingestion updates:

```text
data/real_world/dataset_registry.json
```

Each registry entry tracks:

- dataset version
- source export id
- source app version
- import timestamp
- record count
- approved for training
- validation status

The status `ready_for_training` means validation passed and a future explicit training phase may consume the processed dataset.

## CLI

```powershell
python scripts\ingest_real_world_training_candidate.py v1
```

Optional paths:

```powershell
python scripts\ingest_real_world_training_candidate.py v1 --incoming-root data\real_world\incoming --processed-root data\real_world\processed --registry-path data\real_world\dataset_registry.json
```

## Tests

```powershell
python -m pytest tests/test_real_world_training_candidate_ingestion.py
```
