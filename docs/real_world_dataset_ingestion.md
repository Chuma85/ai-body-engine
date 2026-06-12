# Real-World Dataset Ingestion Bridge

This bridge accepts approved dataset exports from CUSTOM-FASHION-MARKETPLACE and registers them as real-world training candidates.

It does not train models, evaluate candidates, promote models, or overwrite model artifacts.

## Staging Areas

Real-world dataset packages flow through:

```text
data/real_world/
  incoming/
  validated/
  rejected/
  archived/
```

Only `.gitkeep` placeholders are tracked. Export contents, images, and imported packages are local/runtime artifacts.

## Input Package

The import command takes a `dataset_export_manifest.json` path:

```powershell
python scripts\import_real_world_dataset.py data\real_world\incoming\<dataset_version>\dataset_export_manifest.json
```

The manifest must include:

- `schemaVersion: real-world-dataset-export-v1`
- `sourceSystem`
- `sourceExportId`
- `sourceAppVersion`
- `sourceDatasetVersion`
- `datasetVersion`
- `exportTimestamp`
- `approvedOnly: true`
- `imageCount`
- `measurementCount`
- `participantCount`
- `recordsPath`
- `labelsPath`
- `metadataPath`
- `consentMetadataPath`

Supported companion schema versions are:

- `field-measurements-v1`
- `image-quality-v1`

## Validation

The validator rejects an import when:

- the manifest is missing
- image files are missing
- label files or participant labels are missing
- metadata is missing
- consent metadata is missing or record consent is not granted
- `approvedOnly` is not true
- the export schema is unsupported
- rejected records are present
- required front, side, and back views are missing
- required measurements are missing or invalid
- duplicate participants appear
- quality scores fall below the bridge threshold

The required measurement keys are:

- `height`
- `weight`
- `bust_chest`
- `waist`
- `hips`
- `shoulder`
- `dress_length`

## Registry

The registry lives at:

```text
dataset_registry/datasets.json
```

Each entry tracks:

- `dataset_version`
- `source_system`
- `source_export_id`
- `export_timestamp`
- `import_timestamp`
- `schema_version`
- `image_count`
- `measurement_count`
- `participant_count`
- `quality_summary`
- `validation_status`
- `training_status`
- `status`
- lineage with source app version, source dataset version, import timestamp, and validation timestamp

Dataset candidate statuses are:

- `imported`
- `validating`
- `validated`
- `rejected`
- `approved_for_training`

Successful imports finish with `status: approved_for_training`, `validation_status: validated`, and `training_status: not_started`.

## Validation Report

Every import writes:

```text
reports/dataset_validation_report.json
```

The report includes:

- `missing_images`
- `missing_labels`
- `duplicate_participants`
- `low_quality_records`
- `invalid_measurements`
- `schema_mismatches`

It also includes missing metadata, missing consent metadata, missing views, rejected records, quality summary, and lineage identifiers.

## Registry Viewer

Use:

```powershell
python scripts\list_datasets.py
```

The viewer shows:

- `dataset_version`
- `status`
- `record_count`
- `quality_score`

## Verification

Run:

```powershell
python -m pytest tests/test_real_world_training_candidate_ingestion.py
pytest
```
