# Phase H.1 Verified Dataset Ingestion

Phase H.1 adds ingestion support for verified FashionApp training datasets produced by the Measurement Lifecycle, Verification Engine, Verified Dataset Export, and Continuous Learning Pipeline.

This phase does not retrain models, promote models, or modify production model artifacts.

## Dataset Ingestion Architecture

The ingestion entrypoint is `training.datasets.verified_measurement_dataset.VerifiedMeasurementDatasetLoader`.

Supported record files under a dataset root:

- `records.jsonl`
- `verified_measurements.jsonl`
- `records.json`
- `verified_measurements.json`
- `manifest.json`

JSON files may contain a top-level `records`, `samples`, or `items` array. JSONL files use one verified measurement record per line. The loader accepts snake_case and camelCase field names so exports can evolve without forcing a brittle first-version contract.

Supported dataset versions are explicit string versions matching `v1`, `v2`, `v3`, and later `vN` versions. Unknown non-version labels fail ingestion.

## Preserved Lineage

Each normalized record preserves these separate lineage buckets:

- AI estimate
- Customer edit
- Maker adjustment
- Final approved

These values are not flattened into a single measurement map. Training code can later decide how to compare, weight, or target each lineage stage, but Phase H.1 only validates and reports them.

## Ingestion Validation

Default loading validates that every record has:

- front image reference resolving to an existing local file
- side image reference resolving to an existing local file
- back image reference resolving to an existing local file
- final approved measurements
- non-empty AI estimate, customer edit, maker adjustment, and final approved lineage buckets

Tracked metadata gaps are also counted for pose, validation, verification, correction, confidence, and eligibility summaries. Invalid exports can be inspected with `validate=False` or the CLI `--allow-invalid`, but default ingestion fails fast.

## Dataset Statistics

`VerifiedMeasurementDatasetLoader.statistics()` returns:

- record count
- dataset version counts
- measurement coverage from final approved measurements
- confidence distribution from confidence metadata and per-measurement confidence fields
- correction distribution with count, mean delta, mean absolute delta, and max absolute delta
- missing field counts

## Dataset Quality Report Design

`VerifiedMeasurementDatasetLoader.quality_report()` returns a JSON-ready pre-training quality report with:

- dataset and records file paths
- validation status and validation errors
- all dataset statistics
- report design metadata documenting lineage and training boundaries

`write_quality_report(output_dir)` writes:

- `verified_measurement_dataset_quality_report.json`
- `verified_measurement_dataset_quality_report.md`

The report is intended as a readiness gate before any retraining phase. It is not a training artifact and does not update model selection.

## Known Gaps Before Retraining

- The loader validates local image existence but does not download or resolve remote signed URLs.
- No holdout split policy is enforced yet.
- No identity leakage, consent, retention, or duplicate-subject audit is implemented in this engine layer.
- Correction deltas are summarized but not yet checked against target-specific acceptance thresholds.
- Confidence metadata is counted as exported; it is not recalibrated here.
- Eligibility metadata is preserved but not yet converted into training/holdout sampling rules.
- Real-world dataset weighting against synthetic datasets remains undefined.

## Verification

Focused tests cover loading, version support, validation failures, statistics, and report output:

```powershell
python -m pytest tests/test_verified_measurement_dataset.py
```

Full regression verification remains:

```powershell
python -m pytest
```
