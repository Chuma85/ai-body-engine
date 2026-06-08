# Phase H.5 Vision Multimodal Candidate Training

Phase H.5 adds a candidate-only vision training path for verified FashionApp datasets. It does not promote a model, replace production inference, or change live API behavior.

## Training Entry Point

Run the trainer with:

```bash
python -m training.train_vision_candidate_model --dataset <verified-dataset-root> --output <artifact-dir> --dataset-version v1 --device cpu
```

The trainer uses `training/datasets/multimodal_verified_dataset.py` with tensors enabled. Training fails before fitting if any selected record is not `multimodal_ready` or if any front, side, or back image tensor is missing.

## Architecture

The model uses separate branches for each view:

- `front_image_encoder`
- `side_image_encoder`
- `back_image_encoder`
- `metadata_feature_encoder`
- `fusion_layer`
- `measurement_prediction_head`

Each image branch receives its own `[channels, height, width]` tensor. The branch outputs are concatenated with metadata features, passed through the fusion layer, and then projected to measurement predictions.

This is intentionally lightweight for small-data and CPU-compatible candidate runs. It is a foundation for H.6 evaluation, not a production-grade vision model.

## Inputs

Allowed inputs:

- front image tensor
- side image tensor
- back image tensor
- pose metadata
- validation metadata
- verification metadata

Targets:

- final approved measurements

The default target columns are:

- `chest_cm`
- `waist_cm`
- `hip_cm`
- `shoulder_cm`
- `sleeve_cm`
- `inseam_cm`
- `neck_cm`

## Leakage Controls

Final approved measurements are used only as targets. The trainer does not use these fields as input features:

- AI estimate measurements
- customer edits
- maker adjustments
- final approved measurements
- correction deltas

Metadata feature extraction is limited to pose, validation, and verification metadata. Feature names containing leakage-prone target or lineage tokens fail fast instead of being silently accepted.

## Artifacts

Vision candidate artifacts are stored separately from metadata-only candidates:

- `vision_model.json`
- `vision_model.pt`
- `vision_training_config.json`
- `vision_training_metrics.json`
- `vision_candidate_model_registry.json`
- `vision_candidate_training_report.md`

The registry entry includes:

- `candidateType: vision_multimodal`
- `pixelsConsumed: true`
- `productionModelUpdated: false`
- `readyForEvaluation: true`
- `isProduction: false`

## Metrics

The trainer writes MAE metrics in centimeters:

- train MAE
- validation MAE
- test MAE
- per-measurement MAE for every target column

Metrics are written to `vision_training_metrics.json` and copied into the candidate registry entry.

## Limitations

- This phase trains a small candidate architecture only.
- No model is promoted.
- Production inference remains unchanged.
- The model is not yet benchmarked against production in this phase.
- The image branches consume tensors, but the architecture is intentionally minimal until H.6 evaluation establishes trustworthiness.
- Real-world dataset scale, diversity, calibration, and leakage audits are still required before promotion can be considered.

## Readiness For H.6

H.5 produces a candidate artifact with `pixelsConsumed: true` and `readyForEvaluation: true`. H.6 should evaluate the vision candidate against baseline and metadata-only candidates, repeat leakage and split-integrity audits, inspect confidence calibration, and block promotion if regressions or leakage risk are detected.
