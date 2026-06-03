# Phase 3H-E Blend Baseline Training

Phase 3H-E trains a simple measurement baseline on the audited Phase 3H-D Blender dataset:

```text
data/synthetic/phase_3h_blend_250
```

This is a baseline model, not a production tailoring model. The labels are synthetic/generated, the renders are Blender-derived, and `real_world_validated=false`. The goal is to check whether image-derived front, side, and back silhouette features contain learnable signal for the generated measurement labels.

## Command

```powershell
python scripts\train_blend_dataset_baseline.py --dataset data\synthetic\phase_3h_blend_250 --out artifacts\phase_3h_e_blend_baseline --seed 42 --test-size 0.2 --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --strict-audit-required
```

The one-step verifier runs the same training command and checks the expected artifacts:

```powershell
python scripts\verify_phase_3h_e_blend_baseline.py
```

## Validation Before Training

The training script blocks when:

- `labels.csv`, `metadata.json`, or `images/` is missing.
- Any referenced front, side, or back image is missing.
- Required target columns are missing or non-numeric.
- Target labels have no variation.
- `--strict-audit-required` is used and the strict audit report is missing or failed.

## Artifacts

Outputs are written to:

```text
artifacts/phase_3h_e_blend_baseline/
  metrics.json
  metrics_summary.md
  predictions.csv
  train_test_split.json
  feature_summary.csv
  model_ranking.csv
  best_model.joblib
  experiment_metadata.json
```

`metrics_summary.md` is the quick human readout: dataset path, sample/image counts, train/test split sizes, target columns, best model, overall mean MAE, per-measurement MAE, and the synthetic-only warning.

## How To Read Results

A promising result means the best model improves over simple non-image baselines on the same dataset and shows consistently lower MAE for measurements that should be visible from silhouette geometry, such as height, shoulder, chest, waist, and hip. It does not prove real-world measurement accuracy.

A weak result should trigger dataset or feature improvements when models cannot beat simple baselines, when only one target learns while others fail, or when rankings are unstable across deterministic seeds. Common next checks are camera framing consistency, stronger shape-key variation, label-to-geometry alignment, and additional view-specific silhouette features.

## Limitations

This experiment is synthetic-only. Performance is not production tailoring accuracy yet, should not be used for customer-facing fit claims, and must be validated on appropriate real capture data before any real-world accuracy statement.
