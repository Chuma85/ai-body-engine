# Phase 3T Cleanup: Old Mannequin Dataset Quarantine

This cleanup preserves old synthetic datasets while preventing them from being discovered as active training inputs.

## Active Dataset

The active dataset remains:

```text
data/synthetic/phase_3t
```

It must not be moved by this cleanup.

## Archive Location

Older mannequin, procedural-body, render-realism, and early body-asset experiment datasets are quarantined under:

```text
data/synthetic/_archived_old_mannequin/
```

These archived datasets are historical comparison material only. They must not be used for active model training, audits, or benchmark inputs unless a future phase explicitly requests one of the archived paths.

## Training Rule

Training scripts must not glob all `data/synthetic/phase_*` folders. They should require an explicit dataset path or default only to:

```text
data/synthetic/phase_3t
```

Current core training loaders already require an explicit dataset path and fail clearly when `manifest.csv`, `labels/labels.csv`, or expected image files are missing.

## Inspection Command

Use:

```powershell
python scripts\list_active_datasets.py
```

The command prints the active dataset path, archive path, active label row count, active image count, and a warning that archived datasets are not used for training.
