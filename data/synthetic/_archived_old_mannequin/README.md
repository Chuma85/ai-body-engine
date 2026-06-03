# Archived Old Mannequin Synthetic Datasets

This folder quarantines synthetic datasets from earlier mannequin, procedural-body, render-realism, and body-asset experiments.

These datasets are preserved for historical comparison only. They must not be used for active training, audits, or benchmark inputs unless a phase prompt explicitly requests one of these archived paths.

The current active dataset is:

```text
data/synthetic/phase_3t
```

Future training scripts should use an explicit dataset path. Do not automatically glob all `data/synthetic/phase_*` folders, because archived datasets represent older render assumptions and can contaminate current training results.
