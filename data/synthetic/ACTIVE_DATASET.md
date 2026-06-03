# Active Synthetic Dataset

Active dataset: `phase_3t`

Asset source: newer body mesh workflow.

Old mannequin and early body-asset experiment datasets are archived under:

```text
data/synthetic/_archived_old_mannequin/
```

Training must not automatically glob all `data/synthetic/phase_*` folders.

Training scripts must require an explicit `--dataset-dir` or `--dataset` argument, or default only to:

```text
data/synthetic/phase_3t
```

Archived datasets are preserved for historical comparison only. They must not be used for active training unless a phase prompt explicitly requests that historical dataset path.
