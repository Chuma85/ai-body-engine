# Phase 3A Deep Image Model Scaffold

Dataset: `data/synthetic/phase_2v`

Current best non-deep baseline: `phase_2v_image_features`

Current best test MAE: `9.4726`

## Why Phase 3A Starts Now

The Phase 2 non-deep baselines appear to be plateauing:

| Phase | Approach | Test MAE |
| --- | --- | ---: |
| Phase 2V | regular ridge image-feature baseline | 9.4726 |
| Phase 2W | target-tuned ridge | 9.4875 |
| Phase 2Y | advanced silhouette geometry features | 9.6182 |
| Phase 2AA | feature-selected ridge | 9.5483 |

Phase 2AA improved some individual targets, especially `waist_cm`, `neck_cm`, and `shoulder_cm`, but it did not beat the Phase 2V regular ridge baseline overall. That makes Phase 3A a reasonable point to open the learned image-representation path.

## What Was Added

Phase 3A adds a guarded PyTorch scaffold:

- `training/deep/synthetic_body_image_dataset.py`
- `training/deep/models/simple_front_side_cnn.py`
- `training/deep/train_front_side_cnn.py`

The dataset adapter uses the existing manifest-based loader, loads paired front/side images, resizes them, normalizes pixels to `0..1`, and returns the existing 11 measurement targets in the same order as the baseline trainers.

The model is intentionally small:

- lightweight image encoder
- shared front/side encoder by default
- concatenated front/side embeddings
- regression head for the 11 measurement targets

## Dependency Status

`requirements.txt` already lists `torch` and `torchvision`, and PyTorch is available in the current environment.

The scaffold still guards the dependency:

- importing the deep package does not require PyTorch
- the training CLI exits with a clear message if PyTorch is missing
- tests skip or isolate PyTorch-specific checks so users without the dependency do not break the rest of the repo

If PyTorch is missing, install the repository requirements:

```powershell
pip install -r requirements.txt
```

Or install a platform-specific PyTorch build from the official PyTorch install selector.

## Smoke Training Command

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3a_smoke --epochs 1 --limit-samples 32
```

Smoke result in this environment:

| Split | Overall MAE |
| --- | ---: |
| train | 9.8170 |
| val | 9.4449 |

Samples used:

| Split | Count |
| --- | ---: |
| train | 32 |
| val | 32 |

This is only a dependency and training-loop smoke test. It is not a fair benchmark against Phase 2V because it uses a tiny sample limit, runs for one epoch, and does not evaluate the held-out test split.

## What Phase 3A Is Not

Phase 3A does not replace the current best baseline.

Phase 3A does not claim the CNN is better than ridge.

Phase 3A does not introduce long training, GPU requirements, or new synthetic data generation.

## Next Steps

Phase 3B should turn this scaffold into a controlled experiment:

- train on the full 800-sample train split
- evaluate val and test
- save prediction CSVs like the image-feature experiment runner
- compare directly against Phase 2V in the baseline registry
- keep Phase 2V regular ridge as the benchmark until a deep model beats `9.4726` test MAE
