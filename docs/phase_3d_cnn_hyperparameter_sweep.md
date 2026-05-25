# Phase 3D CNN Hyperparameter Sweep

## Goal

Phase 3D added a small, CPU-safe hyperparameter sweep for the simple front/side CNN. The goal was to test whether modest training configuration changes could beat the current best non-deep baseline without changing the dataset, renderer, or model family.

The current best baseline entering this phase remained Phase 2V regular ridge image features:

| Baseline | Test MAE |
| --- | ---: |
| Phase 2V ridge image features | 9.4726 |

## Dataset

- Dataset: `data/synthetic/phase_2v`
- Samples: 1000
- Split: 800 train / 100 val / 100 test
- No new images were generated in this phase.

## Sweep

Command used for dry-run:

```powershell
python -m training.deep.sweep_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3d_cnn_sweep --epochs 10 --patience 3 --max-runs 4 --device cpu --seed 42 --dry-run
```

Command used for the real sweep:

```powershell
python -m training.deep.sweep_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3d_cnn_sweep --epochs 10 --patience 3 --max-runs 4 --device cpu --seed 42
```

The real sweep completed 4 runs in about 2916.5 seconds, or 48.6 minutes.

The default grid supports:

| Parameter | Values |
| --- | --- |
| image_size | 96, 128 |
| learning_rate | 0.001, 0.0005 |
| batch_size | 16, 32 |
| weight_decay | 0.0, 0.0001 |

For this phase, `--max-runs 4` limited the CPU run to the first four prioritized configurations.

## Results

| Rank | Run | Image Size | Learning Rate | Batch Size | Weight Decay | Best Epoch | Val MAE | Test MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `run_001_img128_lr0p001_bs32_wd0p0` | 128 | 0.001 | 32 | 0.0 | 6 | 9.3970 | 9.5814 |
| 2 | `run_003_img96_lr0p001_bs32_wd0p0` | 96 | 0.001 | 32 | 0.0 | 6 | 9.3971 | 9.5815 |
| 3 | `run_004_img128_lr0p001_bs16_wd0p0` | 128 | 0.001 | 16 | 0.0 | 6 | 9.3986 | 9.5830 |
| 4 | `run_002_img128_lr0p0005_bs32_wd0p0` | 128 | 0.0005 | 32 | 0.0 | 6 | 9.3988 | 9.5842 |

Best run by validation MAE:

- Run: `run_001_img128_lr0p001_bs32_wd0p0`
- Image size: 128
- Learning rate: 0.001
- Batch size: 32
- Weight decay: 0.0
- Best epoch: 6
- Train MAE: 9.4665
- Val MAE: 9.3970
- Test MAE: 9.5814

## Ridge Comparison

The best CNN sweep run was compared against the Phase 2V ridge baseline:

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V ridge image features | 9.0564 | 9.4635 | 9.4726 |
| Phase 3D best CNN sweep run | 9.4665 | 9.3970 | 9.5814 |

The CNN improved validation MAE relative to ridge, but it did not improve test MAE. Test MAE was worse by 0.1087, so Phase 2V ridge remains the current best benchmark.

Per-target comparison showed the CNN improved some smaller targets, including `shoulder_cm`, `inseam_cm`, `sleeve_cm`, `neck_cm`, `calf_cm`, and slightly `waist_cm`, but it regressed enough on `weight_kg`, `height_cm`, `chest_cm`, `hip_cm`, and `thigh_cm` to lose overall.

## Recommendation

The small sweep suggests simple hyperparameter changes are not enough to beat the ridge image-feature baseline. The next phase should focus on either a modest architecture improvement, better image input handling, or stronger synthetic realism/label signal rather than expanding the same CPU sweep much further.

Phase 2V regular ridge remains the current best baseline at 9.4726 test MAE.
