# Phase 3E Dual-Branch CNN

## Summary

Phase 3E added a modest dual-branch CNN architecture for paired front/side synthetic body images. The model keeps training CPU-safe while giving the front and side views independent encoders by default.

The Phase 2V regular ridge image-feature baseline remains the current best benchmark.

## Architecture

The new `dual_branch_cnn` model includes:

- separate front and side image encoders by default
- optional shared encoder mode
- convolution blocks with batch normalization and ReLU
- adaptive global pooling
- dropout
- concatenated front/side embeddings
- regression head for the existing 11 measurement targets

Training now supports:

```powershell
python -m training.deep.train_front_side_cnn --model simple_cnn
python -m training.deep.train_front_side_cnn --model dual_branch_cnn
```

The default remains `simple_cnn` for backwards compatibility.

## Dataset And Settings

- Dataset: `data/synthetic/phase_2v`
- Split: 800 train / 100 val / 100 test
- Model: `dual_branch_cnn`
- Shared encoder: `false`
- Dropout: `0.2`
- Image size: `128`
- Batch size: `32`
- Learning rate: `0.001`
- Epochs: `10`
- Patience: `3`
- Device: `cpu`
- Seed: `42`

Command:

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3e_dual_branch_cnn --model dual_branch_cnn --epochs 10 --patience 3 --batch-size 32 --image-size 128 --learning-rate 0.001 --device cpu --seed 42
```

The controlled run completed in about 8.8 minutes.

## Results

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V ridge image features | 9.0564 | 9.4635 | 9.4726 |
| Phase 3D simple CNN best run | 9.4665 | 9.3970 | 9.5814 |
| Phase 3E dual-branch CNN | 9.4612 | 9.3807 | 9.5759 |

Dual-branch CNN result:

- Best epoch: `3`
- Early stopping: triggered after `6` completed epochs
- Train MAE: `9.4612`
- Val MAE: `9.3807`
- Test MAE: `9.5759`

## Interpretation

The dual-branch CNN slightly improved validation MAE and beat the Phase 3D simple CNN test MAE by `0.0054`, but it did not beat the Phase 2V ridge baseline. Compared with ridge, the dual-branch CNN test MAE was worse by `0.1033`.

Per-target comparison showed the dual model was best on `waist_cm`, `shoulder_cm`, and `neck_cm`, but ridge remained better on larger high-error targets including `height_cm`, `weight_kg`, `chest_cm`, `hip_cm`, and `thigh_cm`.

## Recommendation

Keep Phase 2V ridge as the current best baseline. The dual-branch architecture is useful infrastructure, but the next phase should focus on stronger image signal or a more meaningful model architecture change rather than another small training-settings tweak.
