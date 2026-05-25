# Phase 3F Deep Augmentation Benchmark

## Summary

Phase 3F added deterministic, training-only image augmentation controls for deep CNN training. The goal was to test whether modest image perturbations improve generalization for the Phase 3E dual-branch CNN without changing the renderer, dataset, or current best baseline policy.

Phase 2V ridge image features remain the current best benchmark.

## Augmentation Controls

The deep image dataset now supports optional train-only augmentation:

- brightness jitter
- contrast jitter
- horizontal shift with crop/pad
- Gaussian noise
- optional horizontal flip probability
- deterministic seed control

Validation and test splits ignore random augmentation even if augmentation settings are provided.

Command-line controls:

```powershell
--augment
--brightness-jitter
--contrast-jitter
--shift-pixels
--noise-std
--horizontal-flip-prob
```

Defaults keep augmentation disabled for backwards compatibility.

## Dataset And Settings

- Dataset: `data/synthetic/phase_2v`
- Split: 800 train / 100 val / 100 test
- Model: `dual_branch_cnn`
- Image size: `128`
- Batch size: `32`
- Learning rate: `0.001`
- Epochs: `10`
- Patience: `3`
- Device: `cpu`
- Seed: `42`
- Augmentation enabled: `true`
- Brightness jitter: `0.08`
- Contrast jitter: `0.08`
- Shift pixels: `4`
- Noise std: `0.01`
- Horizontal flip probability: `0.0`

Command:

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3f_dual_branch_augmented --model dual_branch_cnn --epochs 10 --patience 3 --batch-size 32 --image-size 128 --learning-rate 0.001 --device cpu --seed 42 --augment --brightness-jitter 0.08 --contrast-jitter 0.08 --shift-pixels 4 --noise-std 0.01
```

The controlled CPU run completed in about 12.1 minutes.

## Results

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V ridge image features | 9.0564 | 9.4635 | 9.4726 |
| Phase 3E dual-branch CNN | 9.4612 | 9.3807 | 9.5759 |
| Phase 3F augmented dual-branch CNN | 9.4559 | 9.3918 | 9.5734 |

Phase 3F augmented result:

- Best epoch: `5`
- Early stopping: triggered after `8` completed epochs
- Train MAE: `9.4559`
- Val MAE: `9.3918`
- Test MAE: `9.5734`

## Interpretation

Augmentation slightly improved test MAE over Phase 3E by `0.0025`, but validation MAE worsened from `9.3807` to `9.3918`. The improvement is too small to treat as a meaningful breakthrough.

Compared with Phase 2V ridge, the augmented dual-branch CNN is still worse by `0.1008` test MAE, so the deep model does not become the current best.

Per-target comparison shows Phase 3F is best among the compared runs for `waist_cm`, `shoulder_cm`, `inseam_cm`, `sleeve_cm`, `neck_cm`, and `calf_cm`, but ridge remains better on larger high-error targets such as `height_cm`, `weight_kg`, `chest_cm`, `hip_cm`, and `thigh_cm`.

## Recommendation

Keep Phase 2V ridge as the current best baseline. The augmentation controls are useful and should remain available, but modest augmentation alone is not enough. The next phase should focus on stronger image signal, target-specific losses/normalization, or a more capable architecture rather than simply increasing augmentation strength.
