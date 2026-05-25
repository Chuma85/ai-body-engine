# Phase 3C Controlled CNN Training

Dataset: `data/synthetic/phase_2v`

Deep run: `artifacts/deep/phase_3c_cnn_controlled`

Baseline comparison: `artifacts/analysis/phase_3c_deep_vs_baseline`

## Goal

Phase 3C improves the deep CNN training loop so it can run a controlled, reproducible CPU experiment on the full synthetic split. This phase adds target normalization, deterministic seed handling, validation-based early stopping, best-checkpoint restore, and richer training diagnostics.

Phase 2V regular ridge remains the benchmark to beat:

```text
Phase 2V ridge test MAE: 9.4726
```

## Training Command

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3c_cnn_controlled --epochs 10 --patience 3 --batch-size 32 --image-size 128 --learning-rate 0.001 --device cpu --seed 42
```

Runtime: about 12.2 minutes on CPU.

## Settings

| Setting | Value |
| --- | --- |
| model | `simple_front_side_cnn` |
| device | `cpu` |
| train samples | 800 |
| val samples | 100 |
| test samples | 100 |
| image size | 128 |
| batch size | 32 |
| learning rate | 0.001 |
| weight decay | 0.0 |
| max epochs | 10 |
| patience | 3 |
| seed | 42 |
| target normalization | enabled |

Target normalization uses train-set means and standard deviations, then inverse-transforms predictions back into original measurement units before metrics and CSV output.

## Training Result

Early stopping triggered after epoch 9. The best validation checkpoint was epoch 6.

| Metric | Value |
| --- | ---: |
| best epoch | 6 |
| epochs completed | 9 |
| best val MAE | 9.3970 |
| train MAE | 9.4665 |
| val MAE | 9.3970 |
| test MAE | 9.5814 |

Per-target test MAE:

| Target | Test MAE |
| --- | ---: |
| height_cm | 14.5033 |
| weight_kg | 16.4737 |
| chest_cm | 11.8798 |
| waist_cm | 13.3370 |
| hip_cm | 11.0381 |
| shoulder_cm | 6.2775 |
| inseam_cm | 7.0559 |
| sleeve_cm | 6.0391 |
| neck_cm | 5.1742 |
| thigh_cm | 8.3031 |
| calf_cm | 5.3133 |

## Baseline Comparison

Command:

```powershell
python -m training.analyze_baseline_errors --runs artifacts/experiments/phase_2v_image_features artifacts/deep/phase_3c_cnn_controlled --output artifacts/analysis/phase_3c_deep_vs_baseline
```

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V regular ridge | 9.0564 | 9.4635 | 9.4726 |
| Phase 3C controlled CNN | 9.4665 | 9.3970 | 9.5814 |

The CNN improved validation MAE but did not beat ridge on test. Test MAE regressed by `0.1087`.

Targets improved by the CNN:

| Target | Ridge Test MAE | CNN Test MAE | Gain |
| --- | ---: | ---: | ---: |
| shoulder_cm | 6.5135 | 6.2775 | 0.2360 |
| neck_cm | 5.3283 | 5.1742 | 0.1541 |
| calf_cm | 5.4193 | 5.3133 | 0.1060 |
| inseam_cm | 7.1232 | 7.0559 | 0.0674 |
| waist_cm | 13.3796 | 13.3370 | 0.0426 |
| sleeve_cm | 6.0523 | 6.0391 | 0.0133 |

Largest regressions:

| Target | Ridge Test MAE | CNN Test MAE | Regression |
| --- | ---: | ---: | ---: |
| hip_cm | 10.4903 | 11.0381 | 0.5477 |
| thigh_cm | 7.8219 | 8.3031 | 0.4811 |
| weight_kg | 16.0824 | 16.4737 | 0.3913 |
| chest_cm | 11.6222 | 11.8798 | 0.2577 |
| height_cm | 14.3656 | 14.5033 | 0.1377 |

## Output Compatibility

The run writes analyzer-compatible artifacts:

- `config.json`
- `metrics.json`
- `predictions_train.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `per_target_errors.json`
- `target_names.json`
- `model.pt`

`model.pt` stores the best validation checkpoint, not the final epoch.

## Recommendation

Do not replace Phase 2V ridge yet. The Phase 3C CNN is now reliable enough for real comparison, but it is not yet better overall.

Next phase should focus on CNN modeling capacity or input signal rather than just longer training:

- add a slightly stronger encoder or separate front/side encoders
- add image augmentation only if it matches the synthetic domain
- inspect prediction rows for `hip_cm`, `thigh_cm`, and `weight_kg`
- consider training longer only after improving the architecture or loss weighting

The current best remains Phase 2V regular ridge with test MAE `9.4726`.
