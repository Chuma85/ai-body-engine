# Phase 3B Deep Experiment Outputs

Dataset: `data/synthetic/phase_2v`

Deep run: `artifacts/deep/phase_3b_cnn_smoke`

Baseline comparison: `artifacts/analysis/phase_3b_deep_vs_baseline`

## Goal

Phase 3B turns the Phase 3A CNN smoke scaffold into a benchmarkable experiment output format. It still does not attempt long training and does not replace the Phase 2V ridge baseline.

## Run Settings

Command:

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_2v --output artifacts/deep/phase_3b_cnn_smoke --epochs 2 --limit-samples 64 --batch-size 16 --image-size 128 --device cpu --seed 42
```

Settings:

| Setting | Value |
| --- | --- |
| model | `simple_front_side_cnn` |
| device | `cpu` |
| epochs | 2 |
| limit samples | 64 per split |
| batch size | 16 |
| image size | 128 |
| seed | 42 |

## Outputs

The deep trainer now writes standard experiment artifacts:

- `config.json`
- `metrics.json`
- `predictions_train.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `per_target_errors.json`
- `target_names.json`
- `model.pt`

Prediction CSVs include:

- `sample_id`
- `split`
- `true_<target>`
- `pred_<target>`
- `abs_error_<target>`

The metrics format is compatible with `training.analyze_baseline_errors`.

## Smoke Metrics

| Split | Samples | Overall MAE |
| --- | ---: | ---: |
| train | 64 | 9.3309 |
| val | 64 | 9.6483 |
| test | 64 | 9.7302 |

Per-target test MAE:

| Target | Test MAE |
| --- | ---: |
| height_cm | 15.1123 |
| weight_kg | 16.5057 |
| chest_cm | 11.8156 |
| waist_cm | 12.6605 |
| hip_cm | 10.8710 |
| shoulder_cm | 6.4121 |
| inseam_cm | 7.3909 |
| sleeve_cm | 6.3307 |
| neck_cm | 5.5575 |
| thigh_cm | 8.9028 |
| calf_cm | 5.4725 |

## Comparison Against Phase 2V Ridge

Command:

```powershell
python -m training.analyze_baseline_errors --runs artifacts/experiments/phase_2v_image_features artifacts/deep/phase_3b_cnn_smoke --output artifacts/analysis/phase_3b_deep_vs_baseline
```

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 2V regular ridge | 9.0564 | 9.4635 | 9.4726 |
| Phase 3B CNN smoke | 9.3309 | 9.6483 | 9.7302 |

The Phase 3B CNN smoke run does not beat the Phase 2V ridge baseline. Its test MAE is worse by `0.2575`.

Targets improved by the CNN smoke run:

| Target | Phase 2V | Phase 3B | MAE Gain |
| --- | ---: | ---: | ---: |
| waist_cm | 13.3796 | 12.6605 | 0.7191 |
| shoulder_cm | 6.5135 | 6.4121 | 0.1014 |

Largest regressions:

| Target | Phase 2V | Phase 3B | Regression |
| --- | ---: | ---: | ---: |
| thigh_cm | 7.8219 | 8.9028 | 1.0809 |
| height_cm | 14.3656 | 15.1123 | 0.7466 |
| weight_kg | 16.0824 | 16.5057 | 0.4233 |

## Benchmark Readiness

Phase 3B is benchmark-ready in the sense that it writes standard metrics, prediction CSVs, per-target errors, and a checkpoint. It is not yet a serious model benchmark because it uses only 64 samples per split and trains for only two CPU epochs.

Phase 2V regular ridge remains the current best baseline with test MAE `9.4726`.

## Recommendation

Next phase should run a still-controlled but more meaningful deep experiment:

- train on the full 800-sample train split
- keep CPU-safe defaults or explicitly document runtime
- preserve the same prediction and metrics artifacts
- compare against Phase 2V and register the result only if test MAE improves

The deep path is now open and inspectable, but the benchmark to beat remains Phase 2V regular ridge.
