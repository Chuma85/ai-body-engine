# Phase 2S Image Feature Experiment Runner

Dataset: `data/synthetic/phase_2q`

Sample count: 500 complete synthetic samples

Split: 400 train / 50 val / 50 test

Phase 2S formalizes the Phase 2P/2R image silhouette regression baseline into a repeatable experiment runner. The model is still the lightweight ridge regressor over deterministic silhouette features; this phase does not add deep learning or change the renderer.

## Command

```powershell
python -m training.experiments.run_image_feature_experiment --dataset data/synthetic/phase_2q --output artifacts/experiments/phase_2s_image_features
```

Optional comparison against the Phase 2R metadata baseline:

```powershell
python -m training.analyze_baseline_errors --runs artifacts/baselines/phase_2r_metadata artifacts/experiments/phase_2s_image_features --output artifacts/analysis/phase_2s
```

## Experiment Output

The experiment output directory contains:

| File | Purpose |
| --- | --- |
| `config.json` | Dataset path, target names, feature extractor metadata, model type, ridge alpha, and run metadata. |
| `metrics.json` | Overall MAE, per-target MAE, split sample counts, target names, and feature count. |
| `predictions_train.csv` | Per-sample train predictions, true values, and absolute errors. |
| `predictions_val.csv` | Per-sample validation predictions, true values, and absolute errors. |
| `predictions_test.csv` | Per-sample test predictions, true values, and absolute errors. |
| `per_target_errors.json` | Per-target error summaries by split. |
| `feature_names.json` | Stable feature names used to train and evaluate the model. |
| `model.json` | Ridge regression artifact with coefficients and feature normalization values. |

Each prediction CSV includes `sample_id`, `split`, and for every measurement target:

```text
true_<target>, pred_<target>, abs_error_<target>
```

These CSVs make it possible to sort by individual target error, inspect outlier samples, and trace weak targets back to exact rendered front/side image pairs.

## Metrics

Feature count: 195

| Split | Overall MAE |
| --- | ---: |
| Train | 8.2487 |
| Val | 9.3730 |
| Test | 9.5982 |

The Phase 2S experiment matches the Phase 2R image-feature baseline test MAE, confirming that the formal runner preserves the same baseline behavior while adding richer outputs.

Compared with the Phase 2R metadata baseline, the image-feature experiment improves test MAE by `1.6144`.

## Per-Target Test MAE

| Target | Test MAE |
| --- | ---: |
| height_cm | 9.5200 |
| weight_kg | 24.0981 |
| chest_cm | 11.8550 |
| waist_cm | 7.5116 |
| hip_cm | 10.2226 |
| shoulder_cm | 4.1036 |
| inseam_cm | 8.3636 |
| sleeve_cm | 6.4653 |
| neck_cm | 5.2677 |
| thigh_cm | 11.8892 |
| calf_cm | 6.2836 |

## Recommendation

Use the Phase 2S runner as the default baseline experiment harness for the next modeling phases. The next phase should inspect the prediction CSV outliers, especially for `weight_kg`, `chest_cm`, and `thigh_cm`, before deciding whether to add more silhouette features, train a stronger non-image baseline, or introduce a first true image regression model.
