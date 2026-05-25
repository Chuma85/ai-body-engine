# Phase 2V Controlled 1000-Sample Benchmark

Dataset: `data/synthetic/phase_2v`

Config used: `synthetic/blender/configs/phase_2v_controlled_variation_config.example.json`

Resume used: yes

Starting state before retry:

| Item | Count |
| --- | ---: |
| Front PNGs | 887 |
| Side PNGs | 887 |
| Label rows | 0 |

The retry used checkpoint-safe `--resume`. Existing front/side PNG pairs were backfilled into `labels.csv`, and missing samples were rendered through `sample_001000`.

Resume command time: 18.64 minutes. This does not include the earlier failed long render attempts.

## Validation

```text
Valid: True
Samples complete: 1000
Front PNGs: 1000
Side PNGs: 1000
Label rows: 1000
```

Manifest split:

| Split | Count |
| --- | ---: |
| train | 800 |
| val | 100 |
| test | 100 |

## Variation Audit

The variation audit completed with no warnings.

| Measurement | Min | Max | Mean | Std | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| height_cm | 150.20 | 204.80 | 177.47 | 15.42 | 54.60 |
| weight_kg | 45.10 | 129.90 | 82.52 | 20.32 | 84.80 |
| chest_cm | 75.00 | 129.90 | 103.27 | 13.46 | 54.90 |
| waist_cm | 55.10 | 124.80 | 84.54 | 16.11 | 69.70 |
| hip_cm | 75.20 | 134.90 | 103.44 | 13.63 | 59.70 |
| shoulder_cm | 35.10 | 59.90 | 49.02 | 7.15 | 24.80 |
| inseam_cm | 65.00 | 94.80 | 80.03 | 8.84 | 29.80 |
| sleeve_cm | 50.00 | 75.00 | 62.36 | 7.21 | 25.00 |
| neck_cm | 30.00 | 49.90 | 39.91 | 5.73 | 19.90 |
| thigh_cm | 40.10 | 80.00 | 59.55 | 9.92 | 39.90 |
| calf_cm | 28.10 | 54.90 | 41.13 | 6.24 | 26.80 |

Missing measurement fields: 0

Non-numeric measurement values: 0

Low-variation warnings: none

Outlier warnings: none

Correlation warnings: none

## Ridge Experiment

Command:

```powershell
python -m training.experiments.run_image_feature_experiment --dataset data/synthetic/phase_2v --output artifacts/experiments/phase_2v_image_features --model ridge
```

| Split | Overall MAE |
| --- | ---: |
| train | 9.0564 |
| val | 9.4635 |
| test | 9.4726 |

Per-target test MAE:

| Target | Test MAE |
| --- | ---: |
| height_cm | 14.3656 |
| weight_kg | 16.0824 |
| chest_cm | 11.6222 |
| waist_cm | 13.3796 |
| hip_cm | 10.4903 |
| shoulder_cm | 6.5135 |
| inseam_cm | 7.1232 |
| sleeve_cm | 6.0523 |
| neck_cm | 5.3283 |
| thigh_cm | 7.8219 |
| calf_cm | 5.4193 |

## Model Comparison

The optional lightweight model-family comparison was run.

| Model | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| mean | 9.4675 | 9.4013 | 9.5880 |
| ridge | 9.0564 | 9.4635 | 9.4726 |
| knn | 8.3130 | 10.1827 | 10.0452 |

Best overall model: `ridge`, test MAE `9.4726`.

KNN has the lowest train MAE but weaker validation and test MAE, so it still does not beat ridge overall.

## Visual Spot-Check

Checked:

- `sample_000001_front.png` and `sample_000001_side.png`
- `sample_000500_front.png` and `sample_000500_side.png`
- `sample_001000_front.png` and `sample_001000_side.png`

The checked front views are centered, full-body, and front-facing. The checked side views are centered, full-body, and true side/profile views. Arms remain visible and slightly forward in side view, but the body outline is usable for the current silhouette-feature baseline.

## Comparison To Phase 2Q / 2T

Phase 2T on the 500-sample `phase_2q` dataset:

| Model | Test MAE |
| --- | ---: |
| mean | 11.1121 |
| ridge | 9.5982 |
| knn | 9.6068 |

Phase 2V on the controlled 1000-sample dataset:

| Model | Test MAE |
| --- | ---: |
| mean | 9.5880 |
| ridge | 9.4726 |
| knn | 10.0452 |

Ridge improved from `9.5982` to `9.4726`, a modest `0.1256` MAE gain. Controlled scaling helped slightly overall, but the gain is not large enough to suggest that sample count alone solves the current bottleneck. The target-level errors shifted: `weight_kg` improved substantially versus the 500-sample benchmark, while `height_cm` and `waist_cm` worsened.

## Recommendation

Keep ridge as the default lightweight baseline and keep using checkpoint-safe rendering for any larger synthetic jobs. Before moving to deep learning, inspect high-error prediction rows for `height_cm`, `weight_kg`, `waist_cm`, and `chest_cm`, and consider improving pose/profile controls or adding more direct scale/landmark features. The 1000-sample controlled dataset is valid and useful, but the modest ridge gain suggests the next phase should improve visual-label alignment rather than only increasing dataset size.
