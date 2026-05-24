# Phase 2T Image Feature Model Family Comparison

Dataset: `data/synthetic/phase_2q`

Sample count: 500 complete synthetic samples

Split: 400 train / 50 val / 50 test

Feature count: 195 deterministic front/side silhouette features

Phase 2T compares lightweight CPU-friendly model families using the existing image silhouette features. This phase does not add deep learning, generate new images, or change the Blender renderer.

## Models Compared

| Model | Notes |
| --- | --- |
| `mean` | Predicts train-set target means. Useful as a simple non-visual floor. |
| `ridge` | Existing Phase 2S ridge regression baseline over standardized silhouette features. |
| `knn` | NumPy k-nearest-neighbor regressor over standardized silhouette features, `k=5`. |

Tree-based models such as random forest and gradient boosting were deferred because scikit-learn is not currently a project dependency. No new ML dependency was added in this phase.

## Command

```powershell
python -m training.experiments.compare_image_feature_models --dataset data/synthetic/phase_2q --output artifacts/experiments/phase_2t_model_comparison
```

## Overall MAE

| Model | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| mean | 11.0615 | 11.0423 | 11.1121 |
| ridge | 8.2487 | 9.3730 | 9.5982 |
| knn | 8.1084 | 10.1426 | 9.6068 |

Best overall model: `ridge`, test MAE `9.5982`.

Ridge remains essentially tied with KNN on test MAE, but ridge has the better validation MAE and a smaller train/validation gap. KNN has the lowest train MAE but weaker validation MAE, which suggests mild overfitting or less stable generalization on this split.

## Per-Target Test MAE

| Target | mean | ridge | knn | Best Model |
| --- | ---: | ---: | ---: | --- |
| height_cm | 11.8940 | 9.5200 | 10.9068 | ridge |
| weight_kg | 22.4912 | 24.0981 | 23.3616 | mean |
| chest_cm | 12.9634 | 11.8550 | 11.4184 | knn |
| waist_cm | 18.7448 | 7.5116 | 7.3852 | knn |
| hip_cm | 13.6789 | 10.2226 | 9.8672 | knn |
| shoulder_cm | 6.3033 | 4.1036 | 4.0844 | knn |
| inseam_cm | 7.6390 | 8.3636 | 8.5844 | mean |
| sleeve_cm | 5.9703 | 6.4653 | 6.0568 | mean |
| neck_cm | 5.0765 | 5.2677 | 5.8456 | mean |
| thigh_cm | 11.1740 | 11.8892 | 11.8444 | mean |
| calf_cm | 6.2977 | 6.2836 | 6.3200 | ridge |

Best-model counts:

| Model | Targets Won |
| --- | ---: |
| mean | 5 |
| ridge | 2 |
| knn | 4 |

## Interpretation

Model choice gives useful target-level signal, but it does not improve the overall benchmark beyond the Phase 2S ridge baseline. Ridge remains the best single model overall, with the same test MAE as Phase 2S: `9.5982`.

KNN improves several silhouette-driven measurements, especially `chest_cm`, `waist_cm`, `hip_cm`, and `shoulder_cm`, but its validation MAE is worse than ridge. The mean model winning `weight_kg`, `inseam_cm`, `sleeve_cm`, `neck_cm`, and `thigh_cm` is a sign that the current silhouette features are still weak or noisy for those targets.

Worst remaining targets by best available test MAE are:

| Target | Best Model | Best Test MAE |
| --- | --- | ---: |
| weight_kg | mean | 22.4912 |
| chest_cm | knn | 11.4184 |
| thigh_cm | mean | 11.1740 |
| hip_cm | knn | 9.8672 |
| height_cm | ridge | 9.5200 |

## Recommendation

Use ridge as the default single-model baseline for now, and keep KNN as a useful comparison because it exposes target-level opportunities for shape-width measurements. The next phase should prioritize better synthetic variation and target-specific feature signal, especially for `weight_kg`, `thigh_cm`, `sleeve_cm`, and `neck_cm`, before adding a heavier image model.
