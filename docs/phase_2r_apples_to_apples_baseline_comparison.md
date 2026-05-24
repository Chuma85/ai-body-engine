# Phase 2R Apples-to-Apples Baseline Comparison

Dataset: `data/synthetic/phase_2q`

Sample count: 500 complete synthetic samples

Split: 400 train / 50 val / 50 test

This phase compares the lightweight metadata baseline and the improved image silhouette feature baseline on the same 500-sample dataset split. Earlier phase-to-phase comparisons were useful directional checks, but they mixed different dataset sizes and splits.

## Commands

```powershell
python -m training.train_baseline_measurements --dataset data/synthetic/phase_2q --output artifacts/baselines/phase_2r_metadata
python -m training.train_image_feature_baseline --dataset data/synthetic/phase_2q --output artifacts/baselines/phase_2r_image_features
python -m training.analyze_baseline_errors --runs artifacts/baselines/phase_2r_metadata artifacts/baselines/phase_2r_image_features --output artifacts/analysis/phase_2r
```

## Overall MAE

| Baseline | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Metadata baseline | 10.9636 | 11.0602 | 11.2126 |
| Image silhouette features | 8.2487 | 9.3730 | 9.5982 |

Test MAE delta, metadata minus image features: `1.6144`.

The image silhouette feature baseline improves overall test MAE on the same dataset split.

## Per-Target Test MAE

| Target | Metadata MAE | Image Feature MAE | Winner |
| --- | ---: | ---: | --- |
| height_cm | 12.0315 | 9.5200 | Image features |
| weight_kg | 22.6082 | 24.0981 | Metadata |
| chest_cm | 13.7049 | 11.8550 | Image features |
| waist_cm | 18.7851 | 7.5116 | Image features |
| hip_cm | 13.6705 | 10.2226 | Image features |
| shoulder_cm | 6.1249 | 4.1036 | Image features |
| inseam_cm | 7.6666 | 8.3636 | Metadata |
| sleeve_cm | 6.0028 | 6.4653 | Metadata |
| neck_cm | 5.0945 | 5.2677 | Metadata |
| thigh_cm | 11.3119 | 11.8892 | Metadata |
| calf_cm | 6.3374 | 6.2836 | Image features |

Image features win 6 of 11 targets. Metadata wins 5 of 11 targets.

## Interpretation

On the same 500-sample split, image silhouette features help overall and produce the largest gains for body-width targets that are visually expressed in the front and side silhouettes. The strongest improvements are on `waist_cm`, `hip_cm`, `height_cm`, `shoulder_cm`, and `chest_cm`.

The image baseline is weaker for `weight_kg`, `inseam_cm`, `sleeve_cm`, `neck_cm`, and `thigh_cm`. This is expected for some targets: weight is not directly visible from a silhouette, and sleeve/neck/inseam quality likely needs more explicit pose, landmark, or part-aware features.

## Recommendation

Continue toward an image-based measurement model, but keep same-dataset comparison reports for each new modeling phase. Before moving to a heavier model, the next useful step is to improve target-specific signal for weak measurements, especially `weight_kg`, `thigh_cm`, `sleeve_cm`, and `neck_cm`, or document which of those may require richer labels, landmarks, segmentation, or a learned visual representation.
