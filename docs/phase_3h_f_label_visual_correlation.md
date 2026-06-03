# Phase 3H-F Label Visual Correlation Audit

Phase 3H-F checks whether the Phase 3H-D Blender labels are visually learnable from deterministic front, side, and back image features. It does not train a model, modify the Blender file, or generate a new dataset.

## Why This Audit Is Needed

Phase 3H-E showed that a baseline can learn some signal from:

```text
data/synthetic/phase_3h_blend_250
```

The error is still high for measurement use. A likely cause is that generated measurement labels may not be tightly coupled to the visible rendered body changes. This audit measures that coupling directly by correlating silhouette features against measurement targets.

## Command

```powershell
python scripts\audit_blend_label_visual_correlation.py --dataset data\synthetic\phase_3h_blend_250 --out artifacts\phase_3h_f_label_visual_correlation --target-columns height_cm chest_cm waist_cm hip_cm shoulder_cm inseam_cm --min-abs-correlation 0.25
```

The one-step verifier runs the audit and checks all expected artifacts:

```powershell
python scripts\verify_phase_3h_f_label_visual_correlation.py
```

## Outputs

```text
artifacts/phase_3h_f_label_visual_correlation/
  correlation_report.json
  correlation_summary.md
  feature_label_correlation.csv
  target_correlation_matrix.csv
  visual_feature_summary.csv
  label_summary.csv
  flagged_targets.csv
  top_features_by_target.csv
```

## How To Interpret Results

High absolute correlation means a target has at least one deterministic visual feature that moves with the label. That is a good sign for learnability, though it does not prove the feature is sufficient for accurate prediction.

Low absolute correlation means the label is weakly tied to the visible silhouette features. If a target remains below the threshold, the label generator, shape-key mapping, render variation, or feature extraction should be improved before scaling up.

Suspicious label behavior includes very low label variation, safe-range violations, or measurement targets that move together almost perfectly. Suspicious visual behavior includes near-zero feature variation, front/side/back features that are too similar, or silhouettes that barely change across samples.

## Before Generating 1000+ Samples

Before spending time on a larger dataset, rerun this audit on the 250-sample checkpoint and confirm that target labels have meaningful variation and that each important measurement has visible correlation. Weak targets should be fixed at the dataset or label-generation level first.

## Not A Production Accuracy Test

This audit is synthetic-only and diagnostic. It does not validate real-world capture quality, tailoring fit, or production measurement accuracy.
