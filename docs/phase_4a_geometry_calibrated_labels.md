# Phase 4A Geometry-Calibrated Labels

Phase 4A tested whether the chest, waist, hip, and thigh error ceiling comes from synthetic formula labels that are not calibrated tightly enough to the rendered geometry.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Samples: 1000
- Manifest split: 800 train / 100 val / 100 test
- Geometry utilities: Phase 3X/3Y localized front/side band proxies
- Ambiguity reference: `artifacts/phase_3z_label_geometry_ambiguity/ambiguity_scores.csv`
- Feature extractor for model benchmark: `silhouette_geometry_v5_hybrid`
- Model feature group: `raw_scale_camera`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

No images were rendered and no larger dataset was generated.

## Calibration Method

For each target, Phase 4A fit a train-split-only ridge calibration from localized geometry proxies to the original synthetic label:

- front band width
- side band depth/width
- front/side combined proxies
- ellipse/circumference proxies
- local area proxies
- `height_cm` as a real-world scale anchor

The fitted calibration was then applied to train, validation, and test samples to produce:

- `original_<target>_cm`
- `calibrated_<target>_cm`
- `blended_<target>_cm`

The blended label used 70% original label and 30% calibrated label.

This is a diagnostic label set. It is intended to test whether geometry-consistent labels reduce the error ceiling, not to claim real-world measurement accuracy.

## Artifacts

Local artifacts were written under `artifacts/phase_4a_geometry_calibrated_labels/`:

- `calibrated_labels.csv`
- `label_delta_summary.csv`
- `label_delta_summary.md`
- `calibrated_benchmark_results.json`
- `calibrated_benchmark_results.csv`
- `per_target_calibrated_results.csv`
- `geometry_calibration_summary.md`

These artifacts are generated locally and are not committed.

## Label Corrections

| Target | Mean Abs Delta | P90 Abs Delta | Max Abs Delta | Ambiguous Mean Abs Delta | Clean Mean Abs Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| chest_cm | 5.5232 | 11.2279 | 22.1040 | 6.6764 | 4.6847 |
| waist_cm | 6.0478 | 12.3605 | 26.1564 | 8.1272 | 4.5358 |
| hip_cm | 6.8801 | 14.5250 | 29.8470 | 9.1669 | 5.2172 |
| thigh_cm | 5.8316 | 11.5480 | 22.0163 | 7.0052 | 4.9782 |

Corrections are larger on Phase 3Z ambiguous samples, especially for waist and hip. That supports the Phase 3Z diagnosis: label/geometry collisions are a real source of model error.

## Ambiguity Score Shift

Using the same geometry-neighbor ambiguity score:

- Original group p85 ambiguity score: 0.4603
- Calibrated group p85 ambiguity score: 0.2308
- Group p85 delta: -0.2295
- Original group median ambiguity score: 0.3470
- Calibrated group median ambiguity score: 0.1761

The absolute ambiguous sample count is not directly comparable because each label set uses percentile thresholds, but the score distribution drops sharply after calibration.

## Benchmark Results

| Label Variant | Model | Test Group MAE | Worst Target | Best Target |
| --- | --- | ---: | --- | --- |
| calibrated_labels | GradientBoosting | 1.6777 | hip_cm | thigh_cm |
| calibrated_labels | RandomForest | 1.7158 | hip_cm | thigh_cm |
| calibrated_labels | Ridge | 1.8101 | hip_cm | thigh_cm |
| calibrated_labels | ElasticNet | 1.8131 | hip_cm | thigh_cm |
| blended_labels | GradientBoosting | 4.2238 | hip_cm | waist_cm |
| blended_labels | Ridge | 4.2465 | hip_cm | chest_cm |
| blended_labels | ElasticNet | 4.2473 | hip_cm | chest_cm |
| blended_labels | RandomForest | 4.3384 | hip_cm | chest_cm |
| original_labels | GradientBoosting | 5.9333 | hip_cm | waist_cm |
| original_labels | ElasticNet | 6.0127 | hip_cm | chest_cm |
| original_labels | Ridge | 6.0144 | hip_cm | chest_cm |
| original_labels | RandomForest | 6.1199 | hip_cm | thigh_cm |

Best run:

- `calibrated_labels__raw_scale_camera__target_specific__gradient_boosting`
- Test group MAE: 1.6777
- Gate on synthetic calibrated labels: stronger candidate

Per-target MAE for the best calibrated-label run:

| Target | MAE |
| --- | ---: |
| chest_cm | 1.5284 |
| waist_cm | 1.7867 |
| hip_cm | 1.9304 |
| thigh_cm | 1.4653 |

The blended-label run also crossed below 5 cm, with best group MAE 4.2238.

## Interpretation

Phase 4A strongly supports the generator/label diagnosis:

- The same rendered images and same model family perform far better when labels are calibrated to visible geometry.
- Geometry-calibrated labels reduce ambiguity scores.
- Blended labels improve substantially over original labels, suggesting even partial calibration helps.
- The original formula labels are likely noisier than the visible geometry for chest, waist, hip, and thigh.

This does not mean the project has production-grade real-world measurements yet. It means the synthetic supervision signal is a bottleneck, and better geometry-derived labels could unlock much lower model error.

## Recommendation

The next phase should move calibration into the synthetic generation pipeline:

- Add renderer-side measurement probes for chest, waist, hip, and thigh.
- Save both sampled formula labels and measured geometry labels.
- Prefer measured geometry labels for silhouette-learnable targets.
- Re-run a controlled dataset benchmark with geometry-derived labels generated at render time.

Do not scale a new dataset until the renderer can write geometry-calibrated labels directly and consistently.
