# Phase 3Z Label-Geometry Ambiguity Filtering

Phase 3Z tested whether label-to-geometry collisions are holding chest, waist, hip, and thigh above the assisted-measurement threshold.

## Inputs

- Dataset: `data/synthetic/phase_3t`
- Samples: 1000
- Existing label/geometry audit: `artifacts/phase_3y_label_geometry_alignment`
- Feature extractor: `silhouette_geometry_v5_hybrid`
- Feature group benchmarked: `raw_scale_camera`
- Targets: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

Phase 3Y showed that these labels generally move visible geometry in the right direction, but similar silhouettes can still have meaningfully different labels. Phase 3Z converts that finding into sample-level ambiguity scores and filtered train/test variants.

## Outputs

Local artifacts were written under `artifacts/phase_3z_label_geometry_ambiguity/`:

- `ambiguity_scores.csv`
- `filtered_manifest_clean_train.csv`
- `filtered_manifest_clean_all.csv`
- `ambiguous_pairs.csv`
- `clean_subset_benchmark_results.json`
- `clean_subset_benchmark_results.csv`
- `per_target_clean_subset_results.csv`
- `ambiguity_summary.md`

These artifacts are generated locally and are not committed.

## Ambiguity Filtering

The filter marked samples as ambiguous when nearby localized geometry proxies had unusually different labels.

- Ambiguous samples: 421 / 1000
- Ambiguous sample rate: 42.1%
- Clean train rows after filtering: 451
- Clean test rows after filtering: 65

This is intentionally a diagnostic filter, not a production dataset definition. The union across four target-specific ambiguity flags is broad, which helps estimate an upper bound but removes a large portion of the dataset.

## Benchmark Results

| Variant | Model | Train Count | Test Count | Test Group MAE |
| --- | --- | ---: | ---: | ---: |
| clean_train_clean_test | RandomForest | 451 | 65 | 5.3305 |
| clean_train_clean_test | ElasticNet | 451 | 65 | 5.3321 |
| clean_train_clean_test | Ridge | 451 | 65 | 5.3370 |
| clean_train_clean_test | GradientBoosting | 451 | 65 | 5.3757 |
| full_baseline | GradientBoosting | 800 | 100 | 5.9333 |
| clean_train_only | ElasticNet | 451 | 100 | 6.0043 |
| clean_train_only | Ridge | 451 | 100 | 6.0085 |
| full_baseline | ElasticNet | 800 | 100 | 6.0127 |
| full_baseline | Ridge | 800 | 100 | 6.0144 |
| clean_train_only | RandomForest | 451 | 100 | 6.0484 |
| ambiguous_only_eval | GradientBoosting | 800 | 35 | 7.2121 |

The best diagnostic upper-bound run was:

- `clean_train_clean_test__raw_scale_camera__target_specific__random_forest`
- Test group MAE: 5.3305
- Gate: research-only

Per-target MAE for that best run:

| Target | MAE | Gate |
| --- | ---: | --- |
| chest_cm | 5.5375 | research-only |
| waist_cm | 5.4384 | research-only |
| hip_cm | 5.2624 | research-only |
| thigh_cm | 5.0835 | research-only |

No target moved below 5 cm, though thigh came closest.

## Interpretation

Filtering ambiguous samples helps the diagnostic upper bound: the best clean-train/clean-test result is substantially better than the full four-target baseline. However, filtering only train data while keeping the original test split does not improve enough, and ambiguous-only evaluation is much worse.

This suggests label-geometry ambiguity is a real ceiling, but simply removing ambiguous training rows is not enough. The remaining clean subset still stays above the assisted-measurement threshold, and the filtered dataset becomes much smaller.

The likely bottleneck remains the synthetic generator and label process:

- Some labels are correlated with visible geometry, but not uniquely enough.
- Chest, waist, hip, and thigh can collide in silhouette space.
- Hip and thigh remain especially vulnerable to overlapping local geometry.
- Labels should probably be checked against renderer-side or mesh-derived measurement probes instead of relying only on sampled formula values.

## Recommendation

Do not scale another large dataset yet. The next phase should improve the generator/label alignment:

- Add mesh or silhouette measurement probes for chest, waist, hip, and thigh during rendering.
- Compare sampled labels to measured rendered geometry per sample.
- Consider writing labels from measured geometry when possible.
- Strengthen independent local deformations so target changes are visually separable.

The current result does not replace the Phase 3L clean ridge baseline. It confirms that reducing label-geometry collisions is the right next data-generation problem to solve.
