# Phase 3R Select And Regularize Hybrid Features

Phase 3R tested whether feature-group selection and stronger regularization could recover the useful raw scale signal from `silhouette_geometry_v5_hybrid` without letting camera/framing drift dominate the ridge baseline.

## What Changed

Added:

```text
training/experiments/select_regularized_hybrid_features.py
```

The runner benchmarks combinations of:

- feature configs
- lightweight regression model families
- drift-aware feature selection
- per-target diagnostics
- group-level feature importance summaries

Outputs:

- `summary.json`
- `report.md`
- `results.csv`
- `per_target_results.csv`
- `feature_selection.json`
- `model_importance.csv`

## Feature Configs

| Feature Config | Meaning |
| --- | --- |
| `normalized_shape` | only canonical mask geometry features |
| `raw_scale_camera` | only raw bbox, area, scale, offset, and raw front/side scale ratios |
| `combined_hybrid` | all v5 hybrid features |
| `combined_hybrid_without_offsets` | all v5 features except raw crop offsets |
| `combined_hybrid_without_area_ratios` | all v5 features except area-heavy features |
| `selected_low_drift_features` | excludes crop offsets and features above the configured drift threshold |

The Phase 3R run used:

```text
drift CSV: artifacts/analysis/phase_3q_feature_drift/feature_drift.csv
drift threshold: 1.0
datasets: six Phase 3N same-body render ablations
models: ridge, elasticnet, random_forest, gradient_boosting
```

The full grid produced 144 completed runs. The console command reached the tool timeout after writing the output artifacts, so the artifacts were checked directly:

```text
run_count: 144
skipped_runs: 0
datasets: 6
```

## Best Result By Dataset

| Dataset | Best Feature Config | Best Model | Train MAE | Val MAE | Test MAE |
| --- | --- | --- | ---: | ---: | ---: |
| `phase_3n_clean_baseline` | `raw_scale_camera` | ridge | 6.8545 | 7.4692 | 7.1803 |
| `phase_3n_background_only` | `raw_scale_camera` | elasticnet | 6.8565 | 7.3176 | 7.0834 |
| `phase_3n_lighting_only` | `raw_scale_camera` | elasticnet | 6.8556 | 7.3437 | 7.1000 |
| `phase_3n_camera_jitter_only` | `combined_hybrid_without_area_ratios` | random_forest | 4.5065 | 7.4560 | 7.1666 |
| `phase_3n_skin_tone_only` | `raw_scale_camera` | elasticnet | 6.8577 | 7.3656 | 7.1044 |
| `phase_3n_combined_realism` | `raw_scale_camera` | gradient_boosting | 3.2427 | 7.7151 | 7.2811 |

Best Phase 3R result:

```text
phase_3n_background_only raw_scale_camera + elasticnet test MAE: 7.0834
```

No Phase 3R result beats the current best:

```text
Phase 3L clean ridge test MAE: 6.5780
```

## Group And Model Findings

Average test MAE by feature config:

| Feature Config | Avg Test MAE |
| --- | ---: |
| `raw_scale_camera` | 7.3031 |
| `combined_hybrid_without_offsets` | 7.6382 |
| `selected_low_drift_features` | 7.6404 |
| `combined_hybrid` | 7.6796 |
| `normalized_shape` | 7.6881 |
| `combined_hybrid_without_area_ratios` | 7.7297 |

Average test MAE by model:

| Model | Avg Test MAE |
| --- | ---: |
| random_forest | 7.3980 |
| gradient_boosting | 7.5421 |
| ridge | 7.6629 |
| elasticnet | 7.8497 |

Interpretation:

- Raw scale/camera cues are still the strongest small-dataset signal overall.
- Random forest is the most stable model family on average.
- ElasticNet wins some raw-scale-only ablations but is not consistently best.
- Removing crop offsets helps in some combined-feature cases, but not enough to become the best overall.
- Removing area-heavy features helps camera jitter specifically, where `combined_hybrid_without_area_ratios + random_forest` is best for that dataset.

## Per-Target Notes

Best per-target winners vary substantially by dataset and target.

Examples:

- `height_cm` often prefers raw scale/camera features.
- `weight_kg` remains difficult and often prefers normalized-shape ridge or raw-scale tree models.
- `waist_cm` frequently benefits from raw scale with gradient boosting or ElasticNet.
- `inseam_cm` often improves with raw scale or low-drift selected features.
- `neck_cm`, `calf_cm`, and `sleeve_cm` still bounce between feature configs, suggesting current silhouette features remain weak for small localized measurements.

This supports the Phase 3Q conclusion: raw scale signal is useful, but it is not yet stable enough to use directly at larger scale without better camera metadata or stronger feature curation.

## Comparison To Recent Baselines

| Benchmark | Best Test MAE |
| --- | ---: |
| Phase 3L clean ridge, 1000 samples | 6.5780 |
| Phase 3R best small ablation | 7.0834 |
| Phase 3N best small ablation | 7.2621 |
| Phase 3O best small ablation | 7.3958 |
| Phase 3P best small ablation | 7.8601 |
| Phase 3Q best small ablation | 7.8636 |

Phase 3R improves over the recent 300-sample ablation experiments, but it does not beat the full 1000-sample Phase 3L clean ridge benchmark.

## Recommendation

Do not promote Phase 3R as the current best baseline. Keep Phase 3L clean ridge as the benchmark anchor.

The next phase should use Phase 3R’s finding that raw scale cues remain valuable, but make them less brittle:

- add render/camera metadata features so raw scale can be normalized by known camera scale or orthographic scale
- prefer ratio and metadata-normalized scale features over raw pixel area counts
- test a curated raw-scale subset with random forest on a full 1000-sample same-body run only after metadata normalization exists
- avoid scaling combined realism until the area and framing features are controlled

Phase 3R is useful because it shows model regularization and feature selection can recover much of the lost signal, but the current handcrafted feature stack still trails the Phase 3L clean ridge benchmark.
