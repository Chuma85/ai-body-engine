# Phase 3S Benchmark Leaderboard And Baseline Selection

Phase 3S consolidated the recent Phase 3 benchmark artifacts into a single leaderboard so the next larger-data or CNN phase has clear comparison anchors.

## Consolidation Output

Added:

```text
training/experiments/consolidate_phase3_benchmarks.py
```

The local consolidation command was:

```text
python -m training.experiments.consolidate_phase3_benchmarks --artifacts artifacts --output artifacts/phase_3s_benchmark_leaderboard
```

Generated local artifacts:

- `artifacts/phase_3s_benchmark_leaderboard/leaderboard.json`
- `artifacts/phase_3s_benchmark_leaderboard/leaderboard.csv`
- `artifacts/phase_3s_benchmark_leaderboard/leaderboard.md`
- `artifacts/phase_3s_benchmark_leaderboard/per_target_leaderboard.csv`
- `artifacts/phase_3s_benchmark_leaderboard/candidate_baselines.json`

The generated artifacts are local benchmark outputs and are not committed.

## Artifact Coverage

The consolidation collected:

```text
Leaderboard rows: 170
Per-target rows: 1804
```

Included sources:

- Phase 3L standard clean and realism ridge experiments
- Phase 3N render ablation results
- Phase 3O robust mask feature ablation results
- Phase 3P camera normalization ablation results
- Phase 3Q hybrid scale-aware ablation results
- Phase 3R feature-selection and regularized/nonlinear comparison results

One warning was expected:

```text
No per-target results found for artifacts/analysis/phase_3n_render_ablation/results.csv
```

The Phase 3N overall results were still included.

## Top Overall Leaderboard

| Rank | Phase | Ablation | Feature Version | Feature Group | Model | Test MAE |
| --- | --- | --- | --- | --- | --- | ---: |
| 1 | 3L | clean | `silhouette_geometry_v2` | all_features | ridge | 6.5780 |
| 2 | 3L | realism | `silhouette_geometry_v2` | all_features | ridge | 6.9717 |
| 3 | 3R | background_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | elasticnet | 7.0834 |
| 4 | 3R | background_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | ridge | 7.0954 |
| 5 | 3R | lighting_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | elasticnet | 7.1000 |
| 6 | 3R | skin_tone_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | elasticnet | 7.1044 |
| 7 | 3R | lighting_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | ridge | 7.1044 |
| 8 | 3R | skin_tone_only | `silhouette_geometry_v5_hybrid` | raw_scale_camera | ridge | 7.1095 |
| 9 | 3R | camera_jitter_only | `silhouette_geometry_v5_hybrid` | combined_hybrid_without_area_ratios | random_forest | 7.1666 |
| 10 | 3R | clean_baseline | `silhouette_geometry_v5_hybrid` | raw_scale_camera | ridge | 7.1803 |

## Selected Baseline Candidates

| Candidate | Phase | Ablation | Feature Group | Model | Test MAE |
| --- | --- | --- | --- | --- | ---: |
| Current best clean baseline | 3L | clean | all_features | ridge | 6.5780 |
| Best Phase 3R regularized result | 3R | background_only | raw_scale_camera | elasticnet | 7.0834 |
| Best camera-jitter robust result | 3R | camera_jitter_only | combined_hybrid_without_area_ratios | random_forest | 7.1666 |
| Combined-realism candidate | 3R | combined_realism | raw_scale_camera | gradient_boosting | 7.2811 |

These should be the comparison anchors for the next controlled dataset/model phase.

## Interpretation

What consistently helped:

- Same-body dataset controls made the ablation comparisons trustworthy.
- Phase 3L clean ridge remains the strongest overall benchmark.
- Phase 3R regularized and tree-based models recovered signal lost by pure normalization.
- Raw scale cues are useful for several measurements and small ablation datasets.

What consistently hurt:

- Pure canonical normalization reduced camera/framing drift but removed body-scale signal.
- Direct raw pixel area and crop-offset cues remain unstable under camera jitter.
- Combined realism remains fragile without explicit camera/render metadata normalization.

Why the handcrafted baseline may be plateauing:

- The best 300-sample hand-feature experiments still trail Phase 3L clean ridge.
- The same targets keep surfacing as hard or volatile, especially weight, waist, neck, and small localized limb measurements.
- Raw scale helps, but without camera metadata it can confuse framing changes with body-size changes.

## Recommendation

Do not generate a new large dataset until render/camera metadata is exposed as training features or used to normalize raw scale cues.

Recommended next phase:

- Add camera/render metadata features to manifests or experiment loaders.
- Normalize raw scale features by known render scale/orthographic settings where possible.
- Re-run a controlled 300-sample same-body benchmark with:
  - Phase 3L clean ridge as the benchmark anchor
  - Phase 3R raw-scale ElasticNet candidate
  - Phase 3R camera-jitter random forest candidate
  - Phase 3R combined-realism gradient boosting candidate
- Only then decide whether to scale another 1000-sample dataset or resume CNN work.

Current official best remains:

```text
Phase 3L clean ridge test MAE: 6.5780
```
