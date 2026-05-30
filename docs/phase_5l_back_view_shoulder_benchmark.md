# Phase 5L Back-View Shoulder Benchmark

Phase 5L tests whether adding an optional back capture improves shoulder and upper-back measurement targets compared with the existing front+side image feature pipeline.

## Why Back View Was Tested

The app workflow can now collect an optional back view. Shoulder width and across-back fit are important for structured garments such as jackets, blazers, coats, agbada/kaftan, and formalwear, but previous synthetic benchmarks mostly emphasized front+side features and the strongest calibrated targets were chest, waist, hip, and thigh.

This phase keeps back capture optional and tests whether it is worth recommending for shoulder-sensitive garments.

## Pipeline Changes

Shared synthetic dataset plumbing now supports an optional `back_image_path`:

- Blender render configs may include `views: ["front", "side", "back"]`.
- The Blender camera supports a true back view.
- Dataset validation can require back images for benchmark datasets.
- Manifest generation includes `back_image_path` while keeping front+side datasets loadable.
- The dataset loader exposes `back_image_path` when present and leaves existing front+side callers unchanged.

The controlled Phase 5L Blender config is:

- `synthetic/blender/configs/phase_5l_back_view_shoulder_benchmark_config.example.json`

The committed benchmark used the lightweight deterministic silhouette generator in:

- `training/experiments/benchmark_back_view_shoulder.py`

This avoids requiring Blender in CI while still exercising front/side/back sample alignment and back-view feature extraction.

## Features Added

Back-view features are implemented in `training/features/back_view_features.py`:

- back shoulder width proxy,
- across-back width proxy,
- upper-back width proxy,
- upper-back area proxy,
- shoulder slope proxy,
- back torso width bands at shoulder, upper-chest, chest, mid-torso, waist, and hip levels,
- front/back shoulder comparison features,
- front/side/back combined volume and balance proxies.

## Benchmark

Dataset:

- 360 synthetic samples
- front, side, and back images for every sample
- deterministic split: 288 train, 36 validation, 36 test

Targets:

- Shoulder/back group: `shoulder_cm`, `across_back_cm`, `upper_back_cm`
- Reference group: `chest_cm`, `waist_cm`, `hip_cm`, `thigh_cm`

Feature sets:

- front+side baseline
- back-only
- front+side+back combined
- geometry+residual compatible front/side/back variant

Models:

- Ridge
- ElasticNet
- RandomForest
- GradientBoosting

## Results

Best front+side shoulder group MAE:

- `front_side_baseline__elasticnet`: 0.9195 cm

Best front+side+back shoulder group MAE:

- `front_side_back_combined__ridge`: 0.8036 cm

Best back-only shoulder group MAE:

- `back_only__elasticnet`: 1.3316 cm

Observed shoulder group improvement:

- 0.1160 cm
- 12.61%

Reference target impact:

- Reference group MAE delta was +0.0139 cm.
- This does not materially worsen the current strong reference targets in this synthetic benchmark.

Artifacts:

- `artifacts/phase_5l_back_view_shoulder_benchmark/dataset_validation.json`
- `artifacts/phase_5l_back_view_shoulder_benchmark/dataset_validation.csv`
- `artifacts/phase_5l_back_view_shoulder_benchmark/benchmark_results.json`
- `artifacts/phase_5l_back_view_shoulder_benchmark/benchmark_results.csv`
- `artifacts/phase_5l_back_view_shoulder_benchmark/per_target_results.csv`
- `artifacts/phase_5l_back_view_shoulder_benchmark/back_view_feature_summary.md`
- `artifacts/phase_5l_back_view_shoulder_benchmark/recommendation_summary.md`

## Product Recommendation

Back capture should remain optional, not mandatory.

Based on this synthetic benchmark, back capture should be recommended for structured garments and shoulder-sensitive fits because it improved the shoulder/back target group without materially worsening chest, waist, hip, or thigh reference targets.

Do not claim real-world production readiness from this phase. The benchmark is synthetic and intentionally controlled.

## Limitations

- The committed benchmark uses deterministic silhouette renders, not real human scans.
- Across-back and upper-back labels are synthetic diagnostic labels.
- Back-only features did not beat the front+side baseline; the benefit came from combining back with front+side.
- Blender back-view rendering support was added and configured, but this run did not require Blender execution.
- Real-world validation with measured back/shoulder data is still required.

## Next Phase

Use the new Blender back-view support to render a higher-fidelity front/side/back dataset and compare it against this silhouette benchmark. Then validate shoulder and across-back estimates on measured human data before changing product capture requirements.
