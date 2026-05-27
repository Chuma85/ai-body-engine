# Phase 3T 1000 Realistic Baseline vs CNN

Phase 3T generated and benchmarked a new 1000-sample realistic synthetic dataset to test whether a larger controlled-realism dataset and a CNN baseline could beat the selected Phase 3S classical candidates.

## Dataset

- Dataset: `data/synthetic/phase_3t`
- Config: `synthetic/blender/configs/phase_3t_realistic_1000_config.example.json`
- Body seed: `530042`
- Render seed: `530314`
- Render realism: enabled
- Sample count: 1000
- Manifest split: 800 train / 100 val / 100 test

Rendering used checkpoint-safe `--resume`.

- First Blender pass: crashed after saving `sample_000547_front.png`; labels were complete through `sample_000546`.
- Resume pass: skipped 546 completed samples and rendered/checkpointed the remaining 454 samples through `sample_001000`.
- Render log wall time: about 116.9 minutes total, split across 64.4 minutes before the crash and 52.5 minutes for resume.

## Validation

- Validation: `True`
- Front PNGs: 1000
- Side PNGs: 1000
- Label rows: 1000
- Manifest rows: 1000
- Variation audit warnings: none

Key measurement ranges were plausible for the synthetic domain:

| Target | Min | Max | Mean | Std |
| --- | ---: | ---: | ---: | ---: |
| height_cm | 150.30 | 204.90 | 176.94 | 15.38 |
| weight_kg | 45.00 | 130.00 | 82.20 | 20.79 |
| chest_cm | 75.20 | 129.90 | 103.14 | 13.90 |
| waist_cm | 55.00 | 124.90 | 84.46 | 16.42 |
| hip_cm | 75.20 | 135.00 | 103.06 | 14.31 |

## Benchmarks

Classical benchmarks used the Phase 3S selected feature/model candidates plus nearby comparison rows. The CNN benchmark used the existing augmented dual-branch model on CPU with 10 max epochs, patience 3, image size 128, batch size 32, learning rate 0.001, and seed 42.

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 3L clean ridge reference | - | - | 6.5780 |
| Phase 3T raw_scale_camera + ridge | 7.1528 | 6.9800 | 7.0196 |
| Phase 3T raw_scale_camera + ElasticNet | 7.1559 | 6.9803 | 7.0239 |
| Phase 3T raw_scale_camera + GradientBoosting | 5.4782 | 7.0721 | 7.0709 |
| Phase 3S background_only + raw_scale_camera + ElasticNet reference | - | - | 7.0834 |
| Phase 3S camera_jitter + hybrid_without_area + RandomForest reference | - | - | 7.1666 |
| Phase 3T combined_hybrid + RandomForest | 5.1744 | 7.1064 | 7.2059 |
| Phase 3T dual-branch CNN augmented | 8.2574 | 7.8774 | 8.1678 |

The best Phase 3T model was `raw_scale_camera + ridge` at test MAE 7.0196. It did not beat the current Phase 3L clean ridge benchmark of 6.5780. The CNN completed a full train/val/test run, but it also did not beat the classical candidates.

## Per-Target Notes

Best Phase 3T per-target winners varied by model:

| Target | Best Test MAE | Best Run |
| --- | ---: | --- |
| shoulder_cm | 3.2750 | combined_hybrid + GradientBoosting |
| calf_cm | 4.3886 | raw_scale_camera + RandomForest |
| neck_cm | 5.1289 | raw_scale_camera + ridge |
| chest_cm | 5.4918 | raw_scale_camera + ridge |
| waist_cm | 5.6725 | raw_scale_camera + GradientBoosting |
| thigh_cm | 5.7530 | raw_scale_camera + ElasticNet |
| sleeve_cm | 6.0560 | raw_scale_camera + ridge |
| hip_cm | 6.1704 | raw_scale_camera + GradientBoosting |
| inseam_cm | 7.4492 | hybrid_without_area + GradientBoosting |
| weight_kg | 13.2513 | raw_scale_camera + ridge |
| height_cm | 13.2877 | raw_scale_camera + ridge |

Height and weight remain the hardest targets in this experiment. The image-derived classical features still rely heavily on scale cues, and the CNN has not yet learned stronger target-specific signal from the rendered pixels.

## Production Readiness

The best Phase 3T result remains in the research-only range.

Provisional gates:

- Research only: >5 cm MAE.
- Assisted sizing/manual confirmation: 3-5 cm MAE.
- Stronger production candidate: 1-3 cm MAE on key targets.

The current 6-8 cm MAE range is useful for benchmark direction, but it is not production-ready for final tailoring measurements.

## Recommendation

Phase 3T does not replace the current best Phase 3L clean ridge baseline. The next phase should focus on improving the learning signal rather than simply scaling the same setup: either add stronger synthetic supervision/camera calibration metadata, introduce target-specific hybrid models that combine labels/metadata with image features, or improve the CNN training objective/architecture with a clear plan for height and weight.
