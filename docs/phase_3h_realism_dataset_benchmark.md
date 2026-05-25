# Phase 3H Realism Dataset Benchmark

## Summary

Phase 3H generated a 1000-sample realism-enabled synthetic dataset and benchmarked the existing ridge image-feature baseline and augmented dual-branch CNN. No renderer architecture changes, model architecture changes, or large follow-up dataset generation were done in this phase.

The realism-enabled ridge benchmark is now the strongest observed result.

## Dataset

- Dataset path: `data/synthetic/phase_3h`
- Config used: `synthetic/blender/configs/phase_3g_render_realism_config.example.json`
- Samples: 1000
- Render command used checkpoint-safe `--resume`
- Render time: `6998` seconds, about `1h 56m 38s`
- Blender log: `artifacts/logs/phase_3h_blender_render.log`
- Blender stderr contained only the known `Material.use_nodes` deprecation warning

Validation:

| Field | Value |
| --- | ---: |
| Valid | True |
| Samples complete | 1000 |
| Front PNGs | 1000 |
| Side PNGs | 1000 |
| Label rows | 1000 |

Manifest split:

| Split | Count |
| --- | ---: |
| train | 800 |
| val | 100 |
| test | 100 |

Variation audit:

- Samples: 1000
- Missing measurement fields: 0
- Non-numeric measurement values: 0
- Warnings: none

## Benchmarks

| Run | Dataset | Train MAE | Val MAE | Test MAE |
| --- | --- | ---: | ---: | ---: |
| Phase 2V ridge image features | `data/synthetic/phase_2v` | 9.0564 | 9.4635 | 9.4726 |
| Phase 3H ridge image features | `data/synthetic/phase_3h` | 6.1539 | 6.6519 | 6.8022 |
| Phase 3H augmented dual-branch CNN | `data/synthetic/phase_3h` | 8.6169 | 8.7007 | 8.9501 |

Phase 3H ridge command:

```powershell
python -m training.experiments.run_image_feature_experiment --dataset data/synthetic/phase_3h --output artifacts/experiments/phase_3h_ridge_image_features --model ridge
```

Phase 3H CNN command:

```powershell
python -m training.deep.train_front_side_cnn --dataset data/synthetic/phase_3h --output artifacts/deep/phase_3h_dual_branch_augmented --model dual_branch_cnn --epochs 10 --patience 3 --batch-size 32 --image-size 128 --learning-rate 0.001 --device cpu --seed 42 --augment --brightness-jitter 0.08 --contrast-jitter 0.08 --shift-pixels 4 --noise-std 0.01
```

CNN details:

- Best epoch: 6
- Early stopping: triggered after 9 completed epochs
- Test MAE: 8.9501

## Interpretation

Render realism helped both benchmark families compared with their previous best observed runs:

- Ridge improved from Phase 2V `9.4726` test MAE to Phase 3H `6.8022`.
- Augmented dual-branch CNN improved from Phase 3F `9.5734` test MAE to Phase 3H `8.9501`.

The ridge image-feature baseline still beats the CNN on the same Phase 3H split by `2.1479` test MAE. Phase 3H ridge also improves over the previous current-best Phase 2V ridge by `2.6704` test MAE.

Per-target results show Phase 3H ridge wins most targets. Phase 2V ridge still wins `inseam_cm` and `sleeve_cm`, so those remain worth watching.

One caveat: this is a full new dataset benchmark, not a perfectly isolated image-only ablation. The renderer currently uses one RNG stream for sample parameters and render realism, so enabling realism can also change later sample parameter draws. The variation audit found no warnings, but a future strict ablation should separate parameter RNG from render-realism RNG.

## Visual Spot-Check

Spot-checked:

- `sample_000001_front.png` and `sample_000001_side.png`
- `sample_000500_front.png` and `sample_000500_side.png`
- `sample_001000_front.png` and `sample_001000_side.png`

Notes:

- Front views remain full-body and front-facing.
- Side views remain full-body true profile views.
- Render realism appears as darker/variable background, lighting, and skin-tone differences.
- No obvious cropping or label/file compatibility issues were observed.

## Recommendation

Treat Phase 3H ridge image features as the new best benchmark result for now. Before scaling further or replacing the model registry, run a small follow-up phase to make parameter sampling and render-realism RNG streams independent, then regenerate a strict matched-control ablation if needed. For deep learning, the next step should focus on a more capable training setup or architecture because realism alone improved CNN results but did not close the gap to ridge.
