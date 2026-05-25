# Phase 3L Same-Body Clean vs Realism Benchmark

Phase 3L used the Phase 3K RNG isolation to run a controlled A/B benchmark where body labels are identical and only render appearance changes.

## Dataset Setup

Both datasets used:

- `body_seed`: 42
- `render_seed`: 314159
- `sample_count`: 1000
- split after manifest generation: 800 train / 100 val / 100 test
- image size: 640 x 896
- same body-parameter ranges and body-shape variation controls

Datasets:

- Clean: `data/synthetic/phase_3l_clean`
- Realism: `data/synthetic/phase_3l_realism`

Configs:

- `synthetic/blender/configs/phase_3l_same_body_clean_config.example.json`
- `synthetic/blender/configs/phase_3l_same_body_realism_config.example.json`

## Rendering And Validation

Clean render:

- render time: about 7,034 seconds, roughly 1h 57m
- validator: valid
- front PNGs: 1000
- side PNGs: 1000
- label rows: 1000

Realism render:

- first render stopped after 843 samples due to a Blender access violation
- checkpoint-safe resume completed the remaining samples
- total render/resume time: about 9,942 seconds, roughly 2h 46m
- validator: valid
- front PNGs: 1000
- side PNGs: 1000
- label rows: 1000

Both variation audits completed with 1000 samples and no warnings.

## Same-Body Check

Measurement-column comparison between clean and realism labels:

```text
MeasurementRowsMatch: True
MismatchCount: 0
CleanRows: 1000
RealismRows: 1000
CleanBodySeed: 42
RealismBodySeed: 42
CleanRealismEnabled: False
RealismEnabled: True
DifferentFrontHashesChecked: 3
```

The checked image hashes differed for `sample_000001`, `sample_000500`, and `sample_001000`, confirming the rendered images changed while the labels stayed fixed.

Visual spot-check of `sample_000001` showed full-body front and side/profile views for both datasets. The realism version showed the expected small framing/appearance differences while preserving the same body.

## Ridge Results

| Run | Train MAE | Val MAE | Test MAE |
| --- | ---: | ---: | ---: |
| Phase 3L clean ridge | 6.0879 | 6.8143 | 6.5780 |
| Phase 3L realism ridge | 6.1974 | 6.7728 | 6.9717 |
| Phase 3H ridge current best before this phase | 6.1539 | 6.6519 | 6.8022 |

The clean same-body ridge run beat the realism same-body ridge run by 0.3937 test MAE. It also beat the previous Phase 3H ridge benchmark by 0.2243 test MAE.

Per-target wins in the clean vs realism comparison:

- Clean won: `height_cm`, `weight_kg`, `waist_cm`, `hip_cm`, `shoulder_cm`, `neck_cm`, `thigh_cm`, `calf_cm`
- Realism won: `chest_cm`, `inseam_cm`, `sleeve_cm`

The largest realism regression was `height_cm`, where test MAE increased from 6.7535 to 10.4315.

## CNN

The optional dual-branch CNN pair was not run in this phase. The render workload was already long, and the primary goal was to isolate the ridge/image-feature effect under matched labels.

## Interpretation

This controlled result does not support the idea that render realism alone improved the ridge baseline. With labels held exactly constant, realism worsened ridge test MAE. The previous Phase 3H improvement over Phase 2V likely came from body-distribution/config differences, image resolution/framing changes, or the interaction of those factors rather than realism alone.

The strongest observed ridge result is now the Phase 3L clean same-body run at 6.5780 test MAE, but it should be registered in a small follow-up registry update before treating it as the official current best.

## Recommendation

Next, update the baseline registry with Phase 3L clean ridge, then run a narrower ablation that separates:

- body seed/body distribution
- image resolution
- camera jitter
- background and lighting realism

That will make it clearer which render/data factors are actually helping the silhouette feature baseline.
