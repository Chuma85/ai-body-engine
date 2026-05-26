# Phase 3N Render Realism Ablation

Phase 3N ran a smaller same-body render-realism ablation to isolate which render controls help or hurt the ridge image-feature baseline.

## Design

All ablation datasets used:

- `body_seed`: 42
- `render_seed`: 314159
- sample count: 300
- split: 240 train / 30 val / 30 test
- image size: 640 x 896
- same sample IDs
- same body measurement ranges
- same body-shape variation profiles

Only one render-control group was changed at a time:

| Ablation | Render Change |
| --- | --- |
| `clean_baseline` | realism disabled |
| `background_only` | background brightness/color variation |
| `lighting_only` | lighting strength realism multiplier |
| `camera_jitter_only` | camera distance/scale/offset jitter |
| `skin_tone_only` | skin/material brightness variation |
| `combined_realism` | background + lighting + camera + material variation |

The configs are tracked under `synthetic/blender/configs/phase_3n_*_config.example.json`.

## Rendering

All datasets validated after rendering.

| Ablation | Render Time | Notes |
| --- | ---: | --- |
| `clean_baseline` | ~2,183s | completed cleanly |
| `background_only` | ~2,039s | completed cleanly |
| `lighting_only` | ~2,081s | completed cleanly |
| `camera_jitter_only` | ~2,430s | completed cleanly |
| `skin_tone_only` | ~5,966s | crashed at 226 samples, checkpoint-safe resume completed to 300 |
| `combined_realism` | ~2,150s | completed cleanly |

Final validation result for every ablation:

```text
Valid: True
Samples complete: 300
Front PNGs: 300
Side PNGs: 300
Label rows: 300
```

All manifests were built successfully with 240/30/30 train/val/test splits. Variation audits completed with 300 samples and no warnings for all six datasets.

## Same-Body Verification

Measurement labels were compared against `clean_baseline` across all ablations:

| Ablation | Measurement Rows Match | Mismatch Count | Checked Front Hash Differences |
| --- | --- | ---: | ---: |
| `clean_baseline` | true | 0 | 0 |
| `background_only` | true | 0 | 3 |
| `lighting_only` | true | 0 | 3 |
| `camera_jitter_only` | true | 0 | 3 |
| `skin_tone_only` | true | 0 | 3 |
| `combined_realism` | true | 0 | 3 |

This confirms the ablation is controlled: body labels stayed fixed while enabled render controls changed image pixels.

## Ridge Results

| Ablation | Train MAE | Val MAE | Test MAE | Delta vs Clean | Effect |
| --- | ---: | ---: | ---: | ---: | --- |
| `clean_baseline` | 5.4672 | 7.2295 | 7.4657 | 0.0000 | matched |
| `background_only` | 5.4067 | 7.5786 | 7.6893 | 0.2237 | hurt |
| `lighting_only` | 5.4038 | 7.4785 | 7.4462 | -0.0195 | helped slightly |
| `camera_jitter_only` | 5.6692 | 7.0283 | 7.2621 | -0.2036 | helped |
| `skin_tone_only` | 5.4301 | 7.6849 | 7.5653 | 0.0996 | hurt |
| `combined_realism` | 5.7233 | 7.0282 | 7.3070 | -0.1587 | helped |

Best small ablation by test MAE:

- `camera_jitter_only`: 7.2621

Worst small ablation by test MAE:

- `background_only`: 7.6893

## Current Best Comparison

The current official best remains Phase 3L clean ridge:

- Phase 3L clean ridge test MAE: 6.5780
- best Phase 3N small ablation test MAE: 7.2621

No Phase 3N 300-sample ablation beats the current best. Because Phase 3N uses only 300 samples, these results should guide the next controlled dataset design rather than replace the full 1000-sample benchmark.

## Recommendation

The small ablation suggests camera jitter may help ridge generalization, while background variation and skin/material brightness variation hurt the current silhouette feature extractor. The next useful phase is a larger same-body dataset that scales only the promising controls:

- clean baseline
- camera jitter only
- combined realism without background color jitter

Keep Phase 3L clean ridge as the current best benchmark until a full same-body run beats 6.5780 test MAE.
