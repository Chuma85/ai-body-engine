# Phase 3G Synthetic Render Realism Controls

## Summary

Phase 3G added optional render realism controls to the Blender synthetic renderer. The goal is to prepare future deep-learning datasets with more image variation while keeping the existing clean renderer behavior backward-compatible.

No large dataset was generated and no model training was run in this phase.

## Controls Added

The renderer now supports an optional `render_realism` config block. It is disabled by default for older configs.

Supported controls:

- background brightness variation
- small background color jitter
- lighting strength multiplier variation
- camera distance jitter within safe bounds
- orthographic scale jitter within safe bounds
- small camera lateral and vertical offset jitter
- optional render resolution override
- material/skin-tone brightness variation

The Phase 3G example config is:

```text
synthetic/blender/configs/phase_3g_render_realism_config.example.json
```

The config keeps Phase 2V body-shape variation and adds modest render realism. Its smoke render uses a resolution override of `640x896`.

## Backward Compatibility

Existing configs do not need a `render_realism` block. When omitted, controls default to:

- realism disabled
- no background jitter
- no lighting multiplier jitter
- no camera distance/offset/scale jitter
- no render resolution override
- no skin-tone brightness jitter

The output structure, filenames, and `labels.csv` format remain compatible with the existing validator and manifest pipeline.

## Smoke Result

Command run:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python .\synthetic\blender\scripts\render_parametric_body.py -- --config .\synthetic\blender\configs\phase_3g_render_realism_config.example.json --output .\data\synthetic\phase_3g_smoke --num-samples 5
```

Validation command:

```powershell
python -m synthetic.validate_synthetic_dataset --dataset data/synthetic/phase_3g_smoke
```

Validation result:

| Field | Value |
| --- | ---: |
| Valid | True |
| Samples complete | 5 |
| Front PNGs | 5 |
| Side PNGs | 5 |
| Label rows | 5 |

The validator still emits the expected TODO warning that camera orientation is not automatically verified.

## Visual Spot-Check

Spot-checked:

- `data/synthetic/phase_3g_smoke/images/front/sample_000001_front.png`
- `data/synthetic/phase_3g_smoke/images/side/sample_000001_side.png`

Notes:

- Front view is front-facing and full-body framed.
- Side view is a true side/profile view and full-body framed.
- The darker background and skin-tone/lighting variation render correctly.
- No obvious cropping or filename/label compatibility issues were observed.

## Recommendation

Use the Phase 3G config for the next small controlled dataset generation phase before scaling again. The next phase should generate a modest realism-enabled dataset, validate/audit it, then benchmark the existing ridge and dual-branch CNN workflows against the same split.
