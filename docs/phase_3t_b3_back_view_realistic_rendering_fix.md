# Phase 3T-B3: Back-View Realistic Rendering Fix

## Problem

Phase 3T-B2 created a repeatable `data/synthetic/phase_3t_enhanced` generation path, but the default wrapper used the lightweight Python silhouette generator. That output is useful for tiny smoke tests only. It is too crude and blocky for model training quality, especially for back-view morphology and garment back-fit experiments.

Poor-quality local images under `data/synthetic/phase_3t_enhanced` should be treated as smoke-only local artifacts and must not be committed as training assets.

## Correct Target

The real enhanced dataset must use the realistic Blender/body mesh rendering path.

- `data/synthetic/phase_3t` remains the legacy front/side dataset.
- `data/synthetic/phase_3t_enhanced` is the enhanced front/side/back target.
- Front, side, and back views for each `sample_id` must come from the same body parameters, morphology, pose, lighting, skin tone, and label row.
- Back images should show a rear body mesh with shoulder/back/waist/hip contour, material, and lighting consistency.
- No real-world accuracy improvement is claimed in this phase.

## Dataset Policy

- Commit generator scripts, configs, validation/audit logic, docs, and tests.
- Do not commit large generated image folders unless explicitly approved.
- Do not commit smoke/lightweight placeholder output as model-training data.
- Keep `data/synthetic/phase_3t_enhanced/` ignored as a local generated artifact path.

## Quality Gate

Enhanced training-candidate datasets should include metadata such as:

- `renderer_mode`
- `render_source`
- `is_smoke_dataset`
- `is_training_candidate`
- `quality_tier`

Expected realistic values:

- `render_source=blender_body_mesh`
- `is_smoke_dataset=false`
- `is_training_candidate=true`
- `quality_tier=training_candidate`

Smoke/lightweight values:

- `renderer_mode=lightweight_smoke`
- `render_source=python_silhouette_placeholder`
- `is_smoke_dataset=true`
- `is_training_candidate=false`
- `quality_tier=smoke_only`

Validation can require realistic training-candidate metadata with:

```powershell
python -m synthetic.validate_synthetic_dataset --dataset data\synthetic\phase_3t_enhanced --require-back --require-realistic
python -m synthetic.build_dataset_manifest --dataset data\synthetic\phase_3t_enhanced --require-back --require-realistic
```

## Correct Generation Commands

Realistic Blender path for the enhanced dataset:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --overwrite
```

Dry-run the Blender command without rendering:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --dry-run
```

Direct Blender command:

```powershell
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json --output data/synthetic/phase_3t_enhanced --num-samples 1000
```

Tiny smoke-only placeholder output:

```powershell
python scripts\generate_phase_3t_enhanced_back_view.py --smoke-lightweight --output-dir $env:TEMP\ai-body-engine-phase-3t-b3-smoke --overwrite
```

Smoke/lightweight output must not be used as the real enhanced training dataset.

## Remaining Scope

This phase fixes routing, config, metadata, and quality gates. It does not retrain models, does not update inference to use back view, and does not validate real-world measurement accuracy.
