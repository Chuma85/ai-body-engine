# Blender Synthetic Pipeline

Phase 2B prepares AI Body Engine for future Blender-based synthetic human rendering.

This is a scaffold only. It does not generate realistic humans yet, does not require SMPL or SMPL-X assets, and does not require Blender for normal Python tests.

## What Phase 2B Adds

- Render configuration schema
- Measurement label schema
- Blender command builder
- Dry-run CLI helper
- A safe-to-import Blender script scaffold
- Tests that validate config and command generation without launching Blender

## Dry Run

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2b_render_config.example.json --dry-run
```

Expected command shape:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2b_render_config.example.json
```

## Future Work

Phase 2C will add actual Blender mesh generation and rendering.

Phase 2D will explore SMPL/SMPL-X integration if licensing, model access, and dataset governance permit it.
