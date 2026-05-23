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

## Phase 2C Procedural Rendering

Phase 2C adds a first working procedural body renderer. It uses Blender primitives to create simple mannequin-like bodies:

- UV sphere head
- Cylinder neck, arms, and legs
- Ellipsoid torso, waist, and hips
- Basic skin-tone materials
- Plain white background
- Front and side camera views
- `labels.csv` using the existing measurement schema

This is still an MVP. It does not use SMPL/SMPL-X, does not dress garments, and does not create production-quality try-on renders.

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2c_render_config.example.json --dry-run
```

Actual Blender run:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2c_render_config.example.json
```

Validate generated output:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2c/labels/labels.csv
```

Phase 2D will add better anatomy and mesh refinement.

Later phases will explore garment placeholders, dressing agents, paid high-quality try-on rendering, and SMPL/SMPL-X integration if licensing, model access, and dataset governance permit it.
