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

## Phase 2D Anatomical Refinement

Phase 2D improves the procedural mannequin so the renders are more useful for future measurement-model training:

- Better chest, waist, abdomen, and hip sections
- Body-shape profiles for slim, average, athletic, curvy, broad, and plus
- Limb radii driven by sleeve, inseam, thigh, and calf measurements
- Slight measurement-friendly pose variation
- Render-quality controls for resolution, ambient occlusion, and contact shadows
- Optional metadata columns in generated labels

It still does not use SMPL/SMPL-X, garment dressing, model training, FashionApp integration, or paid try-on rendering.

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2d_render_config.example.json --dry-run
```

Actual Blender run:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2d_render_config.example.json
```

Validate generated output:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2d/labels/labels.csv
```

## Phase 2E Base Human Mesh Rendering

Phase 2E adds base human mesh import support so the synthetic pipeline can move beyond primitive mannequin geometry.

Supported scaffold behavior:

- Resolve mesh assets relative to the repository root
- Import `.glb` / `.gltf`, `.obj`, and `.fbx` files in Blender
- Keep `.blend` append/link support documented for a later phase
- Normalize imported mesh scale
- Center imported mesh on the origin
- Apply generated skin-tone material
- Fall back to the procedural body renderer if the configured mesh is missing and fallback is enabled

Large mesh assets are intentionally ignored by git. Place local assets under:

```text
assets/body_meshes/
```

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2e_base_mesh_config.example.json --dry-run
```

Actual run:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2e_base_mesh_config.example.json
```

Validate:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2e/labels/labels.csv
```

If no mesh is available at `assets/body_meshes/base_human.obj`, the example config uses `procedural_fallback` mode. Real mesh deformation comes later.

## Phase 2F Mesh Variation and Auto-Framing

Phase 2F adds the first approximate deformation pass for imported base meshes.

The renderer now can:

- Compute measurement-driven region scale factors
- Scale approximate vertical mesh regions for shoulders, chest, waist, hips, legs, and arms
- Preserve target height after deformation
- Center the mesh on the origin after deformation
- Auto-frame front and side cameras around the full body

This is intentionally approximate. It does not require bones, rigging, SMPL/SMPL-X, or garment dressing. The goal is body diversity and better full-body framing, not final scan-grade realism.

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2f_mesh_variation_config.example.json --dry-run
```

Actual run:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2f_mesh_variation_config.example.json
```

Validate:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2f/labels/labels.csv
```

Open render:

```bash
code data/synthetic/phase_2f/images/front/sample_000001_front.png
```
