# AI Body Engine

AI Body Engine is a standalone Python service for future body-capture intelligence in FashionApp. It is intentionally separate from the TypeScript/Next.js/React Native marketplace repo so model training, synthetic data generation, inference, and future agent workflows can evolve independently.

## Current Phase

Phase 1 skeleton:

- FastAPI service shell
- Health endpoint
- PyTorch-ready training folders
- OpenCV/Pillow-ready image-processing dependencies
- Synthetic data folders for future Blender renders and labels
- Placeholder agent classes
- pytest coverage for the initial API
- Docker-ready layout

This phase does not train models, generate synthetic datasets, run Blender automation, predict measurements, or integrate with FashionApp.

## Tech Stack

- Python
- FastAPI
- PyTorch-ready project structure
- Pillow and OpenCV-ready image-processing dependencies
- pytest
- Docker

## Folder Structure

```text
app/          FastAPI application, API routes, config, schemas, services
training/     Future datasets, model definitions, and training scripts
synthetic/    Future Blender scripts, synthetic labels, and renders
inference/    Future inference runtime code
agents/       Future agent workflow placeholders
tests/        pytest test suite
```

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy environment defaults:

```bash
copy .env.example .env
```

## Run The API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

- `GET /`
- `GET /health`

## Run Tests

```bash
python -m pytest
```

## Phase 2A — Synthetic Dataset Prototype

Phase 2A adds a placeholder Python-only silhouette generator. It proves the dataset pipeline before Blender, SMPL, or SMPL-X are introduced.

The generator creates:

- Front silhouette PNG images
- Side silhouette PNG images
- `labels.csv` with synthetic measurement labels
- A validation report that checks required columns, image paths, measurements, and row count

These are simple 2D placeholder silhouettes for pipeline validation. They are not production-quality body modeling, not AI measurements, and not anatomically precise scan outputs.

Generate a sample dataset:

```bash
python -m synthetic.generator.generate_dataset --count 100 --output-dir data/synthetic/phase_2a
```

Validate the generated dataset:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2a/labels/labels.csv
```

Run tests:

```bash
python -m pytest
```

Generated PNG and CSV files are ignored by git so large datasets are not committed by accident. The folder structure is tracked with `.gitkeep` files.

## Phase 2B — Blender Synthetic Pipeline Scaffold

Phase 2A proved the Python-only silhouette generation workflow. Phase 2B prepares the project for a future Blender/3D synthetic rendering pipeline.

This phase adds:

- A Blender render configuration example
- Config validation utilities
- Measurement label schema utilities
- Blender command generation
- A dry-run CLI
- A Blender-compatible script scaffold that can still be imported in normal Python

Normal tests do not require Blender. This phase does not create realistic humans yet; it prepares the path for Phase 2C.

Dry-run the Blender pipeline command:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2b_render_config.example.json --dry-run
```

Run tests:

```bash
python -m pytest
```

## Phase 2C — Procedural Blender Body Rendering MVP

Phase 2C adds the first actual Blender rendering MVP. It creates simple 3D mannequin-like procedural bodies using Blender primitives, then renders front and side PNGs with matching `labels.csv`.

This phase:

- Uses procedural geometry, not SMPL/SMPL-X
- Generates simple mannequin bodies, not production-quality anatomical humans
- Does not train models
- Does not dress garments
- Does not integrate with FashionApp
- Prepares for better anatomy, garment placeholders, and future high-quality try-on rendering

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2c_render_config.example.json --dry-run
```

Actual Blender run:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2c_render_config.example.json
```

Validate generated labels:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2c/labels/labels.csv
```

Run tests:

```bash
python -m pytest
```

## Phase 2D — Anatomical Procedural Body Refinement

Phase 2D improves procedural body realism while still avoiding SMPL/SMPL-X, model training, FashionApp integration, garment dressing, and paid try-on rendering.

This phase improves:

- Torso tapering across chest, waist, abdomen, and hips
- Body-shape profiles for slim, average, athletic, curvy, broad, and plus bodies
- Limb tapering based on sleeve, inseam, thigh, and calf measurements
- Small measurement-friendly pose variation
- Render-quality controls
- Additional metadata columns for generated labels

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2d_render_config.example.json --dry-run
```

Actual Blender render:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2d_render_config.example.json
```

Validate generated labels:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2d/labels/labels.csv
```

Run tests:

```bash
python -m pytest
```

Phase 2D prepares the project for Phase 2E clothing placeholders and the Phase 3 training pipeline.

## Phase 2E — Base Human Mesh Renderer Scaffold

Phase 2E prepares the renderer to move beyond primitive procedural mannequins by supporting external base human mesh assets.

This phase:

- Adds `assets/body_meshes/` for local MakeHuman, MB-Lab, or custom Blender human meshes
- Keeps large mesh assets ignored by git
- Supports future `.blend`, `.fbx`, `.obj`, `.glb`, and `.gltf` assets
- Adds a base-mesh render config with procedural fallback
- Does not use SMPL/SMPL-X yet
- Does not deform meshes yet
- Does not train models or integrate with FashionApp

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2e_base_mesh_config.example.json --dry-run
```

Actual Blender render:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2e_base_mesh_config.example.json
```

Validate generated labels:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2e/labels/labels.csv
```

If `assets/body_meshes/base_human.obj` is not present, the Phase 2E example config falls back to the procedural body renderer.

## Phase 2F — Mesh Variation and Auto-Framing

Phase 2F adds first-pass mesh deformation and full-body camera framing for imported base human meshes.

This phase:

- Applies approximate vertex-region scaling to imported meshes
- Uses generated measurements to vary shoulders, chest, waist, hips, legs, and arms
- Keeps deformation subtle and measurement-friendly
- Preserves mesh height when configured
- Auto-frames front and side cameras around the body bounds
- Does not require rigging, bones, SMPL/SMPL-X, garment dressing, model training, or FashionApp integration

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2f_mesh_variation_config.example.json --dry-run
```

Actual Blender render:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2f_mesh_variation_config.example.json
```

Validate generated labels:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2f/labels/labels.csv
```

Open a render:

```bash
code data/synthetic/phase_2f/images/front/sample_000001_front.png
```

Future phases can improve deformation with rigged meshes, blend shapes, or parametric human models.

## Phase 2G — Rigged / Shape-Key Mesh Pipeline

Phase 2F proved that imported static meshes can be varied, but rough vertex-region scaling can distort non-rigged OBJ bodies. Phase 2G prepares a cleaner path using rigged meshes, armatures, and shape keys.

This phase:

- Adds a rigged mesh config that targets `assets/body_meshes/base_human_rigged.fbx`
- Detects armatures and shape keys when Blender imports a suitable asset
- Prefers shape-key deformation, then conservative bone scaling, then safe object-scale fallback
- Avoids the rough Phase 2F region-scaling path for Phase 2G configs
- Still does not use SMPL/SMPL-X, train models, dress garments, or integrate with FashionApp

Best inputs are MakeHuman, MB-Lab, or custom Blender characters exported with an armature or body morph targets.

Dry run:

```bash
python -m synthetic.blender.run_blender_pipeline --config synthetic/blender/configs/phase_2g_rigged_mesh_config.example.json --dry-run
```

Actual Blender render:

```bash
blender --background --python synthetic/blender/scripts/render_parametric_body.py -- --config synthetic/blender/configs/phase_2g_rigged_mesh_config.example.json
```

Validate generated labels:

```bash
python -m synthetic.generator.validate_dataset data/synthetic/phase_2g/labels/labels.csv
```

Open a render:

```bash
code data/synthetic/phase_2g/images/front/sample_000001_front.png
```

## Future Phases

- Phase 2 synthetic dataset generator
- Phase 3 dataset schema
- Phase 4 training pipeline
- Phase 5 model evaluation
- Phase 6 inference API
- Phase 7 agents
- Phase 8 FashionApp integration
