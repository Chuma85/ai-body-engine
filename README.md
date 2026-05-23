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

## Future Phases

- Phase 2 synthetic dataset generator
- Phase 3 dataset schema
- Phase 4 training pipeline
- Phase 5 model evaluation
- Phase 6 inference API
- Phase 7 agents
- Phase 8 FashionApp integration
