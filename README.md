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

## Future Phases

- Phase 2 synthetic dataset generator
- Phase 3 dataset schema
- Phase 4 training pipeline
- Phase 5 model evaluation
- Phase 6 inference API
- Phase 7 agents
- Phase 8 FashionApp integration
