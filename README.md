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

## Future Phases

- Phase 2 synthetic dataset generator
- Phase 3 dataset schema
- Phase 4 training pipeline
- Phase 5 model evaluation
- Phase 6 inference API
- Phase 7 agents
- Phase 8 FashionApp integration
