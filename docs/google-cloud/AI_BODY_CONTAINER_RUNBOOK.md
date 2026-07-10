# AI Body Engine Container Runbook

## Workload mapping

The images delegate to existing repository commands:

| Image | Existing command | Purpose |
|---|---|---|
| `body-engine-training` | `python -m training.train_candidate_model` | Candidate-only verified-data training |
| `body-engine-evaluation` | `python -m training.evaluate_candidate_model` | Evaluation, leakage audit, split audit, metrics, and recommendation reports |
| `body-engine-inference` | `uvicorn app.main:app` | Existing FastAPI inference/backend service |
| `dataset-validator` | `python -m synthetic.validate_synthetic_dataset` or `python -m training.datasets.verified_measurement_dataset` | Synthetic or verified dataset validation |

Evaluation never promotes a model. Training never activates a production model.

## Runtime inputs

Images support `GCP_PROJECT_ID`, `GCP_REGION`, `DATASET_URI`, `MODEL_INPUT_URI`, `MODEL_OUTPUT_URI`, `REPORT_OUTPUT_URI`, `TRAINING_RUN_ID`, `MODEL_VERSION`, and `CANDIDATE_ID`. Input/output URIs must resolve to container-local paths supplied by a Cloud Storage FUSE mount or a job download step. A raw `gs://` value fails clearly rather than being treated as a local file.

No datasets, participant photos, checkpoints, credentials, `.env` files, dumps, Git metadata, caches, or virtual environments are copied into images. Workloads run as the non-root `bodyengine` user and emit JSON lifecycle logs containing identifiers only, never file contents.

## Local builds

```powershell
$sha = git rev-parse HEAD
docker build -f containers/base/Dockerfile -t body-engine-base:$sha .
docker build -f containers/training/Dockerfile --build-arg BASE_IMAGE=body-engine-base:$sha -t body-engine-training:local .
docker build -f containers/evaluation/Dockerfile --build-arg BASE_IMAGE=body-engine-base:$sha -t body-engine-evaluation:local .
docker build -f containers/inference/Dockerfile --build-arg BASE_IMAGE=body-engine-base:$sha -t body-engine-inference:local .
docker build -f containers/dataset-validator/Dockerfile --build-arg BASE_IMAGE=body-engine-base:$sha -t dataset-validator:local .
```

Run help/startup and tiny synthetic fixture validation:

```powershell
.\scripts\gcp\smoke-test-containers.ps1 -Tag local
```

## Artifact Registry

Preview repository creation:

```powershell
.\scripts\gcp\create-artifact-registry.ps1
```

Create only if absent:

```powershell
.\scripts\gcp\create-artifact-registry.ps1 -Execute
```

Preview immutable image URIs:

```powershell
.\scripts\gcp\build-and-push-containers.ps1
```

Build and push after explicit approval and authentication:

```powershell
gcloud auth login
gcloud config set project fashionai-501816
.\scripts\gcp\build-and-push-containers.ps1 -Execute -VersionTag v1-candidate
```

Commit-SHA tags are immutable identifiers. Optional human-readable tags are added only when the tag does not already exist; the script refuses silent replacement.

## Cloud Build

```powershell
gcloud builds submit --project fashionai-501816 --config cloudbuild/ai-body-containers.yaml --substitutions=_VERSION_TAG=v1-candidate .
```

The build produces images under `northamerica-northeast2-docker.pkg.dev/fashionai-501816/fashionai-containers/`. Cloud Build does not deploy, activate, or promote any model.

## Example job contracts

Training requires mounted `DATASET_URI` and writable `MODEL_OUTPUT_URI`:

```text
DATASET_URI=/mnt/datasets/verified/v1
MODEL_OUTPUT_URI=/mnt/models/candidates/run-123
TRAINING_RUN_ID=run-123
MODEL_VERSION=candidate-v1
```

Evaluation additionally requires `MODEL_INPUT_URI` pointing to candidate `model.json` and writes to `REPORT_OUTPUT_URI`. Dataset validation requires `DATASET_URI`; set `DATASET_KIND=verified` for verified exports, otherwise synthetic validation is used.

Missing required input, missing mounted paths, unsupported URI schemes, invalid dataset kind, underlying command failure, or validation failure all return non-zero status.
