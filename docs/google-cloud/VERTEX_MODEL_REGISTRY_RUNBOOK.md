# Vertex AI Model Registry Controlled Promotion Runbook

## Authority and phase boundary

The existing local lifecycle remains authoritative. Vertex mirrors candidate and promoted metadata; it does not bypass `model_lifecycle/model_registry.json`, completed training runs, written promotion decisions, the local production pointer, or rollback history. GCP-D creates no endpoint and deploys no model.

Vertex inherently gives the first uploaded version its registry `default` alias. That alias is not treated as production approval here: the local promoted pointer remains empty until the explicit promotion command succeeds. Later uploads are attached to the existing parent model with `is_default_version=False`, so registration alone does not move Vertex's default alias.

The audited local sequence is:

1. Approve a training queue entry.
2. Generate `training_manifest.json`; it records `auto_train: false` and requires an explicit runner.
3. Complete a registered training run, creating an `evaluation_pending` local candidate.
4. Evaluate the candidate. The existing evaluator writes metrics, leakage audit, split audit, compatibility metadata, and a recommendation without promotion.
5. Register the candidate in Vertex with candidate status.
6. Promote only with a named identity and written approval reference after all gates pass.
7. Roll back only to a previously promoted version; history is retained.

## Discovered artifacts

The GCP-D audit found `.joblib` and `.pt` model files locally. The adapter also supports `.pkl`, `.pth`, `.onnx`, `.ckpt`, `.h5`, `.keras`, `.pb`, and SavedModel prefixes. Suggested names such as `clean-synthetic-3h-i`, `mobile-realistic-3h-j`, `curriculum-3h-l`, and `pretrained-backbone-v1` are naming guidance only; this phase does not claim matching registrable GCS artifacts exist.

## SDK setup and authentication

Live execution requires Google Application Default Credentials and the optional SDK packages:

```powershell
python -m pip install google-cloud-aiplatform google-cloud-storage
gcloud auth application-default login
gcloud config set project fashionai-501816
```

Mocked tests require neither SDK packages nor credentials.

## Candidate registration metadata

Prepare a JSON input containing:

```json
{
  "model_version_id": "candidate-v1",
  "artifact_uri": "gs://fashionai-ai-body-models-501816/candidates/candidate-v1/model.joblib",
  "model_format": "joblib",
  "training_run_id": "run-v1",
  "source_dataset_version": "dataset-v1",
  "git_commit_sha": "0000000000000000000000000000000000000000",
  "metrics": {
    "evaluation_status": "passed",
    "clean_synthetic_mae": 2.4,
    "mobile_realistic_mae": 3.1
  },
  "evaluation_report_uri": "gs://fashionai-ai-body-artifacts-501816/evaluations/candidate-v1/report.json",
  "leakage_audit_status": "passed",
  "compatibility_metadata": {"status": "passed", "architecture": "measurement_regressor"},
  "candidate_status": "candidate",
  "architecture_backbone": "measurement-regressor-v1"
}
```

The candidate and completed training run must already exist locally with matching lineage. The artifact must exist under `gs://fashionai-ai-body-models-501816/`.

Dry-run and execution:

```powershell
python scripts/gcp/register-model-candidate.py --metadata .\candidate-metadata.json
python scripts/gcp/register-model-candidate.py --metadata .\candidate-metadata.json --execute
```

## List versions

```powershell
python scripts/gcp/list-model-versions.py
python scripts/gcp/list-model-versions.py --include-vertex
```

## Explicit promotion

Dry-run:

```powershell
python scripts/gcp/promote-model-version.py --model-version-id candidate-v1 --approval-identity reviewer@example.com --approval-reference CHANGE-1234
```

Execute only after reviewing the dry-run:

```powershell
python scripts/gcp/promote-model-version.py --model-version-id candidate-v1 --approval-identity reviewer@example.com --approval-reference CHANGE-1234 --execute
```

Promotion is blocked unless evaluation passed, leakage audit passed, compatibility passed, the model version is explicit, and both approval fields are present. Execution writes the existing local promotion-decision record before changing promoted metadata.

## Rollback

Dry-run and explicit execution:

```powershell
python scripts/gcp/rollback-model-version.py --model-version-id candidate-v0 --rolled-back-by release-manager@example.com --reason "Post-promotion regression"
python scripts/gcp/rollback-model-version.py --model-version-id candidate-v0 --rolled-back-by release-manager@example.com --reason "Post-promotion regression" --execute
```

Rollback changes local and Vertex lifecycle metadata pointers. It does not delete the current or historical model artifacts and does not deploy to an endpoint.
