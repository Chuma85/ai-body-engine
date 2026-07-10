# AI Body Engine Google Cloud Migration Plan

## Phase boundary

GCP-A is audit-only. It creates a local metadata inventory and migration plan. It does not create Google Cloud resources or upload, delete, move, rename, train, promote, or activate any dataset or model.

Target project: `fashionai-501816`; primary region: `northamerica-northeast2`.

## Destination architecture

| Asset class | Destination | Controls |
|---|---|---|
| Source code | Google Secure Source Manager; GitHub remains a temporary mirror | Protected branches, review gates, workload identity |
| Synthetic and approved real-world datasets | Separate private Cloud Storage prefixes/buckets | Uniform bucket-level access, versioning, retention, CMEK if required |
| Participant photos | Existing private bucket `fashionai-body-data-uploads` | Never make public; consent, retention, deletion, and least-privilege access |
| Model files and checkpoints | Private Cloud Storage model bucket | Immutable versioned paths and checksums |
| Promoted model metadata | Vertex AI Model Registry | Registration only after explicit evaluation and promotion approval |
| Container images | Regional Artifact Registry | Vulnerability scanning and immutable release tags |
| Secrets | Secret Manager | Recreate values manually; never upload local credential files |
| Database backups | Dedicated private Cloud Storage bucket | Separate IAM, retention/lock policy, restore test |
| Reports and audits | Private Cloud Storage report bucket | Preserve lineage to dataset and model versions |

## Safe migration sequence

1. Review the generated inventory, classifications, sensitivity flags, eligibility decisions, and largest-file list.
2. Resolve every sensitive finding that is not ignored by Git. Rotate any credential suspected of exposure outside this audit.
3. Design buckets and prefixes by data class and environment. Keep participant data isolated from synthetic data and model artifacts.
4. Create service accounts with separate read, writer, training, deployment, and backup roles. Prefer workload identity over key files.
5. Create resources through reviewed infrastructure-as-code in a later authorized phase. Enable audit logs, versioning, lifecycle policies, and public-access prevention.
6. Generate checksums and immutable upload batches. Run the dry-run and obtain approval before any transfer.
7. Upload synthetic data first, verify counts/bytes/checksums, then approved real-world exports. Never upload rejected, temporary, cached, or credential files.
8. Upload model artifacts to versioned GCS paths. Keep training, evaluation, promotion, registry registration, activation, and rollback as separate approved actions.
9. Mirror source into Secure Source Manager and validate CI before changing GitHub's role.
10. Rehearse database backup and restore in isolation before production cutover.

## Dry run

This command scans locally and prints metadata for files that would be eligible. It performs no network calls and uploads nothing:

```powershell
python scripts/gcp/audit-ai-body-assets.py --dry-run
```

To refresh the checked-in metadata inventory:

```powershell
python scripts/gcp/audit-ai-body-assets.py
```

## Explicit exclusions

Never upload `.git/`, `.venv/`, `venv/`, any `__pycache__/`, `.pytest_cache/`, `.pytest-tmp/`, `node_modules/`, temporary or build directories, logs, or local credential files. `.env.example` is documentation, not a secret source; review it before source mirroring.

## Acceptance gates for a later transfer phase

- Resource names, regions, IAM, retention, encryption, and cost controls are approved.
- Every transfer batch has a metadata manifest and checksums.
- Sensitive data has a documented legal basis, consent scope, residency decision, retention, and deletion path.
- Source and destination counts, total bytes, and checksums match.
- Vertex registration references a verified GCS artifact and preserves evaluation and promotion evidence.
- Rollback is tested before production activation.
