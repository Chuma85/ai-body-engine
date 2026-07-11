# Live AI Body Backup Checklist

This phase prepares an explicitly approved, private GCS backup. It never selects real-world participant datasets, verified participant exports, or participant photos. It does not delete local or cloud objects and does not register or promote Vertex AI models.

## Approved source map

| Asset | Current source roots | Destination |
|---|---|---|
| Synthetic datasets | `data/synthetic/`, `synthetic/labels/` | datasets `/synthetic/` |
| Model checkpoints | checkpoint/model files under `artifacts/baselines/`, `artifacts/deep/`, `artifacts/experiments/`, and blend baseline artifact folders | models `/checkpoints/` |
| Candidate models | none in the audited manifest | models `/candidates/` |
| Promoted models | none in the audited manifest | models `/promoted/` |
| Pretrained assets | none in the audited manifest | models `/pretrained/` |
| Evaluation reports | `reports/` and evaluation-classified files under `artifacts/` | artifacts `/evaluations/` |
| Leakage audits | leakage-classified files under `artifacts/` | artifacts `/leakage-audits/` |
| Rendered assets | `assets/body_meshes/`, `synthetic/renders/`, generated renders under `data/synthetic/` and `artifacts/` | artifacts `/rendered-assets/` |
| Training manifests | `dataset_registry/`, `model_lifecycle/`, manifest-classified files under `artifacts/` | datasets `/training-manifests/` |

The current dry-run plan contains 30,316 objects and 10,969,021,482 bytes: datasets 79 objects / 12,621,365 bytes; models 54 / 12,046,222 bytes; artifacts 30,183 / 10,944,353,895 bytes. Regenerate immediately before execution because local assets can change.

## Required safety checks

- [ ] Review `config/google-cloud/storage-layout.yaml` and the generated manifest.
- [ ] Confirm zero `real-world datasets`, `participant images`, and `verified exports` objects.
- [ ] Confirm zero `.git`, environment files, credentials, virtual environments, caches, build output, or database URLs/credential files.
- [ ] Confirm all four buckets enforce public access prevention and uniform bucket-level access.
- [ ] Confirm the active account and project are correct.
- [ ] Run the default upload without database flags first.
- [ ] Upload the PostgreSQL dump only as a separate explicitly approved run with both database switches.
- [ ] Retain local files and remote objects after verification.

## Commands

```powershell
.\scripts\gcp\create-storage-buckets.ps1 -ProjectId fashionai-501816 -Region northamerica-northeast2 -Execute
.\scripts\gcp\preflight-live-upload.ps1
.\scripts\gcp\run-approved-backup.ps1 -Execute -ApproveSyntheticData -ApproveModelAssets -ConfirmProject fashionai-501816
.\scripts\gcp\run-approved-backup.ps1 -Execute -ApproveSyntheticData -ApproveModelAssets -ConfirmProject fashionai-501816 -IncludeDatabaseBackup -ApproveDatabaseBackup -DatabaseBackupPath "C:\path\to\verified.dump"
.\scripts\gcp\post-upload-verification.ps1 -ManifestPath ".tmp\gcp-upload-manifests\upload-manifest-YYYYMMDDTHHMMSSZ.json"
```

`gcloud storage cp --no-clobber` uses the CLI's resumable upload behavior for large files and never overwrites an existing object. The timestamped manifest is also copied to the private artifacts bucket.
