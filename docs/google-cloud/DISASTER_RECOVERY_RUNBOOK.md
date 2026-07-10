# AI Body Engine Disaster Recovery Runbook

## Safety policy

All GCP-F procedures are non-destructive by default. No script automatically deletes real-world participant data, promoted models, database backups, evaluation/leakage/audit reports, or any remote object. Any future cleanup implementation must begin with a dry-run and require explicit approval; no cleanup execution is included in this phase.

Recovery must target an isolated directory, fresh database, or non-production project. Never restore over production and never print participant data, credentials, connection strings, or database rows into logs.

## Retention recommendations

The machine-readable policy is `config/google-cloud/retention-policy.yaml`.

| Class | Recommendation |
|---|---|
| Synthetic datasets | Retain active/reproducible versions at least 365 days; manually review superseded copies only after checksum verification |
| Real-world datasets and participant photos | No automatic deletion; follow consent, privacy deletion, legal basis, and governance policy |
| Verified exports | Retain immutable lineage and audit evidence; no automatic deletion |
| Candidate models/checkpoints | Retain through evaluation and at least a 365-day reproducibility window; archive before review |
| Promoted models | Retain all promoted and rollback versions indefinitely; never automatically delete |
| Archived/pretrained models | Retain at least seven years unless governance requires longer |
| Evaluation, comparison, leakage, manifest, and audit reports | Retain indefinitely; never automatically delete |
| Railway/Cloud SQL database backups | No automatic deletion; apply an approved encrypted schedule only after tested restore points exist |

## Create and validate a backup index

```powershell
python scripts/gcp/create-backup-index.py `
  --upload-manifest .\.tmp\gcp-upload-manifests\upload-manifest-<timestamp>.json `
  --output .\.tmp\backup-indexes\backup-index-<timestamp>.json
```

The index contains only object metadata: URI, category, size, checksum, creation timestamp, dataset/model version, originating Git SHA when available, and retention classification.

Export or mock GCS object metadata into an `objects` array and run:

```powershell
python scripts/gcp/check-backup-integrity.py `
  --backup-index .\.tmp\backup-indexes\backup-index-<timestamp>.json `
  --observed-objects .\.tmp\observed-gcs-objects.json

python scripts/gcp/check-orphaned-models.py `
  --backup-index .\.tmp\backup-indexes\backup-index-<timestamp>.json
```

Integrity checks report missing objects, size/checksum mismatches, model records without artifacts, model artifacts without records, training manifests referencing absent dataset versions, and evaluation reports referencing absent models. They never download or print object contents.

## Isolated object restore planning

```powershell
.\scripts\gcp\restore-backup-dry-run.ps1 `
  -BackupIndex .\.tmp\backup-indexes\backup-index-<timestamp>.json `
  -DestinationRoot C:\recovery\ai-body-engine
```

The default prints a plan only. Explicit non-production download requires both flags and an empty/isolated destination:

```powershell
.\scripts\gcp\restore-backup-dry-run.ps1 `
  -BackupIndex .\.tmp\backup-indexes\backup-index-<timestamp>.json `
  -DestinationRoot C:\recovery\ai-body-engine `
  -ApproveNonProductionDownload -ExecuteDownload
```

This downloads files only. It never restores a database or overwrites production.

## Railway custom-format PostgreSQL dump recovery

Set credentials through the environment or a secret-injection mechanism; never put passwords in commands or files:

```powershell
$env:RESTORE_DATABASE_URL = "postgresql://<RESTORE_USER>:<PASSWORD_FROM_SECRET_MANAGER>@<FRESH_HOST>:5432/<FRESH_DATABASE>"
$dump = "C:\recovery\tailormade-railway-backup.dump"
```

Confirm that the dump is a readable PostgreSQL archive without restoring rows:

```powershell
pg_restore --list $dump
```

Create a fresh empty database using approved administration credentials, then restore only into that new target:

```powershell
createdb --host=<FRESH_HOST> --username=<RESTORE_ADMIN> <FRESH_DATABASE>
pg_restore --exit-on-error --no-owner --no-privileges --dbname=$env:RESTORE_DATABASE_URL $dump
```

Do not add `--clean` or target an existing production database.

## Cloud SQL export recovery

Cloud SQL SQL exports are restored differently from Railway custom-format archives. Export to the private backup bucket:

```powershell
gcloud sql export sql <SOURCE_INSTANCE> `
  gs://fashionai-database-backups-501816/cloud-sql/<timestamp>/database.sql.gz `
  --database=<SOURCE_DATABASE> --project=fashionai-501816
```

Create a fresh recovery instance/database through a separately approved infrastructure procedure, then import:

```powershell
gcloud sql import sql <FRESH_RECOVERY_INSTANCE> `
  gs://fashionai-database-backups-501816/cloud-sql/<timestamp>/database.sql.gz `
  --database=<FRESH_DATABASE> --project=fashionai-501816
```

Never import into the active production database.

## Verification queries

Run counts against both the approved source snapshot and fresh restored database, compare results, and record only counts—not row contents:

```sql
SELECT COUNT(*) AS user_count FROM "User";
SELECT COUNT(*) AS collection_session_count FROM "FieldDataCollectionSession";
SELECT COUNT(*) AS measurement_record_count FROM "FieldDataMeasurementRecord";
SELECT COUNT(*) AS photo_version_count FROM "FieldDataPhotoVersion";
SELECT COUNT(*) AS submission_review_audit_count FROM "FieldDataSubmissionReviewAudit";
SELECT COUNT(*) AS dataset_lineage_count FROM "FieldDatasetLineage";
```

Also verify schema migration state, foreign-key integrity, representative non-sensitive aggregate ranges, and application health against the isolated recovery database. Record the backup index checksum, restore start/end time, operator approval reference, and count comparison in the recovery audit.

## Recovery acceptance and rollback

- Every expected object is present with matching size/checksum.
- No orphaned model records or artifacts remain unexplained.
- Training datasets and evaluation/model relationships resolve.
- `pg_restore --list` succeeds for Railway archives.
- Restore targets are fresh and non-production.
- Important table counts match the approved source snapshot.
- Production configuration remains unchanged throughout the exercise.

If validation fails, stop, preserve logs and metadata, mark the recovery attempt failed, and investigate. Do not compensate by deleting history or overwriting the source backup.
