# Cloud Storage Backup Runbook

## Safety boundary

GCP-B provides repeatable tooling; dry-run is the default. Nothing uploads unless `-Execute` is supplied. Real-world participant data additionally requires configuration enablement plus `-IncludeRealWorld -ApproveRealWorld`. Railway backups require `-IncludeRailwayBackup -ApproveDatabaseBackup`. Remote deletion is intentionally unavailable in this phase.

The scripts never print file contents and reject Git internals, environments, caches, temporary/build output, Node modules, `.env` files, and credential-like filenames. Service-account JSON keys must not be stored in this repository; use interactive authentication or workload identity.

## Prerequisites and authentication

```powershell
gcloud auth login
gcloud config set project fashionai-501816
```

Confirm the active identity and project before any execution:

```powershell
gcloud auth list
gcloud config get-value project
```

The live-upload preflight requires exactly one active authenticated account from `gcloud auth list --filter=status:ACTIVE --format=value(account)`. It also requires that identity to match the configured account and requires the configured project to be exactly `fashionai-501816`. A configured account alone is not accepted when it is not actively authenticated.

For each configured target bucket, preflight reads the stable `gcloud storage` value fields `public_access_prevention`, `uniform_bucket_level_access`, and `location`, then reads raw JSON and validates the documented API property `storageClass`. It fails closed unless they normalize to `enforced`, `true`, `NORTHAMERICA-NORTHEAST2`, and `STANDARD`; it never changes a bucket setting.

## Validate and dry-run

Generate a checksum-bearing sanitized manifest and print the proposed objects without contacting Google Cloud:

```powershell
.\scripts\gcp\sync-ai-body-assets.ps1 -ProjectId fashionai-501816
```

Manifests are written under `.tmp/gcp-upload-manifests/`. Review object counts, sizes, routes, exclusions, and real-world flags before continuing.

Validate a generated manifest locally:

```powershell
python scripts/gcp/generate-upload-manifest.py --validate-manifest .\.tmp\gcp-upload-manifests\upload-manifest-<timestamp>.json
```

## Create private buckets

Preview:

```powershell
.\scripts\gcp\create-storage-buckets.ps1 -ProjectId fashionai-501816
```

Create missing buckets only:

```powershell
.\scripts\gcp\create-storage-buckets.ps1 -ProjectId fashionai-501816 -Region northamerica-northeast2 -Execute
```

During existence detection, `gcloud storage buckets describe` returning `404 not found` is the expected signal that a target bucket is missing. The script creates it only when `-Execute` is present. Authentication, permission, network, and ownership/global-name-conflict failures are fatal; resolve them rather than treating the bucket as absent. Existing buckets are never recreated or relaxed.

Buckets use regional Standard storage, uniform bucket-level access, public access prevention, and default Google-managed encryption. Existing buckets are skipped and never deleted.

Verify the final configuration for every configured bucket:

```powershell
gcloud storage buckets list --project=fashionai-501816 --format="table(name,location,storageClass,uniformBucketLevelAccess,publicAccessPrevention)"
gcloud storage buckets describe gs://BUCKET_NAME --project=fashionai-501816 --format="yaml(name,location,storageClass,uniformBucketLevelAccess,publicAccessPrevention,encryption)"
```

An absent customer-managed encryption key in the describe output confirms the retained Google-managed encryption default.

## Approved asset execution

```powershell
.\scripts\gcp\sync-ai-body-assets.ps1 -ProjectId fashionai-501816 -Execute
```

Uploads use `--no-clobber`, preserve the repository-relative path below each logical prefix, and upload the sanitized upload manifest to `migration-manifests/`. The operation does not delete remote objects.

## Real-world participant data

Real-world uploads are disabled in `storage-layout.yaml` by default. Enabling the policy is a reviewed configuration change. Even after enablement, both explicit flags are required:

```powershell
.\scripts\gcp\sync-ai-body-assets.ps1 -ProjectId fashionai-501816 -IncludeRealWorld -ApproveRealWorld -Execute
```

Participant consent, retention, residency, and deletion requirements must be approved before enabling this path. Synthetic and real-world assets always use separate prefixes.

## Railway database backup

Prefer an environment variable so local usernames are not embedded in commands or configuration:

```powershell
$env:AI_BODY_RAILWAY_BACKUP_PATH = "$HOME\Downloads\tailormade-railway-backup.dump"
.\scripts\gcp\sync-ai-body-assets.ps1 -ProjectId fashionai-501816 -IncludeRailwayBackup -ApproveDatabaseBackup -Execute
```

The destination is `gs://fashionai-database-backups-501816/railway/<timestamp>/tailormade-railway-backup.dump`. The dump is never committed and cannot be selected without the separate database-backup approval flag.

## Verification

```powershell
.\scripts\gcp\verify-cloud-backup.ps1 -ProjectId fashionai-501816 -ManifestPath .\.tmp\gcp-upload-manifests\upload-manifest-<timestamp>.json
```

Verification compares expected object counts and sizes and reports missing or mismatched URIs. GCS normally exposes CRC32C/MD5 rather than SHA-256; the local SHA-256 is retained in the sanitized manifest for integrity checks and can be compared later if stored as object metadata. Sensitive contents are never downloaded or printed.

## Recovery and repetition

Re-run the same manifest-driven upload safely: `--no-clobber` skips existing object names. If replacement or deletion is required, stop and use a separately reviewed destructive migration procedure. GCP-B deliberately supplies no remote-delete implementation.
