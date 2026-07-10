# Secure Source Manager Migration Preparation

## Current state and safety boundary

GitHub remains the active `origin`, default branch source, and rollback source. At the GCP-E audit point, the repository had one local/remote branch (`main`), no tags, no tracked Git LFS objects, no `.github/workflows` pipeline, pytest as the test command, and no configured lint or static type checker. Secret values are not required for tests; live Google operations use interactive or workload identity credentials.

GCP-E does not create a Secure Source Manager instance/repository, push the Google mirror, create a trigger, publish an image, deploy a service, or switch production CI/CD.

## Create or identify Secure Source Manager resources

Authenticate and select the project:

```powershell
gcloud auth login
gcloud config set project fashionai-501816
```

An existing Secure Source Manager instance is required. Replace `<INSTANCE_ID>` and `<DESCRIPTION>`; do not paste credentials into either field:

```powershell
gcloud source-manager repos create ai-body-engine `
  --project=fashionai-501816 `
  --region=northamerica-northeast2 `
  --instance=<INSTANCE_ID> `
  --description=<DESCRIPTION>
```

Describe the repository and copy its `gitHttps` URI:

```powershell
gcloud source-manager repos describe ai-body-engine `
  --project=fashionai-501816 `
  --region=northamerica-northeast2
```

These commands are documented only; run them after explicit approval and IAM review.

## Configure the second remote

Preview without modifying Git configuration:

```powershell
.\scripts\gcp\setup-google-git-remote.ps1 -RepositoryUrl <GIT_HTTPS_URL>
```

Add `google` while preserving `origin`:

```powershell
.\scripts\gcp\setup-google-git-remote.ps1 -RepositoryUrl <GIT_HTTPS_URL> -Execute
```

If `google` already points elsewhere, the script refuses replacement unless `-ApproveReplaceGoogleRemote` is also supplied. It never changes `origin` and never pushes automatically.

After reviewing refs, push without force:

```powershell
git push google --all
git push google --tags
```

## Verify the mirror

```powershell
.\scripts\gcp\verify-google-git-mirror.ps1
```

Verification compares local and Google branch names/SHAs, tags, the default branch, and expected files at the Google default commit. A mismatch exits non-zero. `origin` is checked before and after verification.

## Cloud Build validation

The validation pipeline runs the complete pytest suite, checked-in asset manifest validation, JSON Schema/YAML checks, Python compilation, four container builds, help/startup checks, and a tiny synthetic validator fixture. Lint and type-check gates report `not configured` until the repository adopts real tools/configuration.

```powershell
gcloud builds submit --project fashionai-501816 --config cloudbuild/validate-ai-body-engine.yaml .
```

The publishing pipeline repeats validation, refuses an existing immutable commit-SHA tag, builds and smoke-tests all four images, and only then pushes to:

```text
northamerica-northeast2-docker.pkg.dev/fashionai-501816/fashionai-containers/<IMAGE>:<COMMIT_SHA>
```

```powershell
gcloud builds submit --project fashionai-501816 --config cloudbuild/build-ai-body-containers.yaml .
```

It creates no mutable production tag, endpoint, deployment, or model promotion.

## Connect Secure Source Manager to Cloud Build

After the mirror is verified, create a Cloud Build repository connection/trigger through Developer Connect or the Google Cloud console. Scope the trigger to `main`, point validation to `cloudbuild/validate-ai-body-engine.yaml`, and require that validation before enabling the image publishing trigger. Keep GitHub triggers active until Google-trigger history and commit SHA inputs are verified.

Rollback is administrative: disable the Google trigger and continue from GitHub `origin`. Do not delete either remote, branches, tags, or Secure Source Manager history during rollback.
