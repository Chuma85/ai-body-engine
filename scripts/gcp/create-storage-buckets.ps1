[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectId,
    [ValidateNotNullOrEmpty()][string]$Region = "northamerica-northeast2",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$buckets = @(
    "fashionai-ai-body-datasets-501816",
    "fashionai-ai-body-models-501816",
    "fashionai-ai-body-artifacts-501816",
    "fashionai-database-backups-501816"
)

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
if (-not $Execute) {
    Write-Host "DRY RUN: no buckets will be created. Re-run with -Execute after review."
}

foreach ($bucket in $buckets) {
    $uri = "gs://$bucket"
    if (-not $Execute) {
        Write-Host "WOULD CHECK AND CREATE IF MISSING: $uri (STANDARD, $Region, uniform access, public access prevention, Google-managed encryption)"
        continue
    }
    & gcloud storage buckets describe $uri --project=$ProjectId *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SKIP existing bucket: $uri"
        continue
    }
    if ($PSCmdlet.ShouldProcess($uri, "Create private regional Standard bucket")) {
        & gcloud storage buckets create $uri --project=$ProjectId --location=$Region --default-storage-class=STANDARD --uniform-bucket-level-access --public-access-prevention
        if ($LASTEXITCODE -ne 0) { throw "Bucket creation failed: $uri" }
    }
}

Write-Host "No existing bucket was deleted or made public. Google-managed encryption is the default."
