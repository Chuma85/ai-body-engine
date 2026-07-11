[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()][string]$ProjectId = "fashionai-501816",
    [string]$ManifestPath,
    [switch]$AllowApprovedDatabaseBackup
)

$ErrorActionPreference = "Stop"
$expectedProject = "fashionai-501816"
$buckets = @(
    "fashionai-ai-body-datasets-501816",
    "fashionai-ai-body-models-501816",
    "fashionai-ai-body-artifacts-501816",
    "fashionai-database-backups-501816"
)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$generator = Join-Path $PSScriptRoot "generate-upload-manifest.py"

if ($ProjectId -ne $expectedProject) { throw "Project must be $expectedProject." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
$account = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($account)) { throw "An authenticated gcloud account is required." }
$configuredProject = (& gcloud config get-value project 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $configuredProject -ne $expectedProject) { throw "Active gcloud project must be $expectedProject; found '$configuredProject'." }

foreach ($bucket in $buckets) {
    $json = & gcloud storage buckets describe "gs://$bucket" --project=$ProjectId --format=json
    if ($LASTEXITCODE -ne 0) { throw "Required bucket is missing or inaccessible: $bucket" }
    $description = $json | ConvertFrom-Json
    if ($description.iamConfiguration.publicAccessPrevention -ne "enforced") { throw "Public access prevention is not enforced: $bucket" }
    if (-not $description.iamConfiguration.uniformBucketLevelAccess.enabled) { throw "Uniform bucket-level access is not enabled: $bucket" }
    Write-Host "PASS bucket: $bucket (private, uniform access)"
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $output = & python $generator --project-id $ProjectId | Out-String
    if ($LASTEXITCODE -ne 0) { throw "Manifest generation failed." }
    $ManifestPath = ($output | ConvertFrom-Json).manifest
}
$manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$forbidden = @($manifest.objects | Where-Object { $_.source_relative_path -match '(^|/)(\.git|\.venv|venv|__pycache__|\.pytest_cache|\.pytest-tmp|\.tmp|node_modules|build|dist|\.mypy_cache|\.ruff_cache)(/|$)' -or $_.source_relative_path -match '(^|/)\.env(?:\..+)?$|credential|service[-_]?account|\.pem$|\.key$|\.p12$|\.pfx$|database[_-]?url' })
$realWorld = @($manifest.objects | Where-Object { $_.category -in @("real-world datasets", "participant images", "verified exports") })
$database = @($manifest.objects | Where-Object { $_.category -eq "database backup" })
if ($forbidden.Count -ne 0) { throw "Manifest contains $($forbidden.Count) forbidden paths." }
if ($realWorld.Count -ne 0) { throw "Manifest contains $($realWorld.Count) real-world or participant objects." }
if ($database.Count -ne 0 -and (-not $AllowApprovedDatabaseBackup -or -not $manifest.database_backup_approved)) { throw "Database backup is present without separate approval." }

Write-Host "PASS authenticated account: $account"
Write-Host "PASS project: $configuredProject"
Write-Host "PASS zero real-world/participant objects and zero forbidden paths"
Write-Host "Manifest: $ManifestPath"
Write-Host "Objects: $($manifest.summary.object_count); bytes: $($manifest.summary.total_size_bytes)"
$manifest.objects | Group-Object { ([uri]$_.gcs_uri).Host } | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) objects, $(($_.Group | Measure-Object size_bytes -Sum).Sum) bytes"
}
Write-Host "PREFLIGHT ONLY: no upload was performed."
