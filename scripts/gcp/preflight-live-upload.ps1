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
. (Join-Path $PSScriptRoot "gcloud-command.ps1")

function Get-BucketSetting {
    param(
        [Parameter(Mandatory = $true)][string]$Bucket,
        [Parameter(Mandatory = $true)][string]$Setting
    )

    $result = Invoke-GcloudCommand -Arguments @(
        "storage", "buckets", "describe", "gs://$Bucket",
        "--project=$ProjectId", "--format=value($Setting)"
    )
    if ($result.ExitCode -ne 0) {
        throw "Bucket '$Bucket' setting '$Setting' could not be read (gcloud exit $($result.ExitCode))."
    }
    $values = @($result.StdOut | ForEach-Object { $_.ToString().Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($values.Count -ne 1) {
        throw "Bucket '$Bucket' setting '$Setting' is missing or malformed; expected exactly one value."
    }
    return $values[0]
}

function Get-BucketStorageClass {
    param([Parameter(Mandatory = $true)][string]$Bucket)

    $result = Invoke-GcloudCommand -Arguments @(
        "storage", "buckets", "describe", "gs://$Bucket",
        "--project=$ProjectId", "--raw", "--format=json"
    )
    if ($result.ExitCode -ne 0) {
        throw "Bucket '$Bucket' storage class could not be read (gcloud exit $($result.ExitCode))."
    }
    $json = $result.StdOut -join [Environment]::NewLine
    try { $description = $json | ConvertFrom-Json }
    catch { throw "Bucket '$Bucket' storage class metadata is malformed JSON." }
    if ($null -eq $description -or $description -is [array]) {
        throw "Bucket '$Bucket' storage class metadata is missing or non-singular."
    }
    $properties = @($description.PSObject.Properties | Where-Object { $_.Name -ceq "storageClass" })
    if ($properties.Count -ne 1) {
        throw "Bucket '$Bucket' storageClass is missing or non-singular."
    }
    $value = if ($null -eq $properties[0].Value) { "" } else { $properties[0].Value.ToString().Trim() }
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Bucket '$Bucket' storageClass is empty or malformed."
    }
    return $value
}

if ($ProjectId -ne $expectedProject) { throw "Project must be $expectedProject." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
$accountResult = Invoke-GcloudCommand -Arguments @("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
if ($accountResult.ExitCode -ne 0) {
    throw "Active gcloud account lookup failed (exit $($accountResult.ExitCode)). Check gcloud authentication and connectivity."
}
$accounts = @($accountResult.StdOut | ForEach-Object { $_.ToString().Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($accounts.Count -eq 0) { throw "An authenticated gcloud account is required; no active authenticated account was returned." }
if ($accounts.Count -ne 1) { throw "Exactly one active authenticated gcloud account is required; found $($accounts.Count)." }
$account = $accounts[0]
if ($account -notmatch '^[^\s@]+@[^\s@]+$') { throw "Active gcloud account output is malformed." }

$configuredAccountResult = Invoke-GcloudCommand -Arguments @("config", "get-value", "account")
$configuredAccounts = @($configuredAccountResult.StdOut | ForEach-Object { $_.ToString().Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($configuredAccountResult.ExitCode -ne 0 -or $configuredAccounts.Count -ne 1) { throw "Configured gcloud account lookup failed or returned an ambiguous value." }
if ($configuredAccounts[0] -ne $account) { throw "The active authenticated gcloud account does not match the configured gcloud account." }

$projectResult = Invoke-GcloudCommand -Arguments @("config", "get-value", "project")
$configuredProjects = @($projectResult.StdOut | ForEach-Object { $_.ToString().Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$configuredProject = if ($configuredProjects.Count -eq 1) { $configuredProjects[0] } else { "" }
if ($projectResult.ExitCode -ne 0 -or $configuredProjects.Count -ne 1 -or $configuredProject -ne $expectedProject) { throw "Active gcloud project must be $expectedProject; found '$configuredProject'." }

foreach ($bucket in $buckets) {
    $publicAccessPrevention = Get-BucketSetting -Bucket $bucket -Setting "public_access_prevention"
    if ($publicAccessPrevention -ine "enforced") { throw "Bucket '$bucket' failed public access prevention: expected 'enforced', found '$publicAccessPrevention'." }

    $uniformAccess = Get-BucketSetting -Bucket $bucket -Setting "uniform_bucket_level_access"
    if ($uniformAccess -inotmatch '^(true)$') { throw "Bucket '$bucket' failed uniform bucket-level access: expected 'true', found '$uniformAccess'." }

    $location = Get-BucketSetting -Bucket $bucket -Setting "location"
    if ($location -ine "NORTHAMERICA-NORTHEAST2") { throw "Bucket '$bucket' failed location validation: expected 'NORTHAMERICA-NORTHEAST2', found '$location'." }

    $storageClass = Get-BucketStorageClass -Bucket $bucket
    if ($storageClass -ine "STANDARD") { throw "Bucket '$bucket' failed storage class validation: expected 'STANDARD', found '$storageClass'." }
    Write-Host "PASS bucket: $bucket (public access prevention enforced, uniform access, NORTHAMERICA-NORTHEAST2, STANDARD)"
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
