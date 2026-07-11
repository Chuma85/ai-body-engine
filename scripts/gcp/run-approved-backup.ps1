[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$ApproveSyntheticData,
    [switch]$ApproveModelAssets,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ConfirmProject,
    [switch]$IncludeDatabaseBackup,
    [switch]$ApproveDatabaseBackup,
    [string]$DatabaseBackupPath = $env:AI_BODY_RAILWAY_BACKUP_PATH
)

$ErrorActionPreference = "Stop"
$projectId = "fashionai-501816"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$generator = Join-Path $PSScriptRoot "generate-upload-manifest.py"
$preflight = Join-Path $PSScriptRoot "preflight-live-upload.ps1"
if (-not $Execute) { throw "-Execute is required for run-approved-backup.ps1. Use sync-ai-body-assets.ps1 for a dry run." }
if (-not $ApproveSyntheticData -or -not $ApproveModelAssets) { throw "Live upload requires -ApproveSyntheticData and -ApproveModelAssets." }
if ($ConfirmProject -cne $projectId) { throw "Type the exact project ID: $projectId" }
if ($IncludeDatabaseBackup -xor $ApproveDatabaseBackup) { throw "Database upload requires both -IncludeDatabaseBackup and -ApproveDatabaseBackup." }
if ($IncludeDatabaseBackup -and [string]::IsNullOrWhiteSpace($DatabaseBackupPath)) { throw "Set -DatabaseBackupPath or AI_BODY_RAILWAY_BACKUP_PATH." }

$arguments = @($generator, "--project-id", $projectId)
if ($Execute) { $arguments += "--execute" }
if ($IncludeDatabaseBackup) { $arguments += @("--railway-backup", $DatabaseBackupPath, "--approve-database-backup") }
$result = (& python @arguments | Out-String) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Approved manifest generation failed." }
$manifestPath = $result.manifest
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$realWorld = @($manifest.objects | Where-Object { $_.category -in @("real-world datasets", "participant images", "verified exports") })
$database = @($manifest.objects | Where-Object { $_.category -eq "database backup" })
if ($realWorld.Count -ne 0) { throw "Real-world participant data remains disabled." }
if (-not $IncludeDatabaseBackup -and $database.Count -ne 0) { throw "Database backup appeared without approval flags." }
if ($IncludeDatabaseBackup -and $database.Count -ne 1) { throw "Expected exactly one approved database dump object." }
$preflightArguments = @{ ProjectId = $projectId; ManifestPath = $manifestPath }
if ($IncludeDatabaseBackup) { $preflightArguments.AllowApprovedDatabaseBackup = $true }
& $preflight @preflightArguments
if ($LASTEXITCODE -ne 0) { throw "Preflight failed." }

foreach ($object in $manifest.objects) {
    if ($object.source_relative_path) { $source = Join-Path $repoRoot $object.source_relative_path }
    elseif ($object.category -eq "database backup") { $source = $DatabaseBackupPath }
    else { throw "Unsupported manifest source reference." }
    & gcloud storage cp --no-clobber $source $object.gcs_uri --project=$projectId
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($object.gcs_uri)" }
}
$manifestUri = "gs://fashionai-ai-body-artifacts-501816/migration-manifests/$([IO.Path]::GetFileName($manifestPath))"
& gcloud storage cp --no-clobber $manifestPath $manifestUri --project=$projectId
if ($LASTEXITCODE -ne 0) { throw "Timestamped manifest upload failed." }
Write-Host "Backup complete. Manifest: $manifestPath"
Write-Host "No local files or GCS objects were deleted; no Vertex AI registration or promotion was performed."
