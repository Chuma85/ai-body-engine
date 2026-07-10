[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectId,
    [switch]$Execute,
    [switch]$IncludeRealWorld,
    [switch]$ApproveRealWorld,
    [switch]$IncludeRailwayBackup,
    [switch]$ApproveDatabaseBackup,
    [string]$RailwayBackupPath = $env:AI_BODY_RAILWAY_BACKUP_PATH,
    [switch]$DisableChecksums,
    [switch]$DeleteRemote
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$generator = Join-Path $PSScriptRoot "generate-upload-manifest.py"

if ($DeleteRemote) { throw "Remote deletion is intentionally unavailable in GCP-B. Use a separately reviewed destructive procedure." }
if ($IncludeRealWorld -and -not $ApproveRealWorld) { throw "-IncludeRealWorld requires -ApproveRealWorld." }
if ($IncludeRailwayBackup -and -not $ApproveDatabaseBackup) { throw "-IncludeRailwayBackup requires -ApproveDatabaseBackup." }
if ($IncludeRailwayBackup -and [string]::IsNullOrWhiteSpace($RailwayBackupPath)) {
    $RailwayBackupPath = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads/tailormade-railway-backup.dump"
}

$arguments = @($generator, "--project-id", $ProjectId)
if ($DisableChecksums) { $arguments += "--no-checksums" }
if ($IncludeRealWorld) { $arguments += @("--include-real-world", "--approve-real-world") }
if ($IncludeRailwayBackup) { $arguments += @("--railway-backup", $RailwayBackupPath, "--approve-database-backup") }
if ($Execute) { $arguments += "--execute" }

$generatorOutput = & python @arguments | Out-String
if ($LASTEXITCODE -ne 0) { throw "Upload manifest generation failed." }
$result = $generatorOutput | ConvertFrom-Json
$manifestPath = $result.manifest
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json

Write-Host "Manifest: $manifestPath"
Write-Host "Objects: $($manifest.summary.object_count); bytes: $($manifest.summary.total_size_bytes)"
if (-not $Execute) {
    Write-Host "DRY RUN: no authentication check or upload was attempted. Add -Execute only after reviewing the manifest."
    $manifest.objects | Select-Object -First 25 source_relative_path, size_bytes, gcs_uri
    if ($manifest.summary.object_count -gt 25) { Write-Host "Preview limited to 25 objects; the manifest contains the complete upload plan." }
    exit 0
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required for -Execute." }
& gcloud auth print-access-token *> $null
if ($LASTEXITCODE -ne 0) { throw "Valid gcloud authentication is required for -Execute." }

foreach ($object in $manifest.objects) {
    if ($object.source_relative_path) { $source = Join-Path $repoRoot $object.source_relative_path }
    elseif ($object.source_path_from_environment -eq "AI_BODY_RAILWAY_BACKUP_PATH") { $source = $RailwayBackupPath }
    else { throw "Manifest contains an unsupported source reference." }
    if ($PSCmdlet.ShouldProcess($object.gcs_uri, "Upload $source")) {
        & gcloud storage cp --no-clobber $source $object.gcs_uri --project=$ProjectId
        if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($object.gcs_uri)" }
    }
}

$manifestBucket = "fashionai-ai-body-artifacts-501816"
$manifestUri = "gs://$manifestBucket/migration-manifests/$([IO.Path]::GetFileName($manifestPath))"
& gcloud storage cp --no-clobber $manifestPath $manifestUri --project=$ProjectId
if ($LASTEXITCODE -ne 0) { throw "Sanitized manifest upload failed." }
Write-Host "Upload complete. No remote objects were deleted. Verify with verify-cloud-backup.ps1."
