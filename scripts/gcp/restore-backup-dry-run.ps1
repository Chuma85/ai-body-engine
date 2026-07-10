[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })][string]$BackupIndex,
    [Parameter(Mandatory = $true)][string]$DestinationRoot,
    [switch]$ApproveNonProductionDownload,
    [switch]$ExecuteDownload
)

$ErrorActionPreference = "Stop"
$index = Get-Content -Raw -LiteralPath $BackupIndex | ConvertFrom-Json
$destination = [IO.Path]::GetFullPath($DestinationRoot)
$objects = @($index.objects)
Write-Host "DRY RUN recovery plan: $($objects.Count) objects to isolated destination $destination"
$objects | Select-Object object_uri, category, size_bytes, retention_classification

if (-not $ExecuteDownload) {
    Write-Host "No objects downloaded and no database commands executed. Add both -ApproveNonProductionDownload and -ExecuteDownload for an isolated local download only."
    exit 0
}
if (-not $ApproveNonProductionDownload) { throw "-ExecuteDownload requires -ApproveNonProductionDownload." }
if (Test-Path -LiteralPath $destination) {
    if ((Get-ChildItem -LiteralPath $destination -Force | Select-Object -First 1)) { throw "Restore destination must be empty." }
} else {
    New-Item -ItemType Directory -Path $destination | Out-Null
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required for download execution." }
foreach ($object in $objects) {
    $uri = [Uri]$object.object_uri
    $relative = Join-Path $uri.Host $uri.AbsolutePath.TrimStart('/').Replace('/', [IO.Path]::DirectorySeparatorChar)
    $target = [IO.Path]::GetFullPath((Join-Path $destination $relative))
    if (-not $target.StartsWith($destination, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe restore target path." }
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($target)) | Out-Null
    if ($PSCmdlet.ShouldProcess($target, "Download $($object.object_uri)")) {
        & gcloud storage cp --no-clobber $object.object_uri $target
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($object.object_uri)" }
    }
}
Write-Host "Download completed. No database restore, overwrite, cleanup, or production mutation was performed."
