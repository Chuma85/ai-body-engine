[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })][string]$ManifestPath,
    [ValidateNotNullOrEmpty()][string]$ProjectId = "fashionai-501816"
)

$ErrorActionPreference = "Stop"
if ($ProjectId -ne "fashionai-501816") { throw "Project must be fashionai-501816." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
$manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$missing = [System.Collections.Generic.List[object]]::new()
$mismatched = [System.Collections.Generic.List[object]]::new()
$verifiedCount = 0
[int64]$verifiedBytes = 0
foreach ($expected in $manifest.objects) {
    $json = & gcloud storage objects describe $expected.gcs_uri --project=$ProjectId --format=json 2>$null
    if ($LASTEXITCODE -ne 0) { $missing.Add($expected); continue }
    $actual = $json | ConvertFrom-Json
    $verifiedCount++
    $verifiedBytes += [int64]$actual.size
    $reasons = @()
    if ([int64]$actual.size -ne [int64]$expected.size_bytes) { $reasons += "size" }
    if ($expected.md5_base64 -and $actual.md5Hash -and $actual.md5Hash -ne $expected.md5_base64) { $reasons += "md5" }
    if ($reasons.Count) { $mismatched.Add([pscustomobject]@{ gcs_uri=$expected.gcs_uri; reasons=$reasons -join ',' }) }
}
$summary = [pscustomobject]@{
    expected_object_count = [int64]$manifest.summary.object_count
    uploaded_object_count = $verifiedCount
    expected_total_bytes = [int64]$manifest.summary.total_size_bytes
    uploaded_total_bytes = $verifiedBytes
    missing_object_count = $missing.Count
    mismatched_object_count = $mismatched.Count
    checksums_compared_where_available = $true
}
$summary | Format-List
if ($missing.Count) { Write-Host "Missing objects:"; $missing | Select-Object gcs_uri | Format-Table -AutoSize }
if ($mismatched.Count) { Write-Host "Mismatched objects:"; $mismatched | Format-Table -AutoSize }
if ($missing.Count -or $mismatched.Count -or $verifiedCount -ne $manifest.summary.object_count -or $verifiedBytes -ne $manifest.summary.total_size_bytes) { throw "Post-upload verification failed." }
Write-Host "PASS: object count, total bytes, and available checksums match the manifest."
