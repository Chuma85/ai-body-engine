[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectId,
    [Parameter(Mandatory = $true)][ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })][string]$ManifestPath
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
$manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$missing = @()
$mismatched = @()
$matched = 0

foreach ($expected in $manifest.objects) {
    $json = & gcloud storage objects describe $expected.gcs_uri --project=$ProjectId --format=json 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missing += [pscustomobject]@{ gcs_uri = $expected.gcs_uri; reason = "missing" }
        continue
    }
    $remote = $json | ConvertFrom-Json
    if ([int64]$remote.size -ne [int64]$expected.size_bytes) {
        $mismatched += [pscustomobject]@{ gcs_uri = $expected.gcs_uri; reason = "size"; expected = $expected.size_bytes; actual = $remote.size }
        continue
    }
    if ($expected.md5_base64 -and $remote.md5Hash -and $expected.md5_base64 -ne $remote.md5Hash) {
        $mismatched += [pscustomobject]@{ gcs_uri = $expected.gcs_uri; reason = "md5"; expected = $expected.md5_base64; actual = $remote.md5Hash }
        continue
    }
    # GCS exposes CRC32C/MD5 rather than SHA-256 for normal objects. Size is always checked;
    # SHA-256 remains in the manifest for local integrity and later metadata-based verification.
    $matched++
}

$expectedCount = @($manifest.objects).Count
$expectedBytes = ($manifest.objects | Measure-Object -Property size_bytes -Sum).Sum
$result = [pscustomobject]@{
    expected_object_count = $expectedCount
    expected_total_bytes = $expectedBytes
    matched_object_count = $matched
    missing_object_count = $missing.Count
    mismatched_object_count = $mismatched.Count
    missing = $missing
    mismatched = $mismatched
    sensitive_contents_read_or_printed = $false
}
$result | ConvertTo-Json -Depth 6
if ($missing.Count -or $mismatched.Count) { exit 2 }
