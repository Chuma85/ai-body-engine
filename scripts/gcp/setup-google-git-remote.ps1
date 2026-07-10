[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$RepositoryUrl,
    [switch]$ApproveReplaceGoogleRemote,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$originBefore = (& git -C $repoRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or -not $originBefore) { throw "The existing origin remote is required and will not be modified." }

$googleExists = (& git -C $repoRoot remote) -contains "google"
$googleBefore = if ($googleExists) { (& git -C $repoRoot remote get-url google).Trim() } else { $null }
if ($googleExists -and $googleBefore -ne $RepositoryUrl -and -not $ApproveReplaceGoogleRemote) {
    throw "Remote 'google' already exists with a different URL. Use -ApproveReplaceGoogleRemote explicitly to replace only that remote URL."
}

if (-not $Execute) {
    $action = if (-not $googleExists) { "add" } elseif ($googleBefore -eq $RepositoryUrl) { "keep" } else { "replace-with-approval" }
    Write-Host "DRY RUN: would $action remote 'google' with URL $RepositoryUrl"
} elseif (-not $googleExists) {
    if ($PSCmdlet.ShouldProcess("google", "Add Git remote")) { & git -C $repoRoot remote add google $RepositoryUrl }
} elseif ($googleBefore -ne $RepositoryUrl) {
    if ($PSCmdlet.ShouldProcess("google", "Replace explicitly approved Google remote URL")) { & git -C $repoRoot remote set-url google $RepositoryUrl }
} else {
    Write-Host "SKIP: remote 'google' already has the requested URL."
}
if ($LASTEXITCODE -ne 0) { throw "Git remote operation failed." }

$originAfter = (& git -C $repoRoot remote get-url origin).Trim()
if ($originAfter -ne $originBefore) { throw "Safety violation: origin changed." }
Write-Host "Origin unchanged: $originAfter"
Write-Host "Review, then push without force:"
Write-Host "  git push google --all"
Write-Host "  git push google --tags"
