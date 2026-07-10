[CmdletBinding()]
param(
    [string[]]$ExpectedFiles = @(
        "README.md",
        "requirements.txt",
        "training/model_lifecycle.py",
        "cloudbuild/validate-ai-body-engine.yaml",
        "cloudbuild/build-ai-body-containers.yaml"
    )
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not ((& git -C $repoRoot remote) -contains "google")) { throw "Remote 'google' is not configured." }

function Read-Refs([string[]]$Lines, [string]$Prefix) {
    $map = @{}
    foreach ($line in $Lines) {
        if ($line -match '^([0-9a-f]{40})\s+(.+)$') {
            $sha = $Matches[1]; $ref = $Matches[2]
            if ($ref.StartsWith($Prefix)) { $map[$ref.Substring($Prefix.Length)] = $sha }
        }
    }
    return $map
}

$originUrlBefore = (& git -C $repoRoot remote get-url origin).Trim()
$localBranchLines = & git -C $repoRoot for-each-ref --format='%(objectname) %(refname)' refs/heads
$remoteBranchLines = & git -C $repoRoot ls-remote --heads google
if ($LASTEXITCODE -ne 0) { throw "Unable to read Google branch refs." }
$localBranches = Read-Refs $localBranchLines "refs/heads/"
$remoteBranches = Read-Refs $remoteBranchLines "refs/heads/"

$localTagLines = & git -C $repoRoot show-ref --tags -d
if ($LASTEXITCODE -notin @(0, 1)) { throw "Unable to read local tags." }
$remoteTagLines = & git -C $repoRoot ls-remote --tags google
if ($LASTEXITCODE -ne 0) { throw "Unable to read Google tags." }
$localTags = Read-Refs $localTagLines "refs/tags/"
$remoteTags = Read-Refs $remoteTagLines "refs/tags/"

$symrefLines = & git -C $repoRoot ls-remote --symref google HEAD
if ($LASTEXITCODE -ne 0) { throw "Unable to determine Google default branch." }
$defaultLine = $symrefLines | Where-Object { $_ -match '^ref:\s+refs/heads/' } | Select-Object -First 1
$googleDefault = if ($defaultLine -match '^ref:\s+refs/heads/([^\s]+)') { $Matches[1] } else { $null }
$localDefault = (& git -C $repoRoot symbolic-ref --short refs/remotes/origin/HEAD).Replace("origin/", "")

$branchMismatches = @()
foreach ($name in @(@($localBranches.Keys) + @($remoteBranches.Keys) | Sort-Object -Unique)) {
    if (-not $localBranches.ContainsKey($name) -or -not $remoteBranches.ContainsKey($name) -or $localBranches[$name] -ne $remoteBranches[$name]) {
        $branchMismatches += [pscustomobject]@{ branch = $name; local = $localBranches[$name]; google = $remoteBranches[$name] }
    }
}
$tagMismatches = @()
foreach ($name in @(@($localTags.Keys) + @($remoteTags.Keys) | Sort-Object -Unique)) {
    if (-not $localTags.ContainsKey($name) -or -not $remoteTags.ContainsKey($name) -or $localTags[$name] -ne $remoteTags[$name]) {
        $tagMismatches += [pscustomobject]@{ tag = $name; local = $localTags[$name]; google = $remoteTags[$name] }
    }
}

$missingFiles = @()
if ($googleDefault -and $remoteBranches.ContainsKey($googleDefault)) {
    $defaultSha = $remoteBranches[$googleDefault]
    & git -C $repoRoot cat-file -e "$defaultSha^{commit}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $remoteFiles = @(& git -C $repoRoot ls-tree -r --name-only $defaultSha)
        $missingFiles = @($ExpectedFiles | Where-Object { $_ -notin $remoteFiles })
    } else {
        $missingFiles = @("Unable to inspect Google default commit locally: $defaultSha")
    }
} else {
    $missingFiles = @("Google default branch is missing or unresolved.")
}

$originUrlAfter = (& git -C $repoRoot remote get-url origin).Trim()
if ($originUrlAfter -ne $originUrlBefore) { throw "Safety violation: origin changed during verification." }
$result = [ordered]@{
    match = ($branchMismatches.Count -eq 0 -and $tagMismatches.Count -eq 0 -and $googleDefault -eq $localDefault -and $missingFiles.Count -eq 0)
    local_default_branch = $localDefault
    google_default_branch = $googleDefault
    local_branch_count = $localBranches.Count
    google_branch_count = $remoteBranches.Count
    local_tag_count = $localTags.Count
    google_tag_count = $remoteTags.Count
    branch_mismatches = $branchMismatches
    tag_mismatches = $tagMismatches
    missing_expected_files = $missingFiles
    origin_unchanged = $true
}
$result | ConvertTo-Json -Depth 6
if (-not $result.match) { exit 2 }
