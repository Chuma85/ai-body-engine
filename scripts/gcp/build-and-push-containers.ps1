[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectId = "fashionai-501816",
    [string]$Region = "northamerica-northeast2",
    [string]$Repository = "fashionai-containers",
    [string]$VersionTag,
    [switch]$Execute
)
$ErrorActionPreference = "Stop"
if ($ProjectId -ne "fashionai-501816") { throw "GCP-C is scoped to project fashionai-501816." }
if ($VersionTag -and $VersionTag -match '^(latest|production|promoted)$') { throw "Reserved mutable/promoted tag is not allowed: $VersionTag" }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$sha = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sha -notmatch '^[0-9a-f]{40}$') { throw "A full Git commit SHA is required." }
$images = @("body-engine-training", "body-engine-evaluation", "body-engine-inference", "dataset-validator")
$base = "body-engine-base:$sha"
if (-not $Execute) {
    Write-Host "DRY RUN: no images will be built or pushed."
    $images | ForEach-Object { Write-Host "$Region-docker.pkg.dev/$ProjectId/$Repository/${_}:$sha" }
    exit 0
}
$dirty = & git -C $repoRoot status --porcelain
if ($dirty) { throw "Refusing an immutable SHA build from a dirty working tree." }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker is required." }
& docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker daemon is unavailable." }
& gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
if ($LASTEXITCODE -ne 0) { throw "Artifact Registry Docker authentication failed." }
& docker build --file (Join-Path $repoRoot "containers/base/Dockerfile") --tag $base $repoRoot
if ($LASTEXITCODE -ne 0) { throw "Base image build failed." }
foreach ($image in $images) {
    $folder = if ($image -eq "dataset-validator") { "dataset-validator" } else { $image.Replace("body-engine-", "") }
    $uri = "$Region-docker.pkg.dev/$ProjectId/$Repository/${image}:$sha"
    & gcloud artifacts docker images describe $uri --project=$ProjectId *> $null
    if ($LASTEXITCODE -eq 0) { throw "Refusing to overwrite existing immutable SHA tag: $uri" }
    if ($PSCmdlet.ShouldProcess($uri, "Build and push immutable image")) {
        & docker build --file (Join-Path $repoRoot "containers/$folder/Dockerfile") --build-arg "BASE_IMAGE=$base" --tag $uri $repoRoot
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $image" }
        & docker push $uri
        if ($LASTEXITCODE -ne 0) { throw "Push failed: $uri" }
        Write-Host $uri
    }
    if ($VersionTag) {
        $versionUri = "$Region-docker.pkg.dev/$ProjectId/$Repository/${image}:$VersionTag"
        & gcloud artifacts docker images describe $versionUri --project=$ProjectId *> $null
        if ($LASTEXITCODE -eq 0) { throw "Refusing to overwrite existing version tag: $versionUri" }
        & gcloud artifacts docker tags add $uri $versionUri --project=$ProjectId
        if ($LASTEXITCODE -ne 0) { throw "Version tag creation failed: $versionUri" }
        Write-Host $versionUri
    }
}
