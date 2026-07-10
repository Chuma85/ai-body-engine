[CmdletBinding()]
param([string]$Tag = "local")
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker is required." }
& docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker daemon is unavailable." }

$images = @("body-engine-training", "body-engine-evaluation", "body-engine-inference", "dataset-validator")
foreach ($image in $images) {
    & docker run --rm "${image}:$Tag" --help
    if ($LASTEXITCODE -ne 0) { throw "Startup/help validation failed: $image" }
}

$fixture = Join-Path $repoRoot ".tmp/container-smoke/synthetic"
New-Item -ItemType Directory -Force -Path $fixture | Out-Null
& python -m synthetic.generator.generate_dataset --count 2 --output-dir $fixture --width 32 --height 32 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Tiny synthetic fixture generation failed." }
$mount = "$($fixture.Replace('\','/')):/inputs/synthetic:ro"
& docker run --rm --mount "type=bind,source=$($fixture),target=/inputs/synthetic,readonly" -e DATASET_URI=/inputs/synthetic -e DATASET_KIND=synthetic "dataset-validator:$Tag"
if ($LASTEXITCODE -ne 0) { throw "Dataset validator fixture smoke test failed." }
Write-Host "All container startup checks and tiny-fixture validation passed without participant data."
