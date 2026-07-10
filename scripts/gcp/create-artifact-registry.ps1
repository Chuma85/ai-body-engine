[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectId = "fashionai-501816",
    [string]$Region = "northamerica-northeast2",
    [string]$Repository = "fashionai-containers",
    [switch]$Execute
)
$ErrorActionPreference = "Stop"
if ($ProjectId -ne "fashionai-501816") { throw "GCP-C is scoped to project fashionai-501816." }
if (-not $Execute) {
    Write-Host "DRY RUN: would check Docker repository $Repository in $Region and create it only if absent."
    exit 0
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
& gcloud artifacts repositories describe $Repository --project=$ProjectId --location=$Region *> $null
if ($LASTEXITCODE -eq 0) { Write-Host "SKIP existing repository: $Repository"; exit 0 }
if ($PSCmdlet.ShouldProcess("$Region/$Repository", "Create Docker Artifact Registry repository")) {
    & gcloud artifacts repositories create $Repository --project=$ProjectId --location=$Region --repository-format=docker --description="AI Body Engine reproducible workload containers"
    if ($LASTEXITCODE -ne 0) { throw "Artifact Registry creation failed." }
}
