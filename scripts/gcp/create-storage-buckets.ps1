[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$ProjectId,
    [ValidateNotNullOrEmpty()][string]$Region = "northamerica-northeast2",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$buckets = @(
    "fashionai-ai-body-datasets-501816",
    "fashionai-ai-body-models-501816",
    "fashionai-ai-body-artifacts-501816",
    "fashionai-database-backups-501816"
)

function Invoke-GcloudCapture {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePreference) { $previousNativePreference = $PSNativeCommandUseErrorActionPreference }
    try {
        # A non-zero native exit is data here. Inspect it before deciding whether
        # it is the one expected missing-bucket case or a fatal gcloud failure.
        $ErrorActionPreference = "Continue"
        if ($hasNativePreference) { $PSNativeCommandUseErrorActionPreference = $false }
        $output = @(& gcloud @Arguments 2>&1 | ForEach-Object { $_.ToString() })
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) { $PSNativeCommandUseErrorActionPreference = $previousNativePreference }
    }
    [pscustomobject]@{ ExitCode = $exitCode; Output = $output }
}

function Get-BucketState {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Project
    )

    $result = Invoke-GcloudCapture -Arguments @(
        "storage", "buckets", "describe", $Uri, "--project=$Project", "--format=json"
    )
    if ($result.ExitCode -eq 0) {
        try { $null = ($result.Output -join [Environment]::NewLine) | ConvertFrom-Json }
        catch { throw "gcloud returned invalid bucket metadata for ${Uri}: $($_.Exception.Message)" }
        return "Exists"
    }

    $message = $result.Output -join [Environment]::NewLine
    if ($message -match '(?i)\b404\b' -and $message -match '(?i)\bnot[ -]?found\b') {
        return "Missing"
    }
    throw "Unable to determine bucket state for $Uri (gcloud exit $($result.ExitCode)). $message"
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud CLI is required." }
if (-not $Execute) {
    Write-Host "DRY RUN: no buckets will be created. Re-run with -Execute after review."
}

foreach ($bucket in $buckets) {
    $uri = "gs://$bucket"
    if (-not $Execute) {
        Write-Host "WOULD CHECK AND CREATE IF MISSING: $uri (STANDARD, $Region, uniform access, public access prevention, Google-managed encryption)"
        continue
    }
    $state = Get-BucketState -Uri $uri -Project $ProjectId
    if ($state -eq "Exists") {
        Write-Host "SKIP existing bucket: $uri"
        continue
    }
    if ($PSCmdlet.ShouldProcess($uri, "Create private regional Standard bucket")) {
        & gcloud storage buckets create $uri --project=$ProjectId --location=$Region --default-storage-class=STANDARD --uniform-bucket-level-access --public-access-prevention
        if ($LASTEXITCODE -ne 0) { throw "Bucket creation failed: $uri" }
    }
}

Write-Host "No existing bucket was deleted or made public. Google-managed encryption is the default."
