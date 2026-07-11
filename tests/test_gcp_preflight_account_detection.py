import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gcp/preflight-live-upload.ps1"
ACCOUNT = "chuma@fashionaitechnologies.com"
BUCKETS = {
    "fashionai-ai-body-datasets-501816",
    "fashionai-ai-body-models-501816",
    "fashionai-ai-body-artifacts-501816",
    "fashionai-database-backups-501816",
}


def write_mock(tmp_path: Path) -> Path:
    mock = tmp_path / "gcloud.ps1"
    mock.write_text(r'''
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$mode = $env:MOCK_ACCOUNT_MODE
$command = $Rest -join " "
Add-Content -LiteralPath $env:MOCK_GCLOUD_LOG -Value $command
if ($Rest[0] -eq "auth") {
    if ($mode -eq "error") { Write-Error "authentication service unavailable"; exit 7 }
    if ($mode -eq "none") { exit 0 }
    if ($mode -eq "multiple") { "first@example.com"; "second@example.com"; exit 0 }
    if ($mode -eq "malformed") { "not-an-account"; exit 0 }
    if ($mode -eq "warning") { Write-Error "WARNING: optional component update available" }
    if ($mode -eq "trailing") { Write-Output "chuma@fashionaitechnologies.com`n"; exit 0 }
    "chuma@fashionaitechnologies.com"; exit 0
}
if ($Rest[0] -eq "config" -and $Rest[2] -eq "account") { "chuma@fashionaitechnologies.com"; exit 0 }
if ($Rest[0] -eq "config" -and $Rest[2] -eq "project") { if ($env:MOCK_WRONG_PROJECT -eq "1") { "wrong-project" } else { "fashionai-501816" }; exit 0 }
if ($Rest -contains "describe") {
    $format = $Rest | Where-Object { $_ -like "--format=*" } | Select-Object -First 1
    $setting = $format -replace '^--format=value\(', '' -replace '\)$', ''
    $bucketMode = $env:MOCK_BUCKET_MODE
    if ($bucketMode -eq "gcloud_error") { Write-Error "permission denied"; exit 9 }
    if ($bucketMode -eq "warning") { Write-Error "WARNING: benign update notice" }
    if ($bucketMode -eq "missing_public" -and $setting -eq "public_access_prevention") { exit 0 }
    if ($bucketMode -eq "malformed_public" -and $setting -eq "public_access_prevention") { "enforced"; "extra"; exit 0 }
    if ($setting -eq "public_access_prevention") { if ($bucketMode -eq "inherited") { "inherited" } elseif ($bucketMode -eq "normalized") { " EnFoRcEd `n" } else { "enforced" }; exit 0 }
    if ($setting -eq "uniform_bucket_level_access") { if ($bucketMode -eq "uniform_false") { "false" } elseif ($bucketMode -eq "normalized") { " TrUe `n" } else { "true" }; exit 0 }
    if ($setting -eq "location") { if ($bucketMode -eq "wrong_region") { "US" } elseif ($bucketMode -eq "normalized") { "northamerica-northeast2" } else { "NORTHAMERICA-NORTHEAST2" }; exit 0 }
    if ($setting -eq "storage_class") { if ($bucketMode -eq "wrong_class") { "NEARLINE" } elseif ($bucketMode -eq "normalized") { "standard" } else { "STANDARD" }; exit 0 }
    exit 0
}
if ($Rest -contains "cp") { Write-Error "upload must never be invoked"; exit 99 }
exit 2
''', encoding="utf-8")
    return mock


def write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "database_backup_approved": False,
        "objects": [],
        "summary": {"object_count": 0, "total_size_bytes": 0},
    }), encoding="utf-8")
    return manifest


def run_preflight(tmp_path: Path, mode: str = "valid", wrong_project: bool = False, bucket_mode: str = "valid"):
    mock, manifest, log = write_mock(tmp_path), write_manifest(tmp_path), tmp_path / "calls.log"
    log.unlink(missing_ok=True)
    env = os.environ | {
        "MOCK_ACCOUNT_MODE": mode,
        "MOCK_WRONG_PROJECT": "1" if wrong_project else "0",
        "MOCK_BUCKET_MODE": bucket_mode,
        "MOCK_GCLOUD_LOG": str(log),
    }
    command = (
        f"Set-Alias gcloud '{mock}'; "
        f"& '{SCRIPT}' -ManifestPath '{manifest}'"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], cwd=ROOT, env=env, text=True, capture_output=True)
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_valid_single_active_account_and_full_preflight(tmp_path: Path) -> None:
    result, calls = run_preflight(tmp_path)
    assert result.returncode == 0
    assert f"PASS authenticated account: {ACCOUNT}" in result.stdout
    assert sum("storage buckets describe" in call for call in calls) == 16
    assert not any(" storage cp " in f" {call} " for call in calls)


def test_valid_account_with_trailing_newline(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, "trailing")
    assert result.returncode == 0


def test_valid_powershell_array_output_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, "multiple")
    assert result.returncode != 0 and "found 2" in (result.stdout + result.stderr)


def test_no_active_account_even_when_configured_is_rejected(tmp_path: Path) -> None:
    result, calls = run_preflight(tmp_path, "none")
    assert result.returncode != 0 and "no active authenticated account" in (result.stdout + result.stderr)
    assert not any("config get-value account" in call for call in calls)


def test_malformed_account_output_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, "malformed")
    assert result.returncode != 0 and "malformed" in (result.stdout + result.stderr)


def test_nonzero_gcloud_exit_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, "error")
    assert result.returncode != 0 and "exit 7" in (result.stdout + result.stderr)


def test_stderr_warning_does_not_contaminate_valid_stdout(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, "warning")
    assert result.returncode == 0


def test_wrong_project_is_rejected(tmp_path: Path) -> None:
    result, calls = run_preflight(tmp_path, wrong_project=True)
    assert result.returncode != 0 and "wrong-project" in (result.stdout + result.stderr)
    assert not any("storage buckets describe" in call for call in calls)


def test_bucket_values_are_case_insensitive_and_trimmed(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="normalized")
    assert result.returncode == 0


def test_inherited_public_access_prevention_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="inherited")
    assert result.returncode != 0 and "expected 'enforced', found 'inherited'" in (result.stdout + result.stderr)


def test_missing_public_access_prevention_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="missing_public")
    assert result.returncode != 0 and "public_access_prevention' is missing or malformed" in (result.stdout + result.stderr)


def test_malformed_bucket_value_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="malformed_public")
    assert result.returncode != 0 and "expected exactly one value" in (result.stdout + result.stderr)


def test_uniform_access_false_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="uniform_false")
    assert result.returncode != 0 and "uniform bucket-level access" in (result.stdout + result.stderr)


def test_wrong_region_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="wrong_region")
    assert result.returncode != 0 and "failed location validation" in (result.stdout + result.stderr)


def test_non_standard_storage_class_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="wrong_class")
    assert result.returncode != 0 and "failed storage class validation" in (result.stdout + result.stderr)


def test_bucket_describe_nonzero_exit_is_rejected(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="gcloud_error")
    assert result.returncode != 0 and "gcloud exit 9" in (result.stdout + result.stderr)


def test_bucket_stderr_warning_does_not_replace_valid_stdout(tmp_path: Path) -> None:
    result, _ = run_preflight(tmp_path, bucket_mode="warning")
    assert result.returncode == 0


def test_preflight_invokes_no_mutating_gcloud_commands(tmp_path: Path) -> None:
    result, calls = run_preflight(tmp_path)
    assert result.returncode == 0
    describe_calls = [call for call in calls if "storage buckets describe" in call]
    assert {bucket for bucket in BUCKETS if any(f"gs://{bucket}" in call for call in describe_calls)} == BUCKETS
    assert all(any(f"gs://{bucket}" in call for bucket in BUCKETS) for call in describe_calls)
    forbidden = (" buckets create ", " buckets update ", " storage cp ", " rm ", " delete ")
    assert not any(any(token in f" {call} " for token in forbidden) for call in calls)
