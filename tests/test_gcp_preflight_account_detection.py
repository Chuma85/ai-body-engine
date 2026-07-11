import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gcp/preflight-live-upload.ps1"
ACCOUNT = "chuma@fashionaitechnologies.com"


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
    '{"iamConfiguration":{"publicAccessPrevention":"enforced","uniformBucketLevelAccess":{"enabled":true}}}'
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


def run_preflight(tmp_path: Path, mode: str = "valid", wrong_project: bool = False):
    mock, manifest, log = write_mock(tmp_path), write_manifest(tmp_path), tmp_path / "calls.log"
    log.unlink(missing_ok=True)
    env = os.environ | {
        "MOCK_ACCOUNT_MODE": mode,
        "MOCK_WRONG_PROJECT": "1" if wrong_project else "0",
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
    assert sum("storage buckets describe" in call for call in calls) == 4
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
