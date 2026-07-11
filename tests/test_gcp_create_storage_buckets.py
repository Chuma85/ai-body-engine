import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gcp/create-storage-buckets.ps1"
BUCKETS = [
    "fashionai-ai-body-datasets-501816",
    "fashionai-ai-body-models-501816",
    "fashionai-ai-body-artifacts-501816",
    "fashionai-database-backups-501816",
]


def make_gcloud(tmp_path: Path) -> Path:
    mock = tmp_path / "gcloud.ps1"
    mock.write_text(r'''
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$state = Get-Content -Raw $env:MOCK_GCLOUD_STATE | ConvertFrom-Json
$command = $Rest -join " "
Add-Content -Path $env:MOCK_GCLOUD_LOG -Value $command
$uri = $Rest | Where-Object { $_ -like "gs://*" } | Select-Object -First 1
$name = if ($uri) { $uri.Substring(5) } else { "" }
if ($Rest -contains "describe") {
    $mode = if ($state.PSObject.Properties.Name -contains $name) { $state.$name } else { "missing" }
    if ($mode -eq "exists") { '{"name":"' + $name + '","location":"NORTHAMERICA-NORTHEAST2","storageClass":"STANDARD","uniformBucketLevelAccess":true,"publicAccessPrevention":"enforced"}'; exit 0 }
    if ($mode -eq "permission") { Write-Error "ERROR: forbidden: 403 permission denied"; exit 1 }
    if ($mode -eq "conflict") { Write-Error "ERROR: bucket name is owned by another project: 403 inaccessible"; exit 1 }
    Write-Error "ERROR: $uri not found: 404"; exit 1
}
if ($Rest -contains "create") {
    $state | Add-Member -Force -NotePropertyName $name -NotePropertyValue "exists"
    $state | ConvertTo-Json | Set-Content $env:MOCK_GCLOUD_STATE
    exit 0
}
exit 2
''', encoding="utf-8")
    return mock


def run_script(tmp_path: Path, state: dict[str, str], execute: bool = True):
    mock = make_gcloud(tmp_path)
    state_path, log_path = tmp_path / "state.json", tmp_path / "calls.log"
    log_path.unlink(missing_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = os.environ | {
        "MOCK_GCLOUD_STATE": str(state_path),
        "MOCK_GCLOUD_LOG": str(log_path),
    }
    command = ["powershell", "-NoProfile", "-Command", f"Set-Alias gcloud '{mock}'; & '{SCRIPT}' -ProjectId test-project"]
    if execute:
        command[-1] += " -Execute"
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    return result, calls, json.loads(state_path.read_text(encoding="utf-8"))


def test_dry_run_with_missing_buckets_does_not_contact_gcloud(tmp_path: Path) -> None:
    result, calls, state = run_script(tmp_path, {}, execute=False)
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout and len(result.stdout.split("WOULD CHECK")) == 5
    assert calls == [] and state == {}


def test_execute_creates_simulated_404_buckets_and_second_run_is_idempotent(tmp_path: Path) -> None:
    result, calls, state = run_script(tmp_path, {})
    assert result.returncode == 0
    assert sum(" buckets create " in f" {call} " for call in calls) == 4
    assert all(state[name] == "exists" for name in BUCKETS)
    result, calls, _ = run_script(tmp_path, state)
    assert result.returncode == 0 and "SKIP existing bucket" in result.stdout
    assert not any(" buckets create " in f" {call} " for call in calls)


def test_existing_compliant_bucket_is_skipped(tmp_path: Path) -> None:
    result, calls, _ = run_script(tmp_path, {name: "exists" for name in BUCKETS})
    assert result.returncode == 0
    assert not any(" buckets create " in f" {call} " for call in calls)


def test_permission_failure_is_fatal(tmp_path: Path) -> None:
    result, calls, _ = run_script(tmp_path, {BUCKETS[0]: "permission"})
    assert result.returncode != 0 and "permission denied" in (result.stdout + result.stderr)
    assert not any(" buckets create " in f" {call} " for call in calls)


def test_globally_conflicting_name_is_fatal(tmp_path: Path) -> None:
    result, calls, _ = run_script(tmp_path, {BUCKETS[0]: "conflict"})
    assert result.returncode != 0 and "owned by another project" in (result.stdout + result.stderr)
    assert not any(" buckets create " in f" {call} " for call in calls)


def test_partial_state_skips_existing_and_creates_only_missing(tmp_path: Path) -> None:
    existing = {BUCKETS[0]: "exists", BUCKETS[2]: "exists"}
    result, calls, state = run_script(tmp_path, existing)
    creates = [call for call in calls if " buckets create " in f" {call} "]
    assert result.returncode == 0 and len(creates) == 2
    assert BUCKETS[1] in creates[0] and BUCKETS[3] in creates[1]
    assert all(state[name] == "exists" for name in BUCKETS)
