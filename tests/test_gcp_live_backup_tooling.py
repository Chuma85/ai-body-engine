from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_live_backup_has_explicit_approval_and_no_delete_controls() -> None:
    script = read("scripts/gcp/run-approved-backup.ps1")
    for token in ("$Execute", "$ApproveSyntheticData", "$ApproveModelAssets", "$ConfirmProject", "$IncludeDatabaseBackup", "$ApproveDatabaseBackup", "--no-clobber"):
        assert token in script
    assert "Remove-Item" not in script
    assert "gcloud ai" not in script


def test_preflight_is_upload_free_and_checks_privacy() -> None:
    script = read("scripts/gcp/preflight-live-upload.ps1")
    for token in ("--filter=status:ACTIVE", "--format=value(account)", "public_access_prevention", "uniform_bucket_level_access", "--raw", "storageClass", "NORTHAMERICA-NORTHEAST2", "real-world datasets", "participant images", "forbidden"):
        assert token in script
    assert "gcloud storage cp" not in script


def test_verifier_compares_counts_bytes_checksums_and_differences() -> None:
    script = read("scripts/gcp/post-upload-verification.ps1")
    for token in ("expected_object_count", "uploaded_object_count", "expected_total_bytes", "uploaded_total_bytes", "md5", "missing", "mismatched"):
        assert token in script


def test_real_world_uploads_remain_disabled() -> None:
    assert "real_world_uploads_enabled: false" in read("config/google-cloud/storage-layout.yaml")
