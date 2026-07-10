from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path, PurePosixPath
from typing import Any


class BackupIntegrityError(ValueError):
    pass


DATASET_CATEGORIES = {"synthetic datasets", "real-world datasets", "participant images", "verified exports", "evaluation holdouts"}
MODEL_CATEGORIES = {"candidate models", "promoted models", "archived models", "pretrained models", "model checkpoints"}
REPORT_CATEGORIES = {"evaluation reports", "comparison reports", "leakage audits"}


def create_backup_index(upload_manifest: dict[str, Any], retention_config: dict[str, Any], model_registry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(upload_manifest.get("objects"), list):
        raise BackupIntegrityError("Upload manifest must contain an objects array.")
    retention = _retention_by_category(retention_config)
    records_by_artifact = {item["artifact_uri"]: item for item in model_registry.get("versions", [])}
    records_by_report = {item["evaluation_report_uri"]: item for item in model_registry.get("versions", []) if item.get("evaluation_report_uri")}
    objects = []
    for source in upload_manifest["objects"]:
        uri = source["gcs_uri"]
        category = source["category"]
        model_record = records_by_artifact.get(uri) or records_by_report.get(uri)
        version = source.get("dataset_or_model_version") or source.get("source_dataset_version")
        git_sha = source.get("originating_git_sha") or upload_manifest.get("git_commit_sha")
        if model_record:
            version = model_record["model_version_id"] if uri == model_record["artifact_uri"] else model_record["model_version_id"]
            git_sha = model_record.get("git_commit_sha") or git_sha
        if not version:
            version = _derive_version(source.get("source_relative_path"), category)
        checksum = None
        if source.get("sha256"):
            checksum = {"algorithm": "sha256", "value": source["sha256"]}
        elif source.get("md5_base64"):
            checksum = {"algorithm": "md5_base64", "value": source["md5_base64"]}
        if not checksum:
            raise BackupIntegrityError(f"Backup object is missing a checksum: {uri}")
        objects.append({
            "object_uri": uri,
            "category": category,
            "size_bytes": int(source["size_bytes"]),
            "checksum": checksum,
            "creation_timestamp": source.get("creation_timestamp") or upload_manifest.get("generated_at_utc"),
            "dataset_or_model_version": version,
            "originating_git_sha": git_sha,
            "retention_classification": retention.get(category, "unclassified-manual-review"),
        })
    objects.sort(key=lambda item: item["object_uri"])
    return {
        "schema_version": "ai-body-backup-index-v1",
        "generated_at_utc": _utc_now(),
        "source_manifest_generated_at_utc": upload_manifest.get("generated_at_utc"),
        "object_count": len(objects),
        "total_size_bytes": sum(item["size_bytes"] for item in objects),
        "objects": objects,
    }


def check_integrity(
    backup_index: dict[str, Any],
    observed_objects: list[dict[str, Any]],
    *,
    model_registry: dict[str, Any] | None = None,
    training_runs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = {item["object_uri"]: item for item in backup_index.get("objects", [])}
    observed = {item["object_uri"]: item for item in observed_objects}
    missing = sorted(uri for uri in expected if uri not in observed)
    unexpected = sorted(uri for uri in observed if uri not in expected)
    mismatches = []
    for uri in sorted(expected.keys() & observed.keys()):
        wanted, actual = expected[uri], observed[uri]
        if int(wanted["size_bytes"]) != int(actual["size_bytes"]):
            mismatches.append({"object_uri": uri, "field": "size_bytes", "expected": wanted["size_bytes"], "actual": actual["size_bytes"]})
        checksum = wanted["checksum"]
        actual_value = actual.get(checksum["algorithm"])
        if actual_value is not None and actual_value != checksum["value"]:
            mismatches.append({"object_uri": uri, "field": checksum["algorithm"], "expected": checksum["value"], "actual": actual_value})
    orphan_report = check_orphaned_models(backup_index, model_registry or {"versions": []})
    missing_training_data = _missing_training_data(backup_index, training_runs or {"training_runs": []})
    evaluation_without_model = _evaluation_reports_without_models(backup_index, model_registry or {"versions": []})
    valid = not missing and not mismatches and not orphan_report["model_records_without_artifacts"] and not orphan_report["artifacts_without_model_records"] and not missing_training_data and not evaluation_without_model
    return {
        "valid": valid,
        "expected_object_count": len(expected),
        "observed_object_count": len(observed),
        "missing_objects": missing,
        "unexpected_objects": unexpected,
        "checksum_or_size_mismatches": mismatches,
        **orphan_report,
        "training_manifests_with_missing_data": missing_training_data,
        "evaluation_reports_with_missing_models": evaluation_without_model,
        "sensitive_contents_read_or_printed": False,
    }


def check_orphaned_models(backup_index: dict[str, Any], model_registry: dict[str, Any]) -> dict[str, Any]:
    indexed = {item["object_uri"]: item for item in backup_index.get("objects", [])}
    versions = model_registry.get("versions", [])
    registered_uris = {item["artifact_uri"] for item in versions if item.get("artifact_uri")}
    model_records_without_artifacts = sorted(item["model_version_id"] for item in versions if item.get("artifact_uri") not in indexed)
    artifacts_without_model_records = sorted(uri for uri, item in indexed.items() if item.get("category") in MODEL_CATEGORIES and uri not in registered_uris)
    return {
        "model_records_without_artifacts": model_records_without_artifacts,
        "artifacts_without_model_records": artifacts_without_model_records,
    }


def _missing_training_data(backup_index: dict[str, Any], training_runs: dict[str, Any]) -> list[dict[str, str]]:
    dataset_versions = {item.get("dataset_or_model_version") for item in backup_index.get("objects", []) if item.get("category") in DATASET_CATEGORIES}
    missing = []
    for run in training_runs.get("training_runs", []):
        manifest = run.get("training_manifest") or {}
        version = manifest.get("dataset_version")
        if version and version not in dataset_versions:
            missing.append({"training_run_id": str(run.get("training_run_id")), "dataset_version": str(version)})
    return missing


def _evaluation_reports_without_models(backup_index: dict[str, Any], model_registry: dict[str, Any]) -> list[dict[str, str]]:
    model_versions = {item.get("model_version_id") for item in model_registry.get("versions", [])}
    return [
        {"object_uri": item["object_uri"], "model_version": str(item["dataset_or_model_version"])}
        for item in backup_index.get("objects", [])
        if item.get("category") in REPORT_CATEGORIES and item.get("dataset_or_model_version") and item.get("dataset_or_model_version") not in model_versions
    ]


def _retention_by_category(config: dict[str, Any]) -> dict[str, str]:
    result = {}
    for classification, policy in config.get("classifications", {}).items():
        for category in policy.get("categories", []):
            result[category] = classification
    return result


def _derive_version(relative_path: str | None, category: str) -> str | None:
    if not relative_path:
        return None
    parts = PurePosixPath(relative_path).parts
    if category == "synthetic datasets" and len(parts) >= 3 and parts[:2] == ("data", "synthetic"):
        return parts[2]
    if category in {"real-world datasets", "participant images", "verified exports"} and len(parts) >= 3 and parts[:2] == ("data", "real_world"):
        return parts[2]
    return None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
