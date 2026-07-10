from __future__ import annotations

import json
from pathlib import Path

import yaml

from training.backup_integrity import check_integrity, check_orphaned_models, create_backup_index

ROOT = Path(__file__).resolve().parents[1]


def test_retention_config_protects_sensitive_and_audit_classes() -> None:
    config = yaml.safe_load((ROOT / "config/google-cloud/retention-policy.yaml").read_text(encoding="utf-8"))
    assert config["automatic_deletion_enabled"] is False
    for name in ("real-world-datasets", "verified-exports", "promoted-models", "evaluation-reports", "database-dumps"):
        policy = config["classifications"][name]
        assert policy["cleanup_allowed"] is False
        assert policy["never_automatically_delete"] is True


def test_backup_index_contains_required_metadata() -> None:
    index = _index()
    model = next(item for item in index["objects"] if item["category"] == "candidate models")
    assert set(model) == {"object_uri", "category", "size_bytes", "checksum", "creation_timestamp", "dataset_or_model_version", "originating_git_sha", "retention_classification"}
    assert model["dataset_or_model_version"] == "model-v1"
    assert model["originating_git_sha"] == "a" * 40
    assert model["retention_classification"] == "candidate-models"


def test_integrity_success_with_matching_fixture_metadata() -> None:
    index = _index()
    result = check_integrity(index, _observed(index), model_registry=_registry(), training_runs=_training_runs())
    assert result["valid"] is True
    assert result["missing_objects"] == []
    assert result["checksum_or_size_mismatches"] == []


def test_missing_object_detection() -> None:
    index = _index()
    observed = _observed(index)[1:]
    result = check_integrity(index, observed, model_registry=_registry(), training_runs=_training_runs())
    assert result["valid"] is False
    assert result["missing_objects"] == [index["objects"][0]["object_uri"]]


def test_checksum_mismatch_detection() -> None:
    index = _index()
    observed = _observed(index)
    observed[0]["sha256"] = "f" * 64
    result = check_integrity(index, observed, model_registry=_registry(), training_runs=_training_runs())
    assert result["valid"] is False
    assert result["checksum_or_size_mismatches"][0]["field"] == "sha256"


def test_model_orphan_detection_in_both_directions() -> None:
    index = _index()
    registry = _registry()
    registry["versions"].append({**registry["versions"][0], "model_version_id": "missing-v2", "artifact_uri": "gs://models/missing/model.joblib"})
    model_artifact = next(item for item in index["objects"] if item["category"] == "candidate models")
    index["objects"].append({**model_artifact, "object_uri": "gs://models/orphan/model.pt", "dataset_or_model_version": "orphan"})
    result = check_orphaned_models(index, registry)
    assert result["model_records_without_artifacts"] == ["missing-v2"]
    assert result["artifacts_without_model_records"] == ["gs://models/orphan/model.pt"]


def test_training_and_evaluation_relationship_failures() -> None:
    index = _index()
    runs = _training_runs()
    runs["training_runs"][0]["training_manifest"]["dataset_version"] = "missing-dataset"
    report = next(item for item in index["objects"] if item["category"] == "evaluation reports")
    report["dataset_or_model_version"] = "missing-model"
    result = check_integrity(index, _observed(index), model_registry=_registry(), training_runs=runs)
    assert result["training_manifests_with_missing_data"][0]["dataset_version"] == "missing-dataset"
    assert result["evaluation_reports_with_missing_models"][0]["model_version"] == "missing-model"


def test_restore_documentation_uses_non_destructive_pg_restore_commands() -> None:
    text = (ROOT / "docs/google-cloud/DISASTER_RECOVERY_RUNBOOK.md").read_text(encoding="utf-8")
    assert "pg_restore --list $dump" in text
    assert "--exit-on-error --no-owner --no-privileges" in text
    assert "pg_restore --clean" not in text
    for table in ("User", "FieldDataCollectionSession", "FieldDataMeasurementRecord", "FieldDataPhotoVersion", "FieldDataSubmissionReviewAudit", "FieldDatasetLineage"):
        assert f'FROM "{table}"' in text


def _index() -> dict:
    config = yaml.safe_load((ROOT / "config/google-cloud/retention-policy.yaml").read_text(encoding="utf-8"))
    return create_backup_index(_upload_manifest(), config, _registry())


def _upload_manifest() -> dict:
    return {"generated_at_utc": "2026-07-10T12:00:00Z", "objects": [
        {"gcs_uri": "gs://datasets/synthetic/dataset-v1/labels.csv", "category": "synthetic datasets", "size_bytes": 10, "sha256": "1" * 64, "source_relative_path": "data/synthetic/dataset-v1/labels.csv"},
        {"gcs_uri": "gs://models/candidates/model-v1/model.joblib", "category": "candidate models", "size_bytes": 20, "sha256": "2" * 64},
        {"gcs_uri": "gs://artifacts/evaluations/model-v1/report.json", "category": "evaluation reports", "size_bytes": 30, "sha256": "3" * 64},
    ]}


def _registry() -> dict:
    return {"versions": [{"model_version_id": "model-v1", "artifact_uri": "gs://models/candidates/model-v1/model.joblib", "evaluation_report_uri": "gs://artifacts/evaluations/model-v1/report.json", "source_dataset_version": "dataset-v1", "git_commit_sha": "a" * 40}]}


def _training_runs() -> dict:
    return {"training_runs": [{"training_run_id": "run-v1", "training_manifest": {"dataset_version": "dataset-v1"}}]}


def _observed(index: dict) -> list[dict]:
    return [{"object_uri": item["object_uri"], "size_bytes": item["size_bytes"], item["checksum"]["algorithm"]: item["checksum"]["value"]} for item in index["objects"]]
