from __future__ import annotations

import json
from pathlib import Path

import pytest
import jsonschema
import yaml

from training.vertex_model_registry import (
    GoogleVertexRegistryClient,
    RegistrySettings,
    VertexRegistryError,
    load_registry,
    promote_version,
    register_candidate,
    rollback_version,
)


class MockVertexClient:
    def __init__(self, *, artifact_exists: bool = True) -> None:
        self.exists = artifact_exists
        self.registered: list[dict] = []
        self.updated: list[tuple[str, dict[str, str]]] = []

    def artifact_exists(self, artifact_uri: str) -> bool:
        return self.exists

    def register_version(self, record: dict) -> str:
        self.registered.append(record.copy())
        return f"projects/test/locations/test/models/1@{len(self.registered)}"

    def update_version_metadata(self, vertex_resource_name: str, labels: dict[str, str]) -> None:
        self.updated.append((vertex_resource_name, labels.copy()))

    def list_versions(self, model_name: str) -> list[dict]:
        return self.registered


def test_google_adapter_attaches_new_upload_to_existing_parent() -> None:
    captured: dict = {}

    class Existing:
        resource_name = "projects/p/locations/r/models/123@1"

    class Uploaded:
        resource_name = "projects/p/locations/r/models/123@2"

    class ModelApi:
        @staticmethod
        def list(*, filter: str) -> list[Existing]:
            return [Existing()]

        @staticmethod
        def upload(**kwargs: object) -> Uploaded:
            captured.update(kwargs)
            return Uploaded()

    client = GoogleVertexRegistryClient.__new__(GoogleVertexRegistryClient)
    client._aiplatform = type("FakePlatform", (), {"Model": ModelApi})
    client._settings = RegistrySettings()
    resource = client.register_version(_metadata("candidate-v2", "run-v2", "dataset-v2") | {"labels": {}, "compatibility_metadata": {"status": "passed"}})
    assert resource.endswith("@2")
    assert captured["parent_name"] == "projects/p/locations/r/models/123"
    assert captured["is_default_version"] is False


def test_candidate_registration_with_mock_is_candidate_only(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    registry = tmp_path / "vertex.json"
    client = MockVertexClient()
    result = register_candidate(_metadata("candidate-v1", "run-v1", "dataset-v1"), client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True, created_at="2026-07-10T12:00:00Z")

    assert result["record"]["lifecycle_status"] == "candidate"
    assert result["record"]["labels"]["lifecycle_status"] == "candidate"
    assert load_registry(registry)["current_promoted_version_id"] is None
    assert len(client.registered) == 1
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/model-registry-record.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(result["record"], schema)


def test_registry_configuration_preserves_manual_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/google-cloud/model-registry.yaml").read_text(encoding="utf-8"))
    assert config["project_id"] == "fashionai-501816"
    assert config["region"] == "northamerica-northeast2"
    assert config["policy"]["auto_promote"] is False
    assert config["policy"]["auto_deploy"] is False
    assert config["policy"]["create_endpoint"] is False


def test_registration_requires_gcs_artifact(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    with pytest.raises(VertexRegistryError, match="does not exist"):
        register_candidate(_metadata("candidate-v1", "run-v1", "dataset-v1"), client=MockVertexClient(artifact_exists=False), registry_path=tmp_path / "vertex.json", lifecycle_root=lifecycle)


def test_registration_dry_run_does_not_write_registry(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    registry = tmp_path / "vertex.json"
    result = register_candidate(_metadata("candidate-v1", "run-v1", "dataset-v1"), client=MockVertexClient(), registry_path=registry, lifecycle_root=lifecycle)
    assert result["dry_run"] is True
    assert not registry.exists()


def test_promotion_blocked_without_approval(tmp_path: Path) -> None:
    lifecycle, registry, client = _registered(tmp_path)
    with pytest.raises(VertexRegistryError, match="approval identity"):
        promote_version("candidate-v1", approval_identity="", approval_reference="CHANGE-1", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)


def test_promotion_blocked_when_leakage_audit_fails(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    registry = tmp_path / "vertex.json"
    client = MockVertexClient()
    metadata = _metadata("candidate-v1", "run-v1", "dataset-v1")
    metadata["leakage_audit_status"] = "failed"
    register_candidate(metadata, client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    with pytest.raises(VertexRegistryError, match="leakage audit"):
        promote_version("candidate-v1", approval_identity="reviewer", approval_reference="CHANGE-1", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)


def test_successful_explicit_promotion_writes_both_registries(tmp_path: Path) -> None:
    lifecycle, registry, client = _registered(tmp_path)
    preview = promote_version("candidate-v1", approval_identity="reviewer", approval_reference="CHANGE-1", client=client, registry_path=registry, lifecycle_root=lifecycle)
    assert preview["dry_run"] is True
    assert not (lifecycle / "promotion_decisions.json").exists()
    result = promote_version("candidate-v1", approval_identity="reviewer", approval_reference="CHANGE-1", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True, promoted_at="2026-07-10T13:00:00Z")

    local_vertex = load_registry(registry)
    production = json.loads((lifecycle / "production_models.json").read_text(encoding="utf-8"))
    decisions = json.loads((lifecycle / "promotion_decisions.json").read_text(encoding="utf-8"))
    assert result["record"]["lifecycle_status"] == "promoted"
    assert local_vertex["current_promoted_version_id"] == "candidate-v1"
    assert production["current_production_model"] == "candidate-v1"
    assert decisions["promotion_decisions"][0]["decision_id"] == "CHANGE-1"
    assert client.updated[-1][1]["lifecycle_status"] == "promoted"


def test_rollback_changes_pointer_without_deleting_history(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    _add_local_candidate(lifecycle, "candidate-v2", "run-v2", "dataset-v2")
    registry = tmp_path / "vertex.json"
    client = MockVertexClient()
    register_candidate(_metadata("candidate-v1", "run-v1", "dataset-v1"), client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    register_candidate(_metadata("candidate-v2", "run-v2", "dataset-v2"), client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    promote_version("candidate-v1", approval_identity="reviewer", approval_reference="CHANGE-1", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    promote_version("candidate-v2", approval_identity="reviewer", approval_reference="CHANGE-2", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)

    preview = rollback_version("candidate-v1", rolled_back_by="release-manager", reason="regression", client=client, registry_path=registry, lifecycle_root=lifecycle)
    assert preview["dry_run"] is True
    assert load_registry(registry)["rollback_history"] == []
    result = rollback_version("candidate-v1", rolled_back_by="release-manager", reason="regression", client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    after = load_registry(registry)
    assert result["rollback"]["from_model_version_id"] == "candidate-v2"
    assert after["current_promoted_version_id"] == "candidate-v1"
    assert len(after["versions"]) == 2
    assert len(after["promotion_history"]) == 2
    assert len(after["rollback_history"]) == 1


def _registered(tmp_path: Path) -> tuple[Path, Path, MockVertexClient]:
    lifecycle = _lifecycle(tmp_path, "candidate-v1", "run-v1", "dataset-v1")
    registry = tmp_path / "vertex.json"
    client = MockVertexClient()
    register_candidate(_metadata("candidate-v1", "run-v1", "dataset-v1"), client=client, registry_path=registry, lifecycle_root=lifecycle, execute=True)
    return lifecycle, registry, client


def _lifecycle(tmp_path: Path, model: str, run: str, dataset: str) -> Path:
    root = tmp_path / "lifecycle"
    root.mkdir()
    (root / "model_registry.json").write_text(json.dumps({"models": [_model(model, run, dataset)]}), encoding="utf-8")
    (root / "training_runs.json").write_text(json.dumps({"training_runs": [_run(run, model, dataset)]}), encoding="utf-8")
    return root


def _add_local_candidate(root: Path, model: str, run: str, dataset: str) -> None:
    models = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    runs = json.loads((root / "training_runs.json").read_text(encoding="utf-8"))
    models["models"].append(_model(model, run, dataset))
    runs["training_runs"].append(_run(run, model, dataset))
    (root / "model_registry.json").write_text(json.dumps(models), encoding="utf-8")
    (root / "training_runs.json").write_text(json.dumps(runs), encoding="utf-8")


def _model(model: str, run: str, dataset: str) -> dict:
    return {"model_version": model, "model_type": "measurement_regressor", "parent_model_version": "base-v1", "training_run_id": run, "training_dataset_versions": [dataset], "created_at": "2026-07-10T11:00:00Z", "status": "evaluation_pending", "lineage": {"training_runs": [run], "training_datasets": [dataset], "training_manifests": [], "evaluation_reports": []}}


def _run(run: str, model: str, dataset: str) -> dict:
    return {"training_run_id": run, "dataset_version": dataset, "model_version": model, "model_base_version": "base-v1", "status": "completed", "metrics": {"mae": 2.4}, "training_manifest": {"training_execution": {"auto_train": False}}}


def _metadata(model: str, run: str, dataset: str) -> dict:
    return {"model_version_id": model, "artifact_uri": f"gs://fashionai-ai-body-models-501816/candidates/{model}/model.joblib", "model_format": "joblib", "training_run_id": run, "source_dataset_version": dataset, "git_commit_sha": "a" * 40, "metrics": {"evaluation_status": "passed", "clean_synthetic_mae": 2.4, "mobile_realistic_mae": 3.1}, "evaluation_report_uri": f"gs://fashionai-ai-body-artifacts-501816/evaluations/{model}/report.json", "leakage_audit_status": "passed", "compatibility_metadata": {"status": "passed", "architecture": "measurement_regressor"}, "candidate_status": "candidate", "architecture_backbone": "measurement-regressor-v1"}
