from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Protocol
from urllib.parse import urlparse

from training.model_lifecycle import (
    ModelLifecycleError,
    create_promotion_decision,
    promote_model_to_production,
    rollback_production_model,
)

SUPPORTED_FORMATS = {"pkl", "joblib", "pt", "pth", "onnx", "ckpt", "h5", "keras", "pb", "savedmodel"}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class VertexRegistryError(ValueError):
    pass


class VertexRegistryClient(Protocol):
    def artifact_exists(self, artifact_uri: str) -> bool: ...
    def register_version(self, record: dict[str, Any]) -> str: ...
    def update_version_metadata(self, vertex_resource_name: str, labels: dict[str, str]) -> None: ...
    def list_versions(self, model_name: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class RegistrySettings:
    project_id: str = "fashionai-501816"
    region: str = "northamerica-northeast2"
    model_name: str = "fashionai-body-measurement"
    artifact_bucket: str = "gs://fashionai-ai-body-models-501816"


class GoogleVertexRegistryClient:
    """Lazy Google SDK adapter so local tests do not require credentials or SDK packages."""

    def __init__(self, settings: RegistrySettings) -> None:
        try:
            from google.cloud import aiplatform, storage
        except ImportError as exc:
            raise VertexRegistryError("Install google-cloud-aiplatform and google-cloud-storage for live execution.") from exc
        self._aiplatform = aiplatform
        self._storage = storage
        self._settings = settings
        aiplatform.init(project=settings.project_id, location=settings.region)

    def artifact_exists(self, artifact_uri: str) -> bool:
        parsed = urlparse(artifact_uri)
        bucket = self._storage.Client(project=self._settings.project_id).bucket(parsed.netloc)
        key = parsed.path.lstrip("/")
        if Path(key).suffix:
            return bucket.blob(key).exists()
        return next(self._storage.Client(project=self._settings.project_id).list_blobs(bucket, prefix=key.rstrip("/") + "/", max_results=1), None) is not None

    def register_version(self, record: dict[str, Any]) -> str:
        existing = self._aiplatform.Model.list(filter=f'display_name="{self._settings.model_name}"')
        upload_options = dict(
            display_name=self._settings.model_name,
            artifact_uri=record["artifact_uri"],
            labels=record["labels"],
            is_default_version=False,
            version_aliases=[_label(record["model_version_id"])],
            version_description=f"AI Body Engine candidate {record['model_version_id']}; explicit promotion required.",
            sync=True,
        )
        if existing:
            upload_options["parent_name"] = existing[0].resource_name.split("@", 1)[0]
        if record["compatibility_metadata"].get("serving_container_image_uri"):
            upload_options["serving_container_image_uri"] = record["compatibility_metadata"]["serving_container_image_uri"]
        model = self._aiplatform.Model.upload(**upload_options)
        return model.resource_name

    def update_version_metadata(self, vertex_resource_name: str, labels: dict[str, str]) -> None:
        self._aiplatform.Model(vertex_resource_name).update(labels=labels)

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        models = self._aiplatform.Model.list(filter=f'display_name="{model_name}"')
        return [{"resource_name": item.resource_name, "display_name": item.display_name, "labels": dict(item.labels or {})} for item in models]


def load_registry(path: str | Path, model_name: str = "fashionai-body-measurement") -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return _empty_registry(model_name)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("versions"), list):
        raise VertexRegistryError("Local Vertex registry must contain a versions array.")
    return payload


def register_candidate(
    metadata: dict[str, Any],
    *,
    client: VertexRegistryClient,
    registry_path: str | Path,
    lifecycle_root: str | Path,
    settings: RegistrySettings = RegistrySettings(),
    execute: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    record = _build_candidate_record(metadata, settings, created_at=created_at)
    _require_local_lineage(record, Path(lifecycle_root))
    if not client.artifact_exists(record["artifact_uri"]):
        raise VertexRegistryError("Model artifact does not exist in Google Cloud Storage.")
    registry = load_registry(registry_path, settings.model_name)
    if any(item["model_version_id"] == record["model_version_id"] for item in registry["versions"]):
        raise VertexRegistryError(f"Model version already registered: {record['model_version_id']}")
    if not execute:
        return {"dry_run": True, "record": record}
    record["vertex_resource_name"] = client.register_version(record)
    registry["versions"].append(record)
    _write_json(Path(registry_path), registry)
    return {"dry_run": False, "record": record}


def promote_version(
    model_version_id: str,
    *,
    approval_identity: str,
    approval_reference: str,
    client: VertexRegistryClient,
    registry_path: str | Path,
    lifecycle_root: str | Path,
    execute: bool = False,
    promoted_at: str | None = None,
) -> dict[str, Any]:
    if not approval_identity or not approval_reference:
        raise VertexRegistryError("Promotion requires approval identity and approval reference.")
    registry = load_registry(registry_path)
    record = _require_version(registry, model_version_id)
    _require_promotion_gates(record)
    preview = {"model_version_id": model_version_id, "approval_identity": approval_identity, "approval_reference": approval_reference, "previous_version_id": registry.get("current_promoted_version_id")}
    if not execute:
        return {"dry_run": True, "promotion": preview}
    decision = create_promotion_decision(
        model_version_id,
        status="approved_for_production",
        decided_by=approval_identity,
        decision_id=approval_reference,
        notes="Explicit Vertex Model Registry promotion approval.",
        decided_at=promoted_at,
        lifecycle_root=lifecycle_root,
    )
    promote_model_to_production(
        model_version_id,
        decision_id=decision["decision_id"],
        promoted_by=approval_identity,
        notes=f"Vertex approval reference: {approval_reference}",
        promoted_at=promoted_at,
        lifecycle_root=lifecycle_root,
    )
    previous_id = registry.get("current_promoted_version_id")
    if previous_id and previous_id != model_version_id:
        previous = _require_version(registry, previous_id)
        previous["lifecycle_status"] = "previously_promoted"
        _sync_labels(previous)
        _update_vertex(client, previous)
    record["lifecycle_status"] = "promoted"
    record["approval_identity"] = approval_identity
    record["approval_reference"] = approval_reference
    _sync_labels(record)
    _update_vertex(client, record)
    registry["current_promoted_version_id"] = model_version_id
    registry["promotion_history"].append({**preview, "promoted_at": promoted_at or _utc_now()})
    _write_json(Path(registry_path), registry)
    return {"dry_run": False, "promotion": preview, "record": record}


def rollback_version(
    target_model_version_id: str,
    *,
    rolled_back_by: str,
    reason: str,
    client: VertexRegistryClient,
    registry_path: str | Path,
    lifecycle_root: str | Path,
    execute: bool = False,
    rolled_back_at: str | None = None,
) -> dict[str, Any]:
    if not rolled_back_by or not reason:
        raise VertexRegistryError("Rollback requires rolled_back_by and reason.")
    registry = load_registry(registry_path)
    target = _require_version(registry, target_model_version_id)
    current_id = registry.get("current_promoted_version_id")
    if not current_id or current_id == target_model_version_id:
        raise VertexRegistryError("Rollback target must differ from the current promoted version.")
    current = _require_version(registry, current_id)
    if target["lifecycle_status"] not in {"previously_promoted", "rolled_back"}:
        raise VertexRegistryError("Rollback target must be a previously promoted version.")
    event = {"from_model_version_id": current_id, "to_model_version_id": target_model_version_id, "rolled_back_by": rolled_back_by, "reason": reason, "rolled_back_at": rolled_back_at or _utc_now()}
    if not execute:
        return {"dry_run": True, "rollback": event}
    rollback_production_model(target_model_version_id, rolled_back_by=rolled_back_by, reason=reason, rolled_back_at=rolled_back_at, lifecycle_root=lifecycle_root)
    current["lifecycle_status"] = "rolled_back"
    target["lifecycle_status"] = "promoted"
    _sync_labels(current)
    _sync_labels(target)
    _update_vertex(client, current)
    _update_vertex(client, target)
    registry["current_promoted_version_id"] = target_model_version_id
    registry["rollback_history"].append(event)
    _write_json(Path(registry_path), registry)
    return {"dry_run": False, "rollback": event}


def _build_candidate_record(metadata: dict[str, Any], settings: RegistrySettings, *, created_at: str | None) -> dict[str, Any]:
    required = ("model_version_id", "artifact_uri", "training_run_id", "source_dataset_version", "git_commit_sha", "metrics", "evaluation_report_uri", "leakage_audit_status", "compatibility_metadata", "architecture_backbone")
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        raise VertexRegistryError(f"Missing required registration metadata: {', '.join(missing)}")
    if not GIT_SHA.fullmatch(str(metadata["git_commit_sha"])):
        raise VertexRegistryError("git_commit_sha must be a full 40-character lowercase SHA.")
    if not str(metadata["artifact_uri"]).startswith(settings.artifact_bucket.rstrip("/") + "/"):
        raise VertexRegistryError(f"artifact_uri must be inside {settings.artifact_bucket}.")
    if not str(metadata["evaluation_report_uri"]).startswith("gs://"):
        raise VertexRegistryError("evaluation_report_uri must be a gs:// URI.")
    model_format = str(metadata.get("model_format") or _detect_format(metadata["artifact_uri"])).lower().lstrip(".")
    if model_format not in SUPPORTED_FORMATS:
        raise VertexRegistryError(f"Unsupported model format: {model_format}")
    if not isinstance(metadata["metrics"], dict) or not metadata["metrics"]:
        raise VertexRegistryError("metrics must be a non-empty object.")
    if not isinstance(metadata["compatibility_metadata"], dict) or metadata["compatibility_metadata"].get("status") not in {"passed", "failed"}:
        raise VertexRegistryError("compatibility_metadata.status must be passed or failed.")
    if metadata["leakage_audit_status"] not in {"passed", "failed"}:
        raise VertexRegistryError("leakage_audit_status must be passed or failed.")
    candidate_status = metadata.get("candidate_status", "candidate")
    if candidate_status not in {"candidate", "evaluation_pending", "approved"}:
        raise VertexRegistryError("candidate_status is invalid.")
    record = {
        "schema_version": "vertex-model-registry-record-v1",
        "model_version_id": str(metadata["model_version_id"]),
        "model_name": settings.model_name,
        "artifact_uri": str(metadata["artifact_uri"]),
        "model_format": model_format,
        "training_run_id": str(metadata["training_run_id"]),
        "source_dataset_version": str(metadata["source_dataset_version"]),
        "git_commit_sha": str(metadata["git_commit_sha"]),
        "metrics": metadata["metrics"],
        "evaluation_report_uri": str(metadata["evaluation_report_uri"]),
        "leakage_audit_status": metadata["leakage_audit_status"],
        "compatibility_metadata": metadata["compatibility_metadata"],
        "candidate_status": candidate_status,
        "lifecycle_status": "candidate",
        "architecture_backbone": str(metadata["architecture_backbone"]),
        "created_at": created_at or _utc_now(),
        "approval_identity": None,
        "approval_reference": None,
        "vertex_resource_name": None,
        "labels": {},
    }
    _sync_labels(record)
    return record


def _require_local_lineage(record: dict[str, Any], lifecycle_root: Path) -> None:
    models = _read_collection(lifecycle_root / "model_registry.json", "models")
    runs = _read_collection(lifecycle_root / "training_runs.json", "training_runs")
    model = next((item for item in models if item.get("model_version") == record["model_version_id"]), None)
    run = next((item for item in runs if item.get("training_run_id") == record["training_run_id"]), None)
    if not model:
        raise VertexRegistryError("Candidate must already exist in the local model registry.")
    if not run or run.get("status") != "completed":
        raise VertexRegistryError("Training run must exist locally with completed status.")
    if model.get("training_run_id") != record["training_run_id"] or record["source_dataset_version"] not in model.get("training_dataset_versions", []):
        raise VertexRegistryError("Registration metadata does not match local model lineage.")
    if model.get("status") not in {"evaluation_pending", "approved"}:
        raise VertexRegistryError("Local candidate status is not eligible for Vertex registration.")


def _require_promotion_gates(record: dict[str, Any]) -> None:
    recommendation = record["metrics"].get("recommendation", {})
    evaluation_passed = record["metrics"].get("evaluation_status") == "passed" or recommendation.get("promoteAllowed") is True
    if not evaluation_passed:
        raise VertexRegistryError("Promotion blocked: evaluation has not passed.")
    if record["leakage_audit_status"] != "passed":
        raise VertexRegistryError("Promotion blocked: leakage audit did not pass.")
    if record["compatibility_metadata"].get("status") != "passed":
        raise VertexRegistryError("Promotion blocked: compatibility check did not pass.")


def _sync_labels(record: dict[str, Any]) -> None:
    metrics = record["metrics"]
    labels = {
        "lifecycle_status": _label(record["lifecycle_status"]),
        "candidate_status": _label(record["candidate_status"]),
        "training_data_version": _label(record["source_dataset_version"]),
        "architecture_backbone": _label(record["architecture_backbone"]),
        "created_at": _label(record["created_at"][:10]),
    }
    for key, label_key in (("clean_synthetic_mae", "clean_synth_mae"), ("mobile_realistic_mae", "mobile_real_mae"), ("real_validation_mae", "real_val_mae")):
        if metrics.get(key) is not None:
            labels[label_key] = _label(str(metrics[key]))
    if record.get("approval_identity"):
        labels["approval_identity"] = _label(record["approval_identity"])
    if record.get("approval_reference"):
        labels["approval_reference"] = _label(record["approval_reference"])
    record["labels"] = labels


def _update_vertex(client: VertexRegistryClient, record: dict[str, Any]) -> None:
    resource = record.get("vertex_resource_name")
    if not resource:
        raise VertexRegistryError("Registered version is missing vertex_resource_name.")
    client.update_version_metadata(resource, record["labels"])


def _require_version(registry: dict[str, Any], model_version_id: str) -> dict[str, Any]:
    record = next((item for item in registry["versions"] if item.get("model_version_id") == model_version_id), None)
    if not record:
        raise VertexRegistryError(f"Unknown model version: {model_version_id}")
    return record


def _read_collection(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get(key, [])


def _empty_registry(model_name: str) -> dict[str, Any]:
    return {"schema_version": "vertex-model-registry-v1", "model_name": model_name, "current_promoted_version_id": None, "versions": [], "promotion_history": [], "rollback_history": []}


def _detect_format(uri: str) -> str:
    suffix = Path(urlparse(uri).path).suffix.lower().lstrip(".")
    return suffix or "savedmodel"


def _label(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    return (cleaned or "unknown")[:63]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
