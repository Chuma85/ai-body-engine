from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_REGISTRY_PATH = Path("dataset_registry/datasets.json")
DEFAULT_LIFECYCLE_ROOT = Path("model_lifecycle")
DEFAULT_REPORTS_ROOT = Path("reports")

QUEUE_STATUSES = ("pending", "approved_for_training", "training", "completed", "failed", "cancelled", "archived")
TRAINING_RUN_STATUSES = ("pending", "running", "completed", "failed", "cancelled")
MODEL_STATUSES = ("development", "evaluation_pending", "approved", "production", "archived")
PROMOTION_STATUSES = ("candidate", "approved_for_production", "rejected", "archived")
READY_DATASET_STATUSES = {"approved_for_training", "ready_for_training"}
READY_VALIDATION_STATUSES = {"validated", "ready_for_training"}


class ModelLifecycleError(ValueError):
    pass


@dataclass(frozen=True)
class LifecyclePaths:
    root: Path = DEFAULT_LIFECYCLE_ROOT

    @property
    def training_queue(self) -> Path:
        return self.root / "training_queue.json"

    @property
    def training_runs(self) -> Path:
        return self.root / "training_runs.json"

    @property
    def model_registry(self) -> Path:
        return self.root / "model_registry.json"

    @property
    def promotion_registry(self) -> Path:
        return self.root / "promotion_decisions.json"

    @property
    def production_registry(self) -> Path:
        return self.root / "production_models.json"

    @property
    def training_manifest(self) -> Path:
        return self.root / "training_manifest.json"

    @property
    def dashboard_data(self) -> Path:
        return self.root / "registry_dashboard.json"


def create_training_queue_entry(
    dataset_version: str,
    *,
    model_base_version: str,
    queued_by: str,
    notes: str = "",
    queue_id: str | None = None,
    created_at: str | None = None,
    dataset_registry_path: str | Path = DEFAULT_DATASET_REGISTRY_PATH,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    registry_entry = require_ready_dataset(dataset_version, dataset_registry_path)
    paths = LifecyclePaths(Path(lifecycle_root))
    queue = _load_collection(paths.training_queue, "queue")
    created = created_at or _utc_now()
    entry = {
        "queue_id": queue_id or _make_id("queue", dataset_version, created),
        "dataset_version": dataset_version,
        "source_export_id": registry_entry.get("source_export_id"),
        "source_dataset_registry_id": registry_entry.get("registry_id")
        or registry_entry.get("source_export_id")
        or dataset_version,
        "model_base_version": model_base_version,
        "created_at": created,
        "queued_by": queued_by,
        "notes": notes,
        "status": "pending",
        "audit": [_audit_event("queued", queued_by, created, notes)],
    }
    _upsert_by_id(queue, "queue", "queue_id", entry)
    _write_json(paths.training_queue, queue)
    return entry


def approve_training_queue_entry(
    queue_id: str,
    *,
    approved_by: str,
    notes: str = "",
    approved_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    queue = _load_collection(paths.training_queue, "queue")
    entry = _require_by_id(queue["queue"], "queue_id", queue_id)
    _require_status(entry["status"], {"pending"}, "training queue approval")
    approved = approved_at or _utc_now()
    entry["status"] = "approved_for_training"
    entry["approved_by"] = approved_by
    entry["approved_at"] = approved
    entry.setdefault("audit", []).append(_audit_event("approved_for_training", approved_by, approved, notes))
    _write_json(paths.training_queue, queue)
    return entry


def update_training_queue_status(
    queue_id: str,
    status: str,
    *,
    changed_by: str,
    notes: str = "",
    changed_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    if status not in QUEUE_STATUSES:
        raise ModelLifecycleError(f"Unsupported training queue status: {status}.")
    paths = LifecyclePaths(Path(lifecycle_root))
    queue = _load_collection(paths.training_queue, "queue")
    entry = _require_by_id(queue["queue"], "queue_id", queue_id)
    changed = changed_at or _utc_now()
    entry["status"] = status
    entry.setdefault("audit", []).append(_audit_event(status, changed_by, changed, notes))
    _write_json(paths.training_queue, queue)
    return entry


def build_training_candidate_dashboard(
    *,
    dataset_registry_path: str | Path = DEFAULT_DATASET_REGISTRY_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    registry = _load_collection(Path(dataset_registry_path), "datasets")
    candidates = []
    for entry in registry["datasets"]:
        quality_summary = entry.get("quality_summary") if isinstance(entry.get("quality_summary"), dict) else {}
        candidates.append(
            {
                "dataset_version": entry.get("dataset_version"),
                "participant_count": int(entry.get("participant_count") or entry.get("record_count") or 0),
                "image_count": int(entry.get("image_count") or 0),
                "measurement_count": int(entry.get("measurement_count") or 0),
                "quality_score": float(entry.get("quality_score") or quality_summary.get("quality_score") or 0),
                "validation_status": entry.get("validation_status"),
                "collector_summary": entry.get("collector_summary")
                or {
                    "source_system": entry.get("source_system"),
                    "source_export_id": entry.get("source_export_id"),
                },
                "approval_state": _dataset_approval_state(entry),
            }
        )
    dashboard = {"schema_version": "training-candidate-dashboard-v1", "generated_at": _utc_now(), "candidates": candidates}
    if output_path:
        _write_json(Path(output_path), dashboard)
    return dashboard


def generate_dataset_comparison_report(
    datasets: list[dict[str, Any]],
    *,
    output_path: str | Path = DEFAULT_REPORTS_ROOT / "dataset_comparison_report.json",
    generated_at: str | None = None,
) -> dict[str, Any]:
    compared = []
    for dataset in datasets:
        compared.append(
            {
                "dataset_version": dataset.get("dataset_version"),
                "dataset_type": dataset.get("dataset_type", "real_world"),
                "participant_count": int(dataset.get("participant_count") or dataset.get("record_count") or 0),
                "measurement_coverage": dataset.get("measurement_coverage")
                or _coverage_from_counts(dataset.get("measurement_count"), dataset.get("participant_count")),
                "image_quality": float(dataset.get("quality_score") or dataset.get("image_quality") or 0),
                "demographic_coverage": dataset.get("demographic_coverage", {}),
                "version_lineage": dataset.get("lineage") or dataset.get("version_lineage") or {},
            }
        )
    report = {
        "schema_version": "dataset-comparison-report-v1",
        "generated_at": generated_at or _utc_now(),
        "datasets": compared,
        "summary": {
            "dataset_count": len(compared),
            "dataset_types": sorted({str(item["dataset_type"]) for item in compared}),
            "total_participants": sum(int(item["participant_count"]) for item in compared),
        },
    }
    _write_json(Path(output_path), report)
    return report


def generate_training_manifest(
    queue_id: str,
    *,
    training_parameters: dict[str, Any],
    schema_version: str = "training-manifest-v1",
    generated_timestamp: str | None = None,
    dataset_registry_path: str | Path = DEFAULT_DATASET_REGISTRY_PATH,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    queue = _load_collection(paths.training_queue, "queue")
    queue_entry = _require_by_id(queue["queue"], "queue_id", queue_id)
    _require_status(queue_entry["status"], {"approved_for_training"}, "training manifest generation")
    registry_entry = require_ready_dataset(queue_entry["dataset_version"], dataset_registry_path)
    generated = generated_timestamp or _utc_now()
    manifest = {
        "schema_version": schema_version,
        "generated_timestamp": generated,
        "queue_id": queue_id,
        "dataset_version": queue_entry["dataset_version"],
        "source_registry_entry": registry_entry,
        "model_base_version": queue_entry["model_base_version"],
        "lineage": {
            "source_export_id": queue_entry["source_export_id"],
            "source_dataset_registry_id": queue_entry["source_dataset_registry_id"],
            "source_dataset_version": registry_entry.get("lineage", {}).get("source_dataset_version")
            if isinstance(registry_entry.get("lineage"), dict)
            else registry_entry.get("dataset_version"),
            "queue_created_at": queue_entry["created_at"],
            "manifest_generated_at": generated,
        },
        "training_parameters": training_parameters,
        "training_execution": {
            "auto_train": False,
            "requires_explicit_runner": True,
        },
    }
    _write_json(Path(output_path) if output_path else paths.training_manifest, manifest)
    return manifest


def register_training_run(
    training_manifest: dict[str, Any],
    *,
    training_run_id: str,
    model_version: str,
    status: str = "pending",
    start_time: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    if status not in TRAINING_RUN_STATUSES:
        raise ModelLifecycleError(f"Unsupported training run status: {status}.")
    paths = LifecyclePaths(Path(lifecycle_root))
    runs = _load_collection(paths.training_runs, "training_runs")
    entry = {
        "training_run_id": training_run_id,
        "dataset_version": training_manifest["dataset_version"],
        "model_version": model_version,
        "model_base_version": training_manifest["model_base_version"],
        "start_time": start_time,
        "end_time": None,
        "duration": None,
        "status": status,
        "metrics": {},
        "training_manifest": training_manifest,
    }
    _upsert_by_id(runs, "training_runs", "training_run_id", entry)
    _write_json(paths.training_runs, runs)
    return entry


def complete_training_run(
    training_run_id: str,
    *,
    metrics: dict[str, Any],
    model_type: str,
    completed_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    runs = _load_collection(paths.training_runs, "training_runs")
    run = _require_by_id(runs["training_runs"], "training_run_id", training_run_id)
    if run["status"] not in {"pending", "running"}:
        raise ModelLifecycleError(f"Training run {training_run_id} cannot be completed from status {run['status']}.")
    completed = completed_at or _utc_now()
    if not run.get("start_time"):
        run["start_time"] = completed
    run["end_time"] = completed
    run["duration"] = _duration_label(run["start_time"], completed)
    run["status"] = "completed"
    run["metrics"] = metrics
    _write_json(paths.training_runs, runs)

    models = _load_collection(paths.model_registry, "models")
    manifest = run["training_manifest"]
    model = {
        "model_version": run["model_version"],
        "model_type": model_type,
        "parent_model_version": run["model_base_version"],
        "training_run_id": training_run_id,
        "training_dataset_versions": [run["dataset_version"]],
        "created_at": completed,
        "status": "evaluation_pending",
        "lineage": {
            "parent_model": run["model_base_version"],
            "training_datasets": [run["dataset_version"]],
            "training_manifests": [manifest],
            "training_runs": [training_run_id],
            "evaluation_reports": [],
        },
    }
    _upsert_by_id(models, "models", "model_version", model)
    _write_json(paths.model_registry, models)
    return model


def register_model_candidate(
    *,
    model_version: str,
    model_type: str,
    parent_model_version: str,
    training_run_id: str,
    training_dataset_versions: list[str],
    status: str = "development",
    created_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    if status not in MODEL_STATUSES:
        raise ModelLifecycleError(f"Unsupported model status: {status}.")
    paths = LifecyclePaths(Path(lifecycle_root))
    models = _load_collection(paths.model_registry, "models")
    entry = {
        "model_version": model_version,
        "model_type": model_type,
        "parent_model_version": parent_model_version,
        "training_run_id": training_run_id,
        "training_dataset_versions": training_dataset_versions,
        "created_at": created_at or _utc_now(),
        "status": status,
        "lineage": {
            "parent_model": parent_model_version,
            "training_datasets": training_dataset_versions,
            "training_manifests": [],
            "training_runs": [training_run_id],
            "evaluation_reports": [],
        },
    }
    _upsert_by_id(models, "models", "model_version", entry)
    _write_json(paths.model_registry, models)
    return entry


def create_evaluation_report(
    model_version: str,
    *,
    metrics: dict[str, Any],
    measurement_accuracy: dict[str, Any],
    benchmark_comparison: dict[str, Any],
    regression_analysis: dict[str, Any],
    previous_production_comparison: dict[str, Any],
    confidence_metrics: dict[str, Any] | None = None,
    generated_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
    output_path: str | Path = DEFAULT_REPORTS_ROOT / "evaluation_report.json",
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    models = _load_collection(paths.model_registry, "models")
    model = _require_by_id(models["models"], "model_version", model_version)
    _require_status(model["status"], {"evaluation_pending", "development"}, "evaluation reporting")
    report = {
        "schema_version": "model-evaluation-report-v1",
        "generated_at": generated_at or _utc_now(),
        "model_version": model_version,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "confidence_metrics": confidence_metrics or metrics.get("confidence_metrics", {}),
        "measurement_specific_accuracy": measurement_accuracy,
        "benchmark_comparison": benchmark_comparison,
        "regression_analysis": regression_analysis,
        "previous_production_comparison": previous_production_comparison,
        "promotion_gate": {
            "status": "evaluation_pending",
            "auto_promoted": False,
            "requires_explicit_approval": True,
        },
    }
    report_path = Path(output_path)
    _write_json(report_path, report)
    model.setdefault("lineage", {}).setdefault("evaluation_reports", []).append(str(report_path))
    model["status"] = "evaluation_pending"
    _write_json(paths.model_registry, models)
    return report


def create_promotion_decision(
    model_version: str,
    *,
    status: str = "candidate",
    decided_by: str | None = None,
    notes: str = "",
    decision_id: str | None = None,
    decided_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    if status not in PROMOTION_STATUSES:
        raise ModelLifecycleError(f"Unsupported promotion decision status: {status}.")
    if status == "approved_for_production" and not decided_by:
        raise ModelLifecycleError("Promotion approval requires decided_by.")
    paths = LifecyclePaths(Path(lifecycle_root))
    _require_model(model_version, paths)
    decisions = _load_collection(paths.promotion_registry, "promotion_decisions")
    decided = decided_at or _utc_now()
    decision = {
        "decision_id": decision_id or _make_id("promotion", model_version, decided),
        "model_version": model_version,
        "status": status,
        "decided_by": decided_by,
        "decided_at": decided,
        "notes": notes,
        "auto_promoted": False,
    }
    _upsert_by_id(decisions, "promotion_decisions", "decision_id", decision)
    _write_json(paths.promotion_registry, decisions)
    if status == "approved_for_production":
        _set_model_status(model_version, "approved", paths)
    return decision


def promote_model_to_production(
    model_version: str,
    *,
    decision_id: str,
    promoted_by: str,
    notes: str = "",
    promoted_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    decisions = _load_collection(paths.promotion_registry, "promotion_decisions")
    decision = _require_by_id(decisions["promotion_decisions"], "decision_id", decision_id)
    if decision["model_version"] != model_version or decision["status"] != "approved_for_production":
        raise ModelLifecycleError("Production promotion requires an approved promotion decision for the same model.")
    promoted = promoted_at or _utc_now()
    production = _load_production(paths.production_registry)
    previous = production.get("current_production_model")
    if previous and previous != model_version:
        previous_models = production.setdefault("previous_production_models", [])
        if previous not in previous_models:
            previous_models.append(previous)
        _set_model_status(previous, "approved", paths)
    production["current_production_model"] = model_version
    production.setdefault("promotion_history", []).append(
        {
            "model_version": model_version,
            "previous_model_version": previous,
            "decision_id": decision_id,
            "promoted_by": promoted_by,
            "promoted_at": promoted,
            "notes": notes,
        }
    )
    production.setdefault("rollback_history", [])
    _write_json(paths.production_registry, production)
    _set_model_status(model_version, "production", paths)
    return production


def rollback_production_model(
    target_model_version: str,
    *,
    rolled_back_by: str,
    reason: str,
    rolled_back_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    production = _load_production(paths.production_registry)
    current = production.get("current_production_model")
    if target_model_version not in production.get("previous_production_models", []):
        raise ModelLifecycleError("Rollback target must be a previous production model.")
    rolled_back = rolled_back_at or _utc_now()
    previous_models = [item for item in production.get("previous_production_models", []) if item != target_model_version]
    if current and current not in previous_models:
        previous_models.append(current)
        _set_model_status(current, "approved", paths)
    production["current_production_model"] = target_model_version
    production["previous_production_models"] = previous_models
    production.setdefault("rollback_history", []).append(
        {
            "from_model_version": current,
            "to_model_version": target_model_version,
            "rolled_back_by": rolled_back_by,
            "rolled_back_at": rolled_back,
            "reason": reason,
        }
    )
    _write_json(paths.production_registry, production)
    _set_model_status(target_model_version, "production", paths)
    return production


def build_model_lineage(
    model_version: str,
    *,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    model = _require_model(model_version, paths)
    runs = _load_collection(paths.training_runs, "training_runs")["training_runs"]
    run = next((item for item in runs if item.get("training_run_id") == model.get("training_run_id")), None)
    production = _load_production(paths.production_registry)
    return {
        "model_version": model_version,
        "production_model": production.get("current_production_model") if production.get("current_production_model") == model_version else None,
        "candidate_model": model,
        "training_run": run,
        "dataset_versions": model.get("training_dataset_versions", []),
        "source_exports": [
            item.get("source_export_id")
            for item in _collect_manifest_sources(model)
            if item.get("source_export_id")
        ],
        "evaluation_reports": model.get("lineage", {}).get("evaluation_reports", []),
    }


def build_registry_dashboard_data(
    *,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    models = _load_collection(paths.model_registry, "models")["models"]
    runs = _load_collection(paths.training_runs, "training_runs")["training_runs"]
    queue = _load_collection(paths.training_queue, "queue")["queue"]
    production = _load_production(paths.production_registry)
    dashboard = {
        "schema_version": "model-lifecycle-dashboard-v1",
        "generated_at": _utc_now(),
        "production_model": production.get("current_production_model"),
        "evaluation_candidates": [model for model in models if model.get("status") == "evaluation_pending"],
        "training_queue": [item for item in queue if item.get("status") not in {"archived"}],
        "recent_training_runs": runs[-10:],
        "archived_models": [model for model in models if model.get("status") == "archived"],
    }
    _write_json(Path(output_path) if output_path else paths.dashboard_data, dashboard)
    return dashboard


def archive_model(
    model_version: str,
    *,
    archived_by: str,
    reason: str,
    archived_at: str | None = None,
    lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT,
) -> dict[str, Any]:
    paths = LifecyclePaths(Path(lifecycle_root))
    models = _load_collection(paths.model_registry, "models")
    model = _require_by_id(models["models"], "model_version", model_version)
    if model.get("status") == "production":
        raise ModelLifecycleError("Production model cannot be archived while active.")
    model["status"] = "archived"
    model.setdefault("audit", []).append(_audit_event("archived", archived_by, archived_at or _utc_now(), reason))
    _write_json(paths.model_registry, models)
    return model


def format_training_queue(lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT) -> str:
    rows = _load_collection(LifecyclePaths(Path(lifecycle_root)).training_queue, "queue")["queue"]
    return _format_rows(rows, ("queue_id", "dataset_version", "status", "model_base_version"))


def format_training_runs(lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT) -> str:
    rows = _load_collection(LifecyclePaths(Path(lifecycle_root)).training_runs, "training_runs")["training_runs"]
    return _format_rows(rows, ("training_run_id", "dataset_version", "model_version", "status"))


def format_models(lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT) -> str:
    rows = _load_collection(LifecyclePaths(Path(lifecycle_root)).model_registry, "models")["models"]
    return _format_rows(rows, ("model_version", "model_type", "status", "training_run_id"))


def format_candidates(lifecycle_root: str | Path = DEFAULT_LIFECYCLE_ROOT) -> str:
    rows = [
        model
        for model in _load_collection(LifecyclePaths(Path(lifecycle_root)).model_registry, "models")["models"]
        if model.get("status") in {"development", "evaluation_pending", "approved"}
    ]
    return _format_rows(rows, ("model_version", "model_type", "status", "training_run_id"))


def require_ready_dataset(dataset_version: str, dataset_registry_path: str | Path) -> dict[str, Any]:
    registry = _load_collection(Path(dataset_registry_path), "datasets")
    entry = next((item for item in registry["datasets"] if item.get("dataset_version") == dataset_version), None)
    if entry is None:
        raise ModelLifecycleError(f"Dataset {dataset_version} is not registered.")
    if _dataset_approval_state(entry) != "approved_for_training":
        raise ModelLifecycleError(f"Dataset {dataset_version} is not approved for training.")
    return entry


def _dataset_approval_state(entry: dict[str, Any]) -> str:
    if entry.get("status") in READY_DATASET_STATUSES or entry.get("validation_status") in READY_VALIDATION_STATUSES:
        return "approved_for_training"
    return "not_ready"


def _coverage_from_counts(measurement_count: Any, participant_count: Any) -> dict[str, Any]:
    participants = int(participant_count or 0)
    measurements = int(measurement_count or 0)
    return {
        "measurement_count": measurements,
        "participant_count": participants,
        "measurements_per_participant": round(measurements / participants, 2) if participants else 0,
    }


def _collect_manifest_sources(model: dict[str, Any]) -> list[dict[str, Any]]:
    manifests = model.get("lineage", {}).get("training_manifests", [])
    sources = []
    for manifest in manifests:
        if isinstance(manifest, dict):
            source = manifest.get("source_registry_entry")
            if isinstance(source, dict):
                sources.append(source)
    return sources


def _require_model(model_version: str, paths: LifecyclePaths) -> dict[str, Any]:
    models = _load_collection(paths.model_registry, "models")
    return _require_by_id(models["models"], "model_version", model_version)


def _set_model_status(model_version: str, status: str, paths: LifecyclePaths) -> None:
    models = _load_collection(paths.model_registry, "models")
    model = _require_by_id(models["models"], "model_version", model_version)
    model["status"] = status
    _write_json(paths.model_registry, models)


def _load_production(path: Path) -> dict[str, Any]:
    default = {
        "current_production_model": None,
        "previous_production_models": [],
        "promotion_history": [],
        "rollback_history": [],
    }
    if not path.exists():
        return default
    payload = _read_json(path)
    for key, value in default.items():
        payload.setdefault(key, value)
    return payload


def _load_collection(path: Path, key: str) -> dict[str, Any]:
    if not path.exists():
        return {key: []}
    payload = _read_json(path)
    value = payload.get(key)
    if not isinstance(value, list):
        return {key: []}
    return {key: value}


def _upsert_by_id(collection: dict[str, list[dict[str, Any]]], key: str, id_key: str, entry: dict[str, Any]) -> None:
    rows = [item for item in collection[key] if item.get(id_key) != entry[id_key]]
    rows.append(entry)
    collection[key] = rows


def _require_by_id(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    entry = next((item for item in rows if item.get(key) == value), None)
    if entry is None:
        raise ModelLifecycleError(f"No entry found for {key}={value}.")
    return entry


def _require_status(status: str, allowed: set[str], action: str) -> None:
    if status not in allowed:
        raise ModelLifecycleError(f"{action} requires status in {sorted(allowed)}, got {status}.")


def _audit_event(action: str, actor: str | None, timestamp: str, notes: str) -> dict[str, Any]:
    return {"action": action, "actor": actor, "timestamp": timestamp, "notes": notes}


def _duration_label(start_time: str | None, end_time: str) -> str:
    if not start_time:
        return "unknown"
    return f"{start_time}..{end_time}"


def _format_rows(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "No records."
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for row in rows:
        lines.append("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def _make_id(prefix: str, value: str, timestamp: str) -> str:
    return f"{prefix}_{_safe(value)}_{_safe(timestamp)}"


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ModelLifecycleError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
