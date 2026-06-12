from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.model_lifecycle import (
    ModelLifecycleError,
    approve_training_queue_entry,
    archive_model,
    build_model_lineage,
    build_registry_dashboard_data,
    build_training_candidate_dashboard,
    complete_training_run,
    create_evaluation_report,
    create_promotion_decision,
    create_training_queue_entry,
    format_candidates,
    format_models,
    format_training_queue,
    format_training_runs,
    generate_dataset_comparison_report,
    generate_training_manifest,
    promote_model_to_production,
    register_model_candidate,
    register_training_run,
    rollback_production_model,
)


def test_training_queue_creation_and_approval(tmp_path: Path) -> None:
    dataset_registry_path = _write_dataset_registry(tmp_path)
    lifecycle_root = tmp_path / "model_lifecycle"

    queue_entry = create_training_queue_entry(
        "rw-v1",
        model_base_version="model-base-v1",
        queued_by="ml-admin",
        notes="Ready for supervised candidate training.",
        queue_id="queue-rw-v1",
        created_at="2026-06-12T12:00:00Z",
        dataset_registry_path=dataset_registry_path,
        lifecycle_root=lifecycle_root,
    )
    approved = approve_training_queue_entry(
        "queue-rw-v1",
        approved_by="lead-reviewer",
        approved_at="2026-06-12T12:05:00Z",
        lifecycle_root=lifecycle_root,
    )

    assert queue_entry["status"] == "pending"
    assert approved["status"] == "approved_for_training"
    assert approved["source_export_id"] == "export-rw-v1"
    assert approved["source_dataset_registry_id"] == "registry-rw-v1"
    assert "queue-rw-v1" in format_training_queue(lifecycle_root)


def test_training_candidate_dashboard_and_dataset_comparison_report(tmp_path: Path) -> None:
    dataset_registry_path = _write_dataset_registry(tmp_path)
    dashboard_path = tmp_path / "reports" / "training_candidate_dashboard.json"

    dashboard = build_training_candidate_dashboard(
        dataset_registry_path=dataset_registry_path,
        output_path=dashboard_path,
    )
    comparison = generate_dataset_comparison_report(
        [
            {
                "dataset_version": "synthetic-v1",
                "dataset_type": "synthetic",
                "participant_count": 100,
                "measurement_count": 700,
                "quality_score": 91,
                "demographic_coverage": {"source": "synthetic_balanced"},
                "lineage": {"generator": "blend-v1"},
            },
            dashboard["candidates"][0] | {"dataset_type": "real_world"},
            {
                "dataset_version": "mixed-v1",
                "dataset_type": "mixed",
                "participant_count": 110,
                "measurement_count": 770,
                "quality_score": 90,
                "demographic_coverage": {"blend": "synthetic_plus_field"},
                "lineage": {"components": ["synthetic-v1", "rw-v1"]},
            },
        ],
        output_path=tmp_path / "reports" / "dataset_comparison_report.json",
        generated_at="2026-06-12T12:10:00Z",
    )

    assert dashboard_path.exists()
    assert dashboard["candidates"][0]["dataset_version"] == "rw-v1"
    assert dashboard["candidates"][0]["approval_state"] == "approved_for_training"
    assert comparison["summary"]["dataset_types"] == ["mixed", "real_world", "synthetic"]
    assert comparison["summary"]["total_participants"] == 211
    assert (tmp_path / "reports" / "dataset_comparison_report.json").exists()


def test_manifest_generation_requires_approved_queue(tmp_path: Path) -> None:
    dataset_registry_path = _write_dataset_registry(tmp_path)
    lifecycle_root = tmp_path / "model_lifecycle"
    create_training_queue_entry(
        "rw-v1",
        model_base_version="model-base-v1",
        queued_by="ml-admin",
        queue_id="queue-rw-v1",
        dataset_registry_path=dataset_registry_path,
        lifecycle_root=lifecycle_root,
    )

    with pytest.raises(ModelLifecycleError, match="approved_for_training"):
        generate_training_manifest(
            "queue-rw-v1",
            training_parameters={"epochs": 0},
            dataset_registry_path=dataset_registry_path,
            lifecycle_root=lifecycle_root,
        )

    approve_training_queue_entry("queue-rw-v1", approved_by="lead-reviewer", lifecycle_root=lifecycle_root)
    manifest = generate_training_manifest(
        "queue-rw-v1",
        training_parameters={"epochs": 25, "learning_rate": 0.001},
        generated_timestamp="2026-06-12T12:15:00Z",
        dataset_registry_path=dataset_registry_path,
        lifecycle_root=lifecycle_root,
    )

    assert manifest["dataset_version"] == "rw-v1"
    assert manifest["source_registry_entry"]["source_export_id"] == "export-rw-v1"
    assert manifest["training_execution"]["auto_train"] is False
    assert (lifecycle_root / "training_manifest.json").exists()


def test_training_run_registration_model_registry_and_evaluation_gate(tmp_path: Path) -> None:
    dataset_registry_path = _write_dataset_registry(tmp_path)
    lifecycle_root = tmp_path / "model_lifecycle"
    manifest = _approved_manifest(dataset_registry_path, lifecycle_root)

    run = register_training_run(
        manifest,
        training_run_id="run-rw-v1",
        model_version="model-rw-v1",
        status="pending",
        lifecycle_root=lifecycle_root,
    )
    model = complete_training_run(
        "run-rw-v1",
        metrics={"mae": 2.4, "rmse": 3.1},
        model_type="measurement_regressor",
        completed_at="2026-06-12T12:30:00Z",
        lifecycle_root=lifecycle_root,
    )

    assert run["status"] == "pending"
    assert model["status"] == "evaluation_pending"
    assert model["training_dataset_versions"] == ["rw-v1"]
    assert "run-rw-v1" in format_training_runs(lifecycle_root)
    assert "model-rw-v1" in format_models(lifecycle_root)

    report = create_evaluation_report(
        "model-rw-v1",
        metrics={"mae": 2.4, "rmse": 3.1, "confidence_metrics": {"p90_interval": 4.8}},
        measurement_accuracy={"waist": {"mae": 2.1}, "hips": {"mae": 2.6}},
        benchmark_comparison={"baseline_model": "model-base-v1", "mae_delta": -0.4},
        regression_analysis={"regressions": []},
        previous_production_comparison={"production_model": None, "comparison": "not_available"},
        generated_at="2026-06-12T12:35:00Z",
        lifecycle_root=lifecycle_root,
        output_path=tmp_path / "reports" / "evaluation_report.json",
    )
    production = json.loads((lifecycle_root / "production_models.json").read_text(encoding="utf-8")) if (
        lifecycle_root / "production_models.json"
    ).exists() else {"current_production_model": None}

    assert report["promotion_gate"]["auto_promoted"] is False
    assert report["promotion_gate"]["requires_explicit_approval"] is True
    assert production["current_production_model"] is None


def test_promotion_requires_approval_and_rollback_preserves_history(tmp_path: Path) -> None:
    lifecycle_root = tmp_path / "model_lifecycle"
    _register_model(lifecycle_root, "model-old", status="evaluation_pending")
    old_decision = create_promotion_decision(
        "model-old",
        status="approved_for_production",
        decided_by="lead-reviewer",
        decision_id="decision-old",
        lifecycle_root=lifecycle_root,
    )
    promote_model_to_production(
        "model-old",
        decision_id=old_decision["decision_id"],
        promoted_by="release-manager",
        lifecycle_root=lifecycle_root,
    )

    _register_model(lifecycle_root, "model-new", status="evaluation_pending")
    candidate_decision = create_promotion_decision(
        "model-new",
        status="candidate",
        decision_id="decision-new-candidate",
        lifecycle_root=lifecycle_root,
    )
    with pytest.raises(ModelLifecycleError, match="approved promotion decision"):
        promote_model_to_production(
            "model-new",
            decision_id=candidate_decision["decision_id"],
            promoted_by="release-manager",
            lifecycle_root=lifecycle_root,
        )

    approved_decision = create_promotion_decision(
        "model-new",
        status="approved_for_production",
        decided_by="lead-reviewer",
        decision_id="decision-new-approved",
        lifecycle_root=lifecycle_root,
    )
    production = promote_model_to_production(
        "model-new",
        decision_id=approved_decision["decision_id"],
        promoted_by="release-manager",
        lifecycle_root=lifecycle_root,
    )
    rolled_back = rollback_production_model(
        "model-old",
        rolled_back_by="release-manager",
        reason="Regression detected after approval.",
        lifecycle_root=lifecycle_root,
    )

    assert production["current_production_model"] == "model-new"
    assert "model-old" in production["previous_production_models"]
    assert rolled_back["current_production_model"] == "model-old"
    assert rolled_back["rollback_history"][0]["from_model_version"] == "model-new"


def test_lineage_tracking_dashboard_and_archived_models(tmp_path: Path) -> None:
    dataset_registry_path = _write_dataset_registry(tmp_path)
    lifecycle_root = tmp_path / "model_lifecycle"
    manifest = _approved_manifest(dataset_registry_path, lifecycle_root)
    register_training_run(
        manifest,
        training_run_id="run-rw-v1",
        model_version="model-rw-v1",
        lifecycle_root=lifecycle_root,
    )
    complete_training_run(
        "run-rw-v1",
        metrics={"mae": 2.4, "rmse": 3.1},
        model_type="measurement_regressor",
        lifecycle_root=lifecycle_root,
    )
    create_evaluation_report(
        "model-rw-v1",
        metrics={"mae": 2.4, "rmse": 3.1},
        measurement_accuracy={"waist": {"mae": 2.1}},
        benchmark_comparison={"baseline": "model-base-v1"},
        regression_analysis={"regressions": []},
        previous_production_comparison={"production_model": None},
        lifecycle_root=lifecycle_root,
        output_path=tmp_path / "reports" / "evaluation_report.json",
    )
    _register_model(lifecycle_root, "model-archived", status="development")
    archive_model(
        "model-archived",
        archived_by="ml-admin",
        reason="Superseded development candidate.",
        lifecycle_root=lifecycle_root,
    )

    lineage = build_model_lineage("model-rw-v1", lifecycle_root=lifecycle_root)
    dashboard = build_registry_dashboard_data(lifecycle_root=lifecycle_root)

    assert lineage["candidate_model"]["model_version"] == "model-rw-v1"
    assert lineage["training_run"]["training_run_id"] == "run-rw-v1"
    assert lineage["dataset_versions"] == ["rw-v1"]
    assert lineage["source_exports"] == ["export-rw-v1"]
    assert dashboard["evaluation_candidates"][0]["model_version"] == "model-rw-v1"
    assert dashboard["archived_models"][0]["model_version"] == "model-archived"
    assert "model-rw-v1" in format_candidates(lifecycle_root)


def _approved_manifest(dataset_registry_path: Path, lifecycle_root: Path) -> dict[str, object]:
    create_training_queue_entry(
        "rw-v1",
        model_base_version="model-base-v1",
        queued_by="ml-admin",
        queue_id="queue-rw-v1",
        dataset_registry_path=dataset_registry_path,
        lifecycle_root=lifecycle_root,
    )
    approve_training_queue_entry("queue-rw-v1", approved_by="lead-reviewer", lifecycle_root=lifecycle_root)
    return generate_training_manifest(
        "queue-rw-v1",
        training_parameters={"epochs": 25, "learning_rate": 0.001},
        dataset_registry_path=dataset_registry_path,
        lifecycle_root=lifecycle_root,
    )


def _register_model(lifecycle_root: Path, model_version: str, *, status: str) -> dict[str, object]:
    return register_model_candidate(
        model_version=model_version,
        model_type="measurement_regressor",
        parent_model_version="model-base-v1",
        training_run_id=f"run-{model_version}",
        training_dataset_versions=["rw-v1"],
        status=status,
        lifecycle_root=lifecycle_root,
    )


def _write_dataset_registry(tmp_path: Path) -> Path:
    path = tmp_path / "dataset_registry" / "datasets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "registry_id": "registry-rw-v1",
                        "dataset_version": "rw-v1",
                        "source_system": "CUSTOM-FASHION-MARKETPLACE",
                        "source_export_id": "export-rw-v1",
                        "export_timestamp": "2026-06-12T11:00:00Z",
                        "import_timestamp": "2026-06-12T11:05:00Z",
                        "schema_version": "real-world-dataset-export-v1",
                        "image_count": 3,
                        "measurement_count": 7,
                        "participant_count": 1,
                        "record_count": 1,
                        "quality_score": 98.0,
                        "quality_summary": {"quality_score": 98.0},
                        "validation_status": "validated",
                        "training_status": "not_started",
                        "status": "approved_for_training",
                        "collector_summary": {"collectors": 1, "sites": ["field-beta"]},
                        "lineage": {
                            "source_export_id": "export-rw-v1",
                            "source_app_version": "fashionapp-field-data-beta",
                            "source_dataset_version": "rw-v1",
                        },
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
