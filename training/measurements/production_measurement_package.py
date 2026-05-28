from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.customer_measurement_confirmation import CONFIRMABLE_TARGETS
from training.measurements.maker_measurement_review import (
    BodyMeasurementSource,
    ProductionStatus,
    apply_demo_customer_confirmations,
    apply_demo_maker_updates,
    build_maker_review_payload,
    export_sample_customer_confirmation,
    lock_for_production,
)
from training.measurements.measurement_audit_trail import (
    ActorRole,
    AuditEventType,
    create_audit_event,
    record_lock_event,
    record_revision_request,
)
from training.measurements.measurement_result_schema import write_json

DEFAULT_OUTPUT_DIR = "artifacts/phase_4n_production_measurement_package"
SAMPLE_PACKAGE_JSON = "sample_production_measurement_package.json"
SAMPLE_READINESS_JSON = "sample_readiness_summary.json"
PACKAGE_SUMMARY_MD = "production_package_summary.md"
PACKAGE_SCHEMA_VERSION = "phase_4n_production_measurement_package_v1"


class ProductionPackageError(ValueError):
    """Raised when an order production measurement package is invalid."""


class PackageStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_CUSTOMER_CONFIRMATION = "awaiting_customer_confirmation"
    AWAITING_MAKER_REVIEW = "awaiting_maker_review"
    READY_FOR_PRODUCTION = "ready_for_production"
    LOCKED_FOR_PRODUCTION = "locked_for_production"
    REVISION_REQUESTED = "revision_requested"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProductionPackageTarget:
    target: str
    ai_estimate_cm: float | None
    customer_confirmed_cm: float | None
    customer_manual_cm: float | None
    maker_verified_body_cm: float | None
    selected_body_measurement_cm: float | None
    selected_body_measurement_source: str
    maker_ease_allowance_cm: float | None
    final_garment_cm: float | None
    confidence_tier: str | None
    estimated_error_cm: float | None
    interval_low_cm: float | None
    interval_high_cm: float | None
    product_action: str | None
    quality_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_production_package(
    maker_review_payload: dict[str, Any],
    audit_events: list[dict[str, Any]] | None = None,
    package_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or utc_now()
    targets = [target_from_review(review) for review in maker_review_payload["reviews"]]
    audit_event_ids = [event["event_id"] for event in (audit_events or [])]
    build_event = create_audit_event(
        event_id=f"package_built__{package_id or maker_review_payload['review_id']}",
        event_type=AuditEventType.ADMIN_REVIEW_NOTE_ADDED,
        actor_id="production_package_system",
        actor_role=ActorRole.SYSTEM,
        field_key="production_measurement_package",
        old_value=None,
        new_value=package_id or f"production_package_{maker_review_payload['review_id']}",
        measurement_snapshot_id=maker_review_payload["measurement_snapshot_id"],
        maker_review_id=maker_review_payload["review_id"],
        order_id=maker_review_payload.get("order_id"),
        created_at=timestamp,
        notes="Production measurement package built.",
    )
    audit_event_ids.append(build_event["event_id"])
    package = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": package_id or f"production_package_{maker_review_payload['review_id']}",
        "order_id": maker_review_payload.get("order_id"),
        "customer_id": maker_review_payload.get("customer_id") or maker_review_payload.get("user_id"),
        "maker_id": maker_review_payload["maker_id"],
        "measurement_snapshot_id": maker_review_payload["measurement_snapshot_id"],
        "customer_confirmation_id": maker_review_payload.get("customer_confirmation_id"),
        "maker_review_id": maker_review_payload["review_id"],
        "package_status": package_status_from_maker_review(maker_review_payload),
        "targets": [target.to_payload() for target in targets],
        "created_at": timestamp,
        "updated_at": None,
        "locked_at": maker_review_payload.get("locked_at"),
        "locked_by_maker_id": maker_review_payload.get("locked_by_maker_id"),
        "audit_event_ids": audit_event_ids,
        "warnings": [],
        "readiness_summary": {},
        "synthetic_calibrated_only": True,
        "real_world_validated": False,
    }
    summary = summarize_package_readiness(package)
    package["warnings"] = summary["warnings"]
    package["readiness_summary"] = summary
    validate_production_package(package, require_ready=False)
    return package


def target_from_review(review: dict[str, Any]) -> ProductionPackageTarget:
    return ProductionPackageTarget(
        target=review["target"],
        ai_estimate_cm=review.get("ai_estimate_cm"),
        customer_confirmed_cm=review.get("customer_confirmed_cm"),
        customer_manual_cm=review.get("customer_manual_cm"),
        maker_verified_body_cm=review.get("maker_verified_body_cm"),
        selected_body_measurement_cm=review.get("selected_body_measurement_cm"),
        selected_body_measurement_source=review.get("selected_body_measurement_source"),
        maker_ease_allowance_cm=review.get("maker_ease_allowance_cm"),
        final_garment_cm=review.get("final_garment_cm"),
        confidence_tier=review.get("ai_confidence_tier"),
        estimated_error_cm=estimated_error_from_interval(review),
        interval_low_cm=review.get("ai_interval_low_cm"),
        interval_high_cm=review.get("ai_interval_high_cm"),
        product_action=review.get("product_action"),
        quality_flags=quality_flags_for_review(review),
        notes=list(review.get("maker_notes") or []),
    )


def estimated_error_from_interval(review: dict[str, Any]) -> float | None:
    estimate = review.get("ai_estimate_cm")
    low = review.get("ai_interval_low_cm")
    high = review.get("ai_interval_high_cm")
    if estimate is None or low is None or high is None:
        return None
    return round(max(abs(float(estimate) - float(low)), abs(float(high) - float(estimate))), 4)


def quality_flags_for_review(review: dict[str, Any]) -> list[str]:
    flags = []
    if review.get("ai_confidence_tier") == "low_confidence":
        flags.append("low_confidence")
    if review.get("selected_body_measurement_source") == BodyMeasurementSource.AI_ESTIMATE.value:
        flags.append("ai_only_body_measurement")
    if review.get("source") in {"manual_user_input_required", "landmark_required"}:
        flags.append(str(review["source"]))
    return flags


def package_status_from_maker_review(payload: dict[str, Any]) -> str:
    statuses = {review.get("production_status") for review in payload.get("reviews", [])}
    if payload.get("locked_at") or statuses == {ProductionStatus.LOCKED_FOR_PRODUCTION.value}:
        return PackageStatus.LOCKED_FOR_PRODUCTION.value
    if ProductionStatus.REVISION_REQUESTED.value in statuses:
        return PackageStatus.REVISION_REQUESTED.value
    if ProductionStatus.AWAITING_CUSTOMER_CONFIRMATION.value in statuses:
        return PackageStatus.AWAITING_CUSTOMER_CONFIRMATION.value
    if ProductionStatus.AWAITING_MAKER_REVIEW.value in statuses:
        return PackageStatus.AWAITING_MAKER_REVIEW.value
    if statuses and statuses <= {ProductionStatus.READY_FOR_PRODUCTION.value, ProductionStatus.LOCKED_FOR_PRODUCTION.value}:
        return PackageStatus.READY_FOR_PRODUCTION.value
    return PackageStatus.DRAFT.value


def summarize_package_readiness(package: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    for target in package.get("targets", []):
        name = target["target"]
        if target.get("selected_body_measurement_cm") is None:
            blockers.append(f"{name}: missing selected_body_measurement_cm")
        if target.get("maker_ease_allowance_cm") is None:
            blockers.append(f"{name}: missing maker_ease_allowance_cm")
        if target.get("final_garment_cm") is None:
            blockers.append(f"{name}: missing final_garment_cm")
        if target.get("confidence_tier") == "low_confidence" and target.get("selected_body_measurement_source") == BodyMeasurementSource.AI_ESTIMATE.value:
            blockers.append(f"{name}: low-confidence AI-only measurement cannot be used for production")
        if target.get("product_action") in {"user_input_required", "require_manual_confirmation"} and target.get("selected_body_measurement_source") == BodyMeasurementSource.AI_ESTIMATE.value:
            blockers.append(f"{name}: target requiring confirmation cannot use AI-only value")
        if target.get("selected_body_measurement_source") == BodyMeasurementSource.AI_ESTIMATE.value:
            warnings.append(f"{name}: using AI-only body measurement")
        if "low_confidence" in target.get("quality_flags", []):
            warnings.append(f"{name}: low confidence")
    if package.get("synthetic_calibrated_only"):
        warnings.append("Package contains synthetic-calibrated AI measurement lineage.")
    if package.get("real_world_validated"):
        warnings.append("Package is marked real-world validated; verify validation provenance.")
    status = PackageStatus.READY_FOR_PRODUCTION.value if not blockers else PackageStatus.BLOCKED.value
    if package.get("package_status") == PackageStatus.LOCKED_FOR_PRODUCTION.value and not blockers:
        status = PackageStatus.LOCKED_FOR_PRODUCTION.value
    return {
        "ready": not blockers,
        "recommended_status": status,
        "blockers": blockers,
        "warnings": warnings,
        "target_count": len(package.get("targets", [])),
        "required_targets_complete": not blockers,
    }


def validate_production_package(package: dict[str, Any], require_ready: bool = False) -> None:
    required = {
        "package_schema_version",
        "package_id",
        "order_id",
        "maker_id",
        "measurement_snapshot_id",
        "maker_review_id",
        "package_status",
        "targets",
        "created_at",
        "audit_event_ids",
        "warnings",
        "readiness_summary",
        "synthetic_calibrated_only",
        "real_world_validated",
    }
    missing = sorted(required - set(package))
    if missing:
        raise ProductionPackageError(f"Production package is missing required fields: {', '.join(missing)}")
    if package["package_schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise ProductionPackageError(f"Unsupported production package schema version: {package['package_schema_version']}")
    PackageStatus(package["package_status"])
    if not package["maker_id"]:
        raise ProductionPackageError("maker_id is required.")
    if not package["measurement_snapshot_id"]:
        raise ProductionPackageError("measurement_snapshot_id is required.")
    if package["real_world_validated"]:
        raise ProductionPackageError("real_world_validated must remain false until a future validated workflow sets it.")
    if not package["synthetic_calibrated_only"]:
        raise ProductionPackageError("synthetic_calibrated_only caveat must be preserved.")
    targets = package.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ProductionPackageError("Production package requires target records.")
    target_names = [target.get("target") for target in targets]
    if target_names != CONFIRMABLE_TARGETS:
        raise ProductionPackageError(f"Package targets must use stable order: {CONFIRMABLE_TARGETS}")
    for target in targets:
        validate_package_target(target)
    summary = summarize_package_readiness(package)
    if require_ready and summary["blockers"]:
        raise ProductionPackageError(f"Production package is not ready: {'; '.join(summary['blockers'])}")
    if package["package_status"] == PackageStatus.LOCKED_FOR_PRODUCTION.value:
        if not package.get("locked_at") or not package.get("locked_by_maker_id"):
            raise ProductionPackageError("Locked production package requires locked_at and locked_by_maker_id.")


def validate_package_target(target: dict[str, Any]) -> None:
    required = {
        "target",
        "ai_estimate_cm",
        "customer_confirmed_cm",
        "customer_manual_cm",
        "maker_verified_body_cm",
        "selected_body_measurement_cm",
        "selected_body_measurement_source",
        "maker_ease_allowance_cm",
        "final_garment_cm",
        "confidence_tier",
        "estimated_error_cm",
        "interval_low_cm",
        "interval_high_cm",
        "product_action",
        "quality_flags",
        "notes",
    }
    missing = sorted(required - set(target))
    if missing:
        raise ProductionPackageError(f"Package target {target.get('target', '<unknown>')} is missing fields: {', '.join(missing)}")
    if target["target"] not in CONFIRMABLE_TARGETS:
        raise ProductionPackageError(f"Unsupported package target: {target['target']}")
    if target.get("final_garment_cm") is not None and float(target["final_garment_cm"]) <= 0:
        raise ProductionPackageError(f"final_garment_cm for {target['target']} must be positive.")


def lock_production_package(package: dict[str, Any], maker_id: str, locked_at: str | None = None) -> dict[str, Any]:
    if package.get("package_status") == PackageStatus.LOCKED_FOR_PRODUCTION.value:
        raise ProductionPackageError("Production package is already locked.")
    validate_production_package(package, require_ready=True)
    timestamp = locked_at or utc_now()
    event = record_lock_event(
        event_id=f"package_locked__{package['package_id']}",
        maker_review_id=package["maker_review_id"],
        actor_id=maker_id,
        order_id=package.get("order_id"),
        measurement_snapshot_id=package["measurement_snapshot_id"],
        created_at=timestamp,
    )
    locked = {
        **package,
        "package_status": PackageStatus.LOCKED_FOR_PRODUCTION.value,
        "locked_at": timestamp,
        "locked_by_maker_id": maker_id,
        "updated_at": timestamp,
        "audit_event_ids": [*package.get("audit_event_ids", []), event["event_id"]],
    }
    locked["readiness_summary"] = summarize_package_readiness(locked)
    locked["warnings"] = locked["readiness_summary"]["warnings"]
    validate_production_package(locked, require_ready=True)
    return locked


def request_package_revision(
    package: dict[str, Any],
    requested_by_actor_id: str,
    requested_by_role: str,
    reason: str,
    changed_fields: list[str],
    previous_values: dict[str, Any],
    revised_values: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    if package.get("package_status") == PackageStatus.LOCKED_FOR_PRODUCTION.value and not reason:
        raise ProductionPackageError("Revision reason is required for locked production package changes.")
    revision = record_revision_request(
        maker_review_id=package["maker_review_id"],
        requested_by_actor_id=requested_by_actor_id,
        requested_by_role=requested_by_role,
        reason=reason,
        changed_fields=changed_fields,
        previous_values=previous_values,
        revised_values=revised_values,
        created_at=created_at,
    )
    revised = {
        **package,
        "package_status": PackageStatus.REVISION_REQUESTED.value,
        "updated_at": created_at or utc_now(),
        "locked_at": None,
        "locked_by_maker_id": None,
        "audit_event_ids": [*package.get("audit_event_ids", []), revision["revision_id"]],
        "readiness_summary": {
            **package.get("readiness_summary", {}),
            "revision": revision,
        },
    }
    return revised


def export_package_json(path: str | Path, package: dict[str, Any]) -> None:
    validate_production_package(package, require_ready=False)
    write_json(Path(path), package)


def export_sample_production_package(output_dir: str | Path = DEFAULT_OUTPUT_DIR, created_at: str | None = None) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    maker_review = sample_locked_maker_review(created_at=created_at)
    package = build_production_package(maker_review, package_id="sample_production_package_phase_4n", created_at=created_at)
    validate_production_package(package, require_ready=True)
    readiness = package["readiness_summary"]
    package_path = output_path / SAMPLE_PACKAGE_JSON
    readiness_path = output_path / SAMPLE_READINESS_JSON
    summary_path = output_path / PACKAGE_SUMMARY_MD
    export_package_json(package_path, package)
    write_json(readiness_path, readiness)
    summary_path.write_text(format_package_summary(package), encoding="utf-8")
    return {
        "sample_production_measurement_package_json": str(package_path),
        "sample_readiness_summary_json": str(readiness_path),
        "production_package_summary_md": str(summary_path),
    }


def sample_locked_maker_review(created_at: str | None = None) -> dict[str, Any]:
    customer = export_sample_customer_confirmation("artifacts/phase_4j_customer_measurement_confirmation", created_at=created_at)["payload"]
    customer = apply_demo_customer_confirmations(customer, updated_at=created_at)
    maker = build_maker_review_payload(customer, maker_id="demo_maker", customer_confirmation_id="sample_customer_confirmation_phase_4j", created_at=created_at)
    maker = apply_demo_maker_updates(maker, updated_at=created_at)
    return lock_for_production(maker, maker_id="demo_maker", locked_at=created_at)


def format_package_summary(package: dict[str, Any]) -> str:
    summary = package["readiness_summary"]
    lines = [
        "# Phase 4N Production Measurement Package",
        "",
        f"Package: `{package['package_id']}`",
        f"Order: `{package.get('order_id')}`",
        f"Maker: `{package['maker_id']}`",
        f"Status: `{package['package_status']}`",
        f"Ready: `{summary['ready']}`",
        f"Targets: `{summary['target_count']}`",
        "",
        "| Target | Body Source | Body cm | Ease cm | Final Garment cm |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for target in package["targets"]:
        lines.append(
            f"| {target['target']} | {target['selected_body_measurement_source']} | {format_optional(target['selected_body_measurement_cm'])} | "
            f"{format_optional(target['maker_ease_allowance_cm'])} | {format_optional(target['final_garment_cm'])} |"
        )
    if summary["blockers"]:
        lines.extend(["", "## Blockers", "", *[f"- {blocker}" for blocker in summary["blockers"]]])
    if summary["warnings"]:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in summary["warnings"]]])
    lines.append("")
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a sample production measurement package.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = export_sample_production_package(args.output)
    print(f"Production package: {paths['sample_production_measurement_package_json']}")
    print(f"Readiness summary: {paths['sample_readiness_summary_json']}")
    print(f"Summary: {paths['production_package_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
