from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

from training.measurements.measurement_audit_trail import sample_audit_data
from training.measurements.measurement_field_guidance import (
    RoleVisibility,
    list_guidance_for_role,
)
from training.measurements.measurement_result_schema import write_json
from training.measurements.production_measurement_package import (
    SAMPLE_PACKAGE_JSON,
    build_production_package,
    sample_locked_maker_review,
    validate_production_package,
)

DEFAULT_OUTPUT_DIR = "artifacts/phase_4o_fashionapp_measurement_mapping"
SAMPLE_CUSTOMER_RESPONSE_JSON = "sample_customer_measurement_response.json"
SAMPLE_MAKER_RESPONSE_JSON = "sample_maker_measurement_response.json"
SAMPLE_ADMIN_RESPONSE_JSON = "sample_admin_measurement_response.json"
MAPPING_SUMMARY_MD = "mapping_summary.md"

PACKAGE_REQUIRED_FIELDS = {
    "package_id",
    "order_id",
    "measurement_snapshot_id",
    "package_status",
    "targets",
    "synthetic_calibrated_only",
    "real_world_validated",
}

CUSTOMER_TARGET_FIELDS = {
    "target",
    "ai_estimate_cm",
    "customer_confirmed_cm",
    "customer_manual_cm",
    "selected_body_measurement_cm",
    "selected_body_measurement_source",
    "confidence_tier",
    "estimated_error_cm",
    "interval_low_cm",
    "interval_high_cm",
    "product_action",
    "quality_flags",
    "notes",
}

CUSTOMER_FORBIDDEN_CAMEL_KEYS = {
    "makerEaseAllowanceCm",
    "finalGarmentCm",
    "makerVerifiedBodyCm",
    "makerId",
    "makerReviewId",
    "lockedByMakerId",
}


class FashionAppMappingError(ValueError):
    """Raised when Body AI objects cannot be mapped into FashionApp payloads."""


@dataclass(frozen=True)
class FashionAppPayloadShape:
    """Small descriptor for database/API payload families exposed to FashionApp."""

    name: str
    required_fields: tuple[str, ...]
    role: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required_fields": list(self.required_fields),
            "role": self.role,
        }


API_PAYLOAD_SHAPES = [
    FashionAppPayloadShape("measurement_snapshot", ("snapshotId", "scanId", "measurementResult")),
    FashionAppPayloadShape("customer_measurement_confirmation", ("measurementSnapshotId", "confirmations"), "customer"),
    FashionAppPayloadShape("maker_measurement_review", ("reviewId", "makerId", "reviews"), "maker"),
    FashionAppPayloadShape("measurement_audit_event", ("eventId", "eventType", "actorId", "actorRole"), "admin"),
    FashionAppPayloadShape("production_measurement_package", ("packageId", "orderId", "targets")),
    FashionAppPayloadShape("production_package_target", ("target", "selectedBodyMeasurementCm", "productAction")),
    FashionAppPayloadShape("measurement_field_guidance", ("fieldKey", "label", "helperText", "infoIconText")),
]


def snake_to_camel(value: str) -> str:
    if not value:
        return value
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def camelize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {snake_to_camel(str(key)): camelize_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [camelize_keys(item) for item in value]
    return value


def payload_from(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_payload"):
        payload = value.to_payload()
    elif hasattr(value, "__dataclass_fields__"):
        payload = asdict(value)
    else:
        payload = value
    if not isinstance(payload, dict):
        raise FashionAppMappingError("Expected a dictionary-like payload.")
    return payload


def require_fields(payload: dict[str, Any], required_fields: set[str], label: str) -> None:
    missing = sorted(required_fields - set(payload))
    if missing:
        raise FashionAppMappingError(f"{label} is missing required fields: {', '.join(missing)}")


def map_measurement_result_to_api_payload(measurement_result: Any) -> dict[str, Any]:
    return camelize_keys(payload_from(measurement_result))


def map_snapshot_to_api_payload(snapshot: Any) -> dict[str, Any]:
    payload = payload_from(snapshot)
    require_fields(payload, {"snapshot_id", "scan_id", "measurement_result"}, "measurement snapshot")
    return camelize_keys(payload)


def map_customer_confirmation_to_api_payload(customer_confirmation: Any) -> dict[str, Any]:
    payload = payload_from(customer_confirmation)
    require_fields(payload, {"measurement_snapshot_id", "confirmations"}, "customer confirmation")
    return camelize_keys(payload)


def map_maker_review_to_api_payload(maker_review: Any) -> dict[str, Any]:
    payload = payload_from(maker_review)
    require_fields(payload, {"review_id", "maker_id", "reviews"}, "maker review")
    return camelize_keys(payload)


def map_audit_event_to_api_payload(audit_event: Any) -> dict[str, Any]:
    payload = payload_from(audit_event)
    require_fields(payload, {"event_id", "event_type", "actor_id", "actor_role"}, "audit event")
    return camelize_keys(payload)


def map_production_package_to_api_payload(production_package: Any) -> dict[str, Any]:
    package = payload_from(production_package)
    require_fields(package, PACKAGE_REQUIRED_FIELDS, "production measurement package")
    validate_production_package(package, require_ready=False)
    return camelize_keys(package)


def map_field_guidance_to_api_payload(field_guidance: Any) -> dict[str, Any]:
    payload = payload_from(field_guidance)
    require_fields(payload, {"field_key", "label", "helper_text", "info_icon_text"}, "field guidance")
    return camelize_keys(payload)


def build_customer_measurement_response(production_package: dict[str, Any]) -> dict[str, Any]:
    require_fields(production_package, PACKAGE_REQUIRED_FIELDS, "production measurement package")
    validate_production_package(production_package, require_ready=False)
    response = {
        "role": "customer",
        "package": {
            "package_id": production_package["package_id"],
            "order_id": production_package.get("order_id"),
            "customer_id": production_package.get("customer_id"),
            "measurement_snapshot_id": production_package["measurement_snapshot_id"],
            "package_status": production_package["package_status"],
            "targets": [customer_target_payload(target) for target in production_package["targets"]],
            "synthetic_calibrated_only": production_package["synthetic_calibrated_only"],
            "real_world_validated": production_package["real_world_validated"],
            "warnings": production_package.get("warnings", []),
        },
        "field_guidance": list_guidance_for_role(RoleVisibility.CUSTOMER),
        "caveats": [
            "Customer payloads contain body measurements and fit guidance only.",
            "Maker ease/allowance and final garment measurements are hidden from customer responses.",
            "Synthetic-calibrated AI estimates are not real-world production validation.",
        ],
    }
    camel = camelize_keys(response)
    reject_customer_forbidden_fields(camel)
    return camel


def build_maker_measurement_response(production_package: dict[str, Any]) -> dict[str, Any]:
    api_package = map_production_package_to_api_payload(production_package)
    return {
        "role": "maker",
        "package": api_package,
        "fieldGuidance": camelize_keys(list_guidance_for_role(RoleVisibility.MAKER)),
        "caveats": [
            "Maker payloads include maker-only ease/allowance and final garment measurements.",
            "Final garment measurements remain blocked until readiness checks pass.",
        ],
    }


def build_admin_measurement_response(
    production_package: dict[str, Any],
    audit_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    api_package = map_production_package_to_api_payload(production_package)
    events = [map_audit_event_to_api_payload(event) for event in (audit_events or [])]
    return {
        "role": "admin",
        "package": api_package,
        "auditEvents": events,
        "auditEventIds": api_package.get("auditEventIds", []),
        "fieldGuidance": camelize_keys(list_guidance_for_role(RoleVisibility.ADMIN)),
        "payloadShapes": [shape.to_payload() for shape in API_PAYLOAD_SHAPES],
        "metadata": {
            "syntheticCalibratedOnly": api_package["syntheticCalibratedOnly"],
            "realWorldValidated": api_package["realWorldValidated"],
        },
    }


def customer_target_payload(target: dict[str, Any]) -> dict[str, Any]:
    return {key: target.get(key) for key in sorted(CUSTOMER_TARGET_FIELDS)}


def reject_customer_forbidden_fields(payload: Any) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(CUSTOMER_FORBIDDEN_CAMEL_KEYS & set(payload))
        if forbidden:
            raise FashionAppMappingError(f"Customer response contains maker-only fields: {', '.join(forbidden)}")
        for value in payload.values():
            reject_customer_forbidden_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            reject_customer_forbidden_fields(item)


def contains_key(payload: Any, key: str) -> bool:
    if isinstance(payload, dict):
        return key in payload or any(contains_key(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(contains_key(item, key) for item in payload)
    return False


def export_sample_mapping(output_dir: str | Path = DEFAULT_OUTPUT_DIR, created_at: str | None = None) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = created_at or "2026-05-27T00:00:00Z"
    maker_review = sample_locked_maker_review(created_at=timestamp)
    events, _ = sample_audit_data()
    package = build_production_package(
        maker_review,
        audit_events=events,
        package_id="sample_production_package_phase_4o",
        created_at=timestamp,
    )
    customer = build_customer_measurement_response(package)
    maker = build_maker_measurement_response(package)
    admin = build_admin_measurement_response(package, audit_events=events)
    paths = {
        "sample_customer_measurement_response_json": output_path / SAMPLE_CUSTOMER_RESPONSE_JSON,
        "sample_maker_measurement_response_json": output_path / SAMPLE_MAKER_RESPONSE_JSON,
        "sample_admin_measurement_response_json": output_path / SAMPLE_ADMIN_RESPONSE_JSON,
        "mapping_summary_md": output_path / MAPPING_SUMMARY_MD,
    }
    write_json(paths["sample_customer_measurement_response_json"], customer)
    write_json(paths["sample_maker_measurement_response_json"], maker)
    write_json(paths["sample_admin_measurement_response_json"], admin)
    paths["mapping_summary_md"].write_text(format_mapping_summary(customer, maker, admin), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def format_mapping_summary(
    customer_response: dict[str, Any],
    maker_response: dict[str, Any],
    admin_response: dict[str, Any],
) -> str:
    customer_targets = len(customer_response["package"]["targets"])
    maker_targets = len(maker_response["package"]["targets"])
    audit_count = len(admin_response["auditEvents"])
    return "\n".join(
        [
            "# Phase 4O FashionApp Measurement Mapping",
            "",
            f"Customer targets: `{customer_targets}`",
            f"Maker targets: `{maker_targets}`",
            f"Admin audit events: `{audit_count}`",
            "",
            "The mapper converts internal Body AI snake_case payloads into FashionApp API camelCase payloads.",
            "Customer responses exclude maker-only ease/allowance fields.",
            "Maker responses include maker ease/allowance and final garment measurements.",
            "Admin responses include audit references, audit events, and internal validation caveats.",
            "",
        ]
    )


def json_dumps_deterministic(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export FashionApp measurement mapping sample payloads.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = export_sample_mapping(args.output)
    print(f"Customer response: {paths['sample_customer_measurement_response_json']}")
    print(f"Maker response: {paths['sample_maker_measurement_response_json']}")
    print(f"Admin response: {paths['sample_admin_measurement_response_json']}")
    print(f"Summary: {paths['mapping_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
