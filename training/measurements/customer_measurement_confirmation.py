from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.measurement_result_schema import SUPPORTED_TARGETS, write_json
from training.measurements.measurement_snapshot_store import (
    SAMPLE_SNAPSHOT_JSON,
    create_snapshot,
    load_snapshot,
    validate_snapshot,
)
from training.measurements.body_ai_inference import run_body_ai_measurement

CONFIRMATION_SCHEMA_VERSION = "phase_4j_customer_measurement_confirmation_v1"
DEFAULT_OUTPUT_DIR = "artifacts/phase_4j_customer_measurement_confirmation"
SAMPLE_CONFIRMATION_JSON = "sample_customer_confirmation_payload.json"
CONFIRMATION_SUMMARY_MD = "customer_confirmation_summary.md"

CONFIRMABLE_TARGETS = [
    "chest",
    "waist",
    "hip",
    "thigh",
    "shoulder",
    "calf",
    "height",
    "inseam",
    "sleeve",
    "neck",
]
MANUAL_REQUIRED_TARGETS = {"height", "inseam", "sleeve", "neck"}
MANUAL_OR_MAKER_REVIEW_TARGETS = {"shoulder", "calf"}
EASE_ALLOWANCE_FIELDS = {
    "ease_cm",
    "allowance_cm",
    "maker_ease_cm",
    "garment_allowance_cm",
    "wearing_ease_cm",
    "design_ease_cm",
}
REALISTIC_RANGES_CM = {
    "chest": (40.0, 180.0),
    "waist": (35.0, 170.0),
    "hip": (45.0, 190.0),
    "thigh": (25.0, 100.0),
    "shoulder": (20.0, 80.0),
    "calf": (15.0, 70.0),
    "height": (90.0, 230.0),
    "inseam": (35.0, 130.0),
    "sleeve": (30.0, 110.0),
    "neck": (20.0, 70.0),
}
THIGH_CAUTION_NOTE = "Thigh requires caution because Phase 4F documented 0.8400 synthetic interval coverage."


class CustomerConfirmationError(ValueError):
    """Raised when a customer measurement confirmation payload is invalid."""


class FitPreference(str, Enum):
    SNUG = "snug"
    REGULAR = "regular"
    RELAXED = "relaxed"
    LOOSE = "loose"
    CUSTOM_NOTE = "custom_note"


class ConfirmationStatus(str, Enum):
    NEEDS_CUSTOMER_REVIEW = "needs_customer_review"
    CUSTOMER_CONFIRMED = "customer_confirmed"
    MANUAL_VALUE_REQUIRED = "manual_value_required"
    MANUAL_VALUE_PROVIDED = "manual_value_provided"
    BLOCKED_MISSING_REQUIRED_VALUE = "blocked_missing_required_value"


@dataclass(frozen=True)
class CustomerMeasurementConfirmation:
    measurement_snapshot_id: str
    user_id: str | None
    order_id: str | None
    target: str
    ai_estimate_cm: float | None
    interval_low_cm: float | None
    interval_high_cm: float | None
    confidence_tier: str
    product_action: str
    customer_confirmed_cm: float | None
    customer_manual_cm: float | None
    source: str
    confirmation_status: ConfirmationStatus
    fit_preference: FitPreference
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now())
    updated_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confirmation_status"] = self.confirmation_status.value
        payload["fit_preference"] = self.fit_preference.value
        return payload


def build_customer_confirmation_payload(
    snapshot: dict[str, Any],
    fit_preference: str | FitPreference = FitPreference.REGULAR,
    created_at: str | None = None,
) -> dict[str, Any]:
    validate_snapshot(snapshot)
    preference = coerce_fit_preference(fit_preference)
    result_targets = {target["target"]: target for target in snapshot["measurement_result"]["targets"]}
    confirmations = [
        confirmation_from_target(snapshot, result_targets[target], preference, created_at=created_at)
        for target in CONFIRMABLE_TARGETS
    ]
    payload = {
        "confirmation_schema_version": CONFIRMATION_SCHEMA_VERSION,
        "measurement_snapshot_id": snapshot["snapshot_id"],
        "scan_id": snapshot["scan_id"],
        "user_id": snapshot.get("user_id"),
        "order_id": snapshot.get("order_id"),
        "synthetic_calibrated_only": snapshot["synthetic_calibrated_only"],
        "real_world_validated": snapshot["real_world_validated"],
        "fit_preference": preference.value,
        "confirmations": [confirmation.to_payload() for confirmation in confirmations],
        "caveats": [
            "Customers confirm body measurements and fit preference only.",
            "Maker ease and allowance are intentionally excluded from the customer payload.",
            "Synthetic-calibrated AI estimates require appropriate manual or maker review before garment production.",
        ],
        "created_at": created_at or utc_now(),
        "updated_at": None,
    }
    validate_customer_confirmation_payload(payload, require_complete=False)
    return payload


def confirmation_from_target(
    snapshot: dict[str, Any],
    target: dict[str, Any],
    fit_preference: FitPreference,
    created_at: str | None = None,
) -> CustomerMeasurementConfirmation:
    target_name = target["target"]
    interval = target.get("interval") or {}
    status = initial_status(target)
    notes = list(target.get("notes") or [])
    if target_name == "thigh":
        notes.append(THIGH_CAUTION_NOTE)
    if target_name in MANUAL_OR_MAKER_REVIEW_TARGETS:
        notes.append(f"{target_name} should be manually confirmed or reviewed by a maker before final use.")
    return CustomerMeasurementConfirmation(
        measurement_snapshot_id=snapshot["snapshot_id"],
        user_id=snapshot.get("user_id"),
        order_id=snapshot.get("order_id"),
        target=target_name,
        ai_estimate_cm=target.get("estimate_cm"),
        interval_low_cm=interval.get("low_cm"),
        interval_high_cm=interval.get("high_cm"),
        confidence_tier=str(target.get("confidence_tier")),
        product_action=str(target.get("product_action")),
        customer_confirmed_cm=None,
        customer_manual_cm=None,
        source=str(target.get("source")),
        confirmation_status=status,
        fit_preference=fit_preference,
        notes=notes,
        created_at=created_at or utc_now(),
    )


def initial_status(target: dict[str, Any]) -> ConfirmationStatus:
    target_name = target["target"]
    product_action = target.get("product_action")
    estimate = target.get("estimate_cm")
    if target_name in MANUAL_REQUIRED_TARGETS:
        return ConfirmationStatus.MANUAL_VALUE_REQUIRED
    if product_action in {"user_input_required", "require_manual_confirmation", "maker_review_required"}:
        return ConfirmationStatus.NEEDS_CUSTOMER_REVIEW
    if estimate is None:
        return ConfirmationStatus.NEEDS_CUSTOMER_REVIEW
    return ConfirmationStatus.NEEDS_CUSTOMER_REVIEW


def apply_customer_measurement_updates(
    payload: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    updated_at: str | None = None,
) -> dict[str, Any]:
    reject_ease_allowance_fields(updates)
    validate_customer_confirmation_payload(payload, require_complete=False)
    timestamp = updated_at or utc_now()
    updated_confirmations = []
    for confirmation in payload["confirmations"]:
        target = confirmation["target"]
        next_confirmation = dict(confirmation)
        if target in updates:
            update = updates[target]
            for field in ("customer_confirmed_cm", "customer_manual_cm", "notes", "fit_preference"):
                if field in update:
                    next_confirmation[field] = update[field]
            next_confirmation["updated_at"] = timestamp
            next_confirmation["confirmation_status"] = status_after_update(next_confirmation)
        updated_confirmations.append(next_confirmation)
    updated_payload = {**payload, "confirmations": updated_confirmations, "updated_at": timestamp}
    validate_customer_confirmation_payload(updated_payload, require_complete=False)
    return updated_payload


def status_after_update(confirmation: dict[str, Any]) -> str:
    target = confirmation["target"]
    manual_value = confirmation.get("customer_manual_cm")
    confirmed_value = confirmation.get("customer_confirmed_cm")
    if manual_value is not None:
        return ConfirmationStatus.MANUAL_VALUE_PROVIDED.value
    if confirmed_value is not None:
        return ConfirmationStatus.CUSTOMER_CONFIRMED.value
    if target in MANUAL_REQUIRED_TARGETS:
        return ConfirmationStatus.MANUAL_VALUE_REQUIRED.value
    return ConfirmationStatus.NEEDS_CUSTOMER_REVIEW.value


def validate_customer_confirmation_payload(payload: dict[str, Any], require_complete: bool = False) -> None:
    reject_ease_allowance_fields(payload)
    required = {
        "confirmation_schema_version",
        "measurement_snapshot_id",
        "user_id",
        "order_id",
        "synthetic_calibrated_only",
        "real_world_validated",
        "fit_preference",
        "confirmations",
        "caveats",
        "created_at",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CustomerConfirmationError(f"Customer confirmation payload is missing required fields: {', '.join(missing)}")
    if payload["confirmation_schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise CustomerConfirmationError(f"Unsupported confirmation schema version: {payload['confirmation_schema_version']}")
    if not payload["measurement_snapshot_id"]:
        raise CustomerConfirmationError("measurement_snapshot_id is required.")
    if not payload["synthetic_calibrated_only"]:
        raise CustomerConfirmationError("synthetic_calibrated_only must be preserved as true.")
    if payload["real_world_validated"]:
        raise CustomerConfirmationError("real_world_validated must remain false until real validation exists.")
    coerce_fit_preference(payload["fit_preference"])
    confirmations = payload.get("confirmations")
    if not isinstance(confirmations, list) or not confirmations:
        raise CustomerConfirmationError("confirmations must be a non-empty list.")
    target_names = [confirmation.get("target") for confirmation in confirmations]
    if target_names != CONFIRMABLE_TARGETS:
        raise CustomerConfirmationError(f"Customer confirmations must use stable target order: {CONFIRMABLE_TARGETS}")
    for confirmation in confirmations:
        validate_confirmation_record(confirmation)
    if require_complete:
        missing_manual = [
            confirmation["target"]
            for confirmation in confirmations
            if confirmation["target"] in MANUAL_REQUIRED_TARGETS and confirmation.get("customer_manual_cm") is None and confirmation.get("customer_confirmed_cm") is None
        ]
        if missing_manual:
            raise CustomerConfirmationError(f"Missing required manual measurement values: {', '.join(missing_manual)}")
        blocked_ai = [
            confirmation["target"]
            for confirmation in confirmations
            if confirmation["product_action"] == "require_manual_confirmation"
            and confirmation.get("customer_confirmed_cm") is None
            and confirmation.get("customer_manual_cm") is None
        ]
        if blocked_ai:
            raise CustomerConfirmationError(f"Measurements requiring manual confirmation are not finalized: {', '.join(blocked_ai)}")


def validate_confirmation_record(confirmation: dict[str, Any]) -> None:
    reject_ease_allowance_fields(confirmation)
    required = {
        "measurement_snapshot_id",
        "user_id",
        "order_id",
        "target",
        "ai_estimate_cm",
        "interval_low_cm",
        "interval_high_cm",
        "confidence_tier",
        "product_action",
        "customer_confirmed_cm",
        "customer_manual_cm",
        "source",
        "confirmation_status",
        "fit_preference",
        "notes",
        "created_at",
        "updated_at",
    }
    missing = sorted(required - set(confirmation))
    if missing:
        raise CustomerConfirmationError(f"Customer confirmation for {confirmation.get('target', '<unknown>')} is missing fields: {', '.join(missing)}")
    target = confirmation["target"]
    if target not in CONFIRMABLE_TARGETS:
        raise CustomerConfirmationError(f"Unsupported customer confirmation target: {target}")
    coerce_fit_preference(confirmation["fit_preference"])
    coerce_status(confirmation["confirmation_status"])
    for field in ("customer_confirmed_cm", "customer_manual_cm"):
        value = confirmation.get(field)
        if value is not None:
            validate_measurement_value(target, float(value), field)


def validate_measurement_value(target: str, value: float, field: str) -> None:
    if value <= 0:
        raise CustomerConfirmationError(f"{field} for {target} must be a positive number.")
    low, high = REALISTIC_RANGES_CM[target]
    if value < low or value > high:
        raise CustomerConfirmationError(f"{field} for {target} is outside plausible range {low}-{high} cm: {value}")


def reject_ease_allowance_fields(payload: Any) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(EASE_ALLOWANCE_FIELDS & set(payload))
        if forbidden:
            raise CustomerConfirmationError(f"Customer payload must not include maker ease/allowance fields: {', '.join(forbidden)}")
        for value in payload.values():
            reject_ease_allowance_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            reject_ease_allowance_fields(item)


def coerce_fit_preference(value: str | FitPreference) -> FitPreference:
    if isinstance(value, FitPreference):
        return value
    try:
        return FitPreference(str(value))
    except ValueError as exc:
        raise CustomerConfirmationError(f"Unsupported fit preference: {value}") from exc


def coerce_status(value: str | ConfirmationStatus) -> ConfirmationStatus:
    if isinstance(value, ConfirmationStatus):
        return value
    try:
        return ConfirmationStatus(str(value))
    except ValueError as exc:
        raise CustomerConfirmationError(f"Unsupported confirmation status: {value}") from exc


def write_customer_confirmation_payload(path: str | Path, payload: dict[str, Any]) -> None:
    validate_customer_confirmation_payload(payload, require_complete=False)
    write_json(Path(path), payload)


def format_confirmation_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 4J Customer Measurement Confirmation",
        "",
        f"Snapshot: `{payload['measurement_snapshot_id']}`",
        f"User: `{payload.get('user_id')}`",
        f"Order: `{payload.get('order_id')}`",
        f"Fit preference: `{payload['fit_preference']}`",
        f"Synthetic calibrated only: `{payload['synthetic_calibrated_only']}`",
        f"Real-world validated: `{payload['real_world_validated']}`",
        "",
        "| Target | AI Estimate | Interval | Action | Status | Source |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for confirmation in payload["confirmations"]:
        if confirmation["ai_estimate_cm"] is None:
            estimate = ""
            interval = ""
        else:
            estimate = f"{float(confirmation['ai_estimate_cm']):.4f}"
            interval = f"{float(confirmation['interval_low_cm']):.4f} to {float(confirmation['interval_high_cm']):.4f}"
        lines.append(
            f"| {confirmation['target']} | {estimate} | {interval} | {confirmation['product_action']} | {confirmation['confirmation_status']} | {confirmation['source']} |"
        )
    lines.extend(
        [
            "",
            "Customers confirm body measurements and fit preference only. Maker ease and allowance are intentionally excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def export_sample_customer_confirmation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    snapshot_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = load_snapshot(snapshot_path) if snapshot_path is not None else sample_snapshot(created_at=created_at)
    payload = build_customer_confirmation_payload(snapshot, fit_preference=FitPreference.REGULAR, created_at=created_at)
    write_customer_confirmation_payload(output_path / SAMPLE_CONFIRMATION_JSON, payload)
    (output_path / CONFIRMATION_SUMMARY_MD).write_text(format_confirmation_summary(payload), encoding="utf-8")
    return {
        "sample_customer_confirmation_payload_json": str(output_path / SAMPLE_CONFIRMATION_JSON),
        "customer_confirmation_summary_md": str(output_path / CONFIRMATION_SUMMARY_MD),
        "payload": payload,
    }


def sample_snapshot(created_at: str | None = None) -> dict[str, Any]:
    result = run_body_ai_measurement(
        scan_id="sample_000007",
        user_id="demo_user",
        order_id="demo_order",
        front_image_path="data/synthetic/phase_3t/images/front/sample_000001_front.png",
        side_image_path="data/synthetic/phase_3t/images/side/sample_000001_side.png",
        height_cm=172.0,
        generated_at=created_at,
    )
    return create_snapshot(
        snapshot_id="sample_snapshot_phase_4j",
        scan_id="sample_000007",
        user_id="demo_user",
        order_id="demo_order",
        front_image_path="data/synthetic/phase_3t/images/front/sample_000001_front.png",
        side_image_path="data/synthetic/phase_3t/images/side/sample_000001_side.png",
        height_cm=172.0,
        measurement_result=result,
        created_at=created_at,
    )


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a customer measurement confirmation payload.")
    parser.add_argument("--snapshot")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = export_sample_customer_confirmation(args.output, snapshot_path=args.snapshot)
    print(f"Customer confirmation payload: {result['sample_customer_confirmation_payload_json']}")
    print(f"Customer confirmation summary: {result['customer_confirmation_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
