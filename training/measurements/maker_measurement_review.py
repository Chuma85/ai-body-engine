from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.customer_measurement_confirmation import (
    CONFIRMABLE_TARGETS,
    DEFAULT_OUTPUT_DIR as CUSTOMER_CONFIRMATION_OUTPUT_DIR,
    REALISTIC_RANGES_CM,
    SAMPLE_CONFIRMATION_JSON,
    CustomerConfirmationError,
    FitPreference,
    export_sample_customer_confirmation,
    reject_ease_allowance_fields,
    validate_customer_confirmation_payload,
)
from training.measurements.measurement_result_schema import write_json

MAKER_REVIEW_SCHEMA_VERSION = "phase_4k_maker_measurement_review_v1"
DEFAULT_OUTPUT_DIR = "artifacts/phase_4k_maker_measurement_review"
SAMPLE_MAKER_REVIEW_JSON = "sample_maker_review_payload.json"
SAMPLE_FINAL_GARMENT_JSON = "sample_final_garment_measurements.json"
MAKER_REVIEW_SUMMARY_MD = "maker_review_summary.md"
EASE_MIN_CM = -20.0
EASE_MAX_CM = 80.0


class MakerReviewError(ValueError):
    """Raised when maker measurement review data is invalid."""


class ProductionStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_CUSTOMER_CONFIRMATION = "awaiting_customer_confirmation"
    AWAITING_MAKER_REVIEW = "awaiting_maker_review"
    READY_FOR_PRODUCTION = "ready_for_production"
    LOCKED_FOR_PRODUCTION = "locked_for_production"
    REVISION_REQUESTED = "revision_requested"


class BodyMeasurementSource(str, Enum):
    MAKER_VERIFIED = "maker_verified_body_cm"
    CUSTOMER_CONFIRMED = "customer_confirmed_cm"
    CUSTOMER_MANUAL = "customer_manual_cm"
    AI_ESTIMATE = "ai_estimate_cm"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MakerMeasurementReview:
    review_id: str
    measurement_snapshot_id: str
    customer_confirmation_id: str | None
    user_id: str | None
    customer_id: str | None
    maker_id: str
    order_id: str | None
    target: str
    ai_estimate_cm: float | None
    ai_interval_low_cm: float | None
    ai_interval_high_cm: float | None
    ai_confidence_tier: str | None
    product_action: str | None
    source: str | None
    customer_confirmed_cm: float | None
    customer_manual_cm: float | None
    maker_verified_body_cm: float | None
    maker_ease_allowance_cm: float | None
    final_garment_cm: float | None
    selected_body_measurement_cm: float | None
    selected_body_measurement_source: BodyMeasurementSource
    fit_preference: FitPreference
    maker_notes: list[str] = field(default_factory=list)
    production_status: ProductionStatus = ProductionStatus.DRAFT
    created_at: str = field(default_factory=lambda: utc_now())
    updated_at: str | None = None
    locked_at: str | None = None
    locked_by_maker_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_body_measurement_source"] = self.selected_body_measurement_source.value
        payload["fit_preference"] = self.fit_preference.value
        payload["production_status"] = self.production_status.value
        return payload


def build_maker_review_payload(
    customer_payload: dict[str, Any],
    maker_id: str,
    customer_confirmation_id: str | None = None,
    review_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not maker_id:
        raise MakerReviewError("maker_id is required for maker measurement review.")
    validate_customer_confirmation_payload(customer_payload, require_complete=False)
    timestamp = created_at or utc_now()
    reviews = [
        review_from_customer_confirmation(record, customer_payload, maker_id, customer_confirmation_id, timestamp)
        for record in customer_payload["confirmations"]
    ]
    payload = {
        "maker_review_schema_version": MAKER_REVIEW_SCHEMA_VERSION,
        "review_id": review_id or f"maker_review_{customer_payload['measurement_snapshot_id']}",
        "measurement_snapshot_id": customer_payload["measurement_snapshot_id"],
        "customer_confirmation_id": customer_confirmation_id,
        "user_id": customer_payload.get("user_id"),
        "customer_id": customer_payload.get("user_id"),
        "maker_id": maker_id,
        "order_id": customer_payload.get("order_id"),
        "fit_preference": customer_payload["fit_preference"],
        "reviews": [review.to_payload() for review in reviews],
        "created_at": timestamp,
        "updated_at": None,
        "locked_at": None,
        "locked_by_maker_id": None,
        "caveats": [
            "Maker review converts body measurements into garment measurements.",
            "maker_ease_allowance_cm is maker-only and must not appear in customer confirmation payloads.",
            "Synthetic-calibrated AI estimates still require maker judgment before production.",
        ],
    }
    validate_maker_review_payload(payload, require_ready=False)
    return payload


def review_from_customer_confirmation(
    confirmation: dict[str, Any],
    customer_payload: dict[str, Any],
    maker_id: str,
    customer_confirmation_id: str | None,
    created_at: str,
) -> MakerMeasurementReview:
    selected_value, selected_source = select_body_measurement(confirmation)
    return MakerMeasurementReview(
        review_id=f"{customer_payload['measurement_snapshot_id']}__{confirmation['target']}",
        measurement_snapshot_id=customer_payload["measurement_snapshot_id"],
        customer_confirmation_id=customer_confirmation_id,
        user_id=customer_payload.get("user_id"),
        customer_id=customer_payload.get("user_id"),
        maker_id=maker_id,
        order_id=customer_payload.get("order_id"),
        target=confirmation["target"],
        ai_estimate_cm=confirmation.get("ai_estimate_cm"),
        ai_interval_low_cm=confirmation.get("interval_low_cm"),
        ai_interval_high_cm=confirmation.get("interval_high_cm"),
        ai_confidence_tier=confirmation.get("confidence_tier"),
        product_action=confirmation.get("product_action"),
        source=confirmation.get("source"),
        customer_confirmed_cm=confirmation.get("customer_confirmed_cm"),
        customer_manual_cm=confirmation.get("customer_manual_cm"),
        maker_verified_body_cm=None,
        maker_ease_allowance_cm=None,
        final_garment_cm=None,
        selected_body_measurement_cm=selected_value,
        selected_body_measurement_source=selected_source,
        fit_preference=FitPreference(customer_payload["fit_preference"]),
        maker_notes=[],
        production_status=initial_status_for_review(confirmation, selected_source),
        created_at=created_at,
    )


def select_body_measurement(record: dict[str, Any]) -> tuple[float | None, BodyMeasurementSource]:
    if record.get("maker_verified_body_cm") is not None:
        return float(record["maker_verified_body_cm"]), BodyMeasurementSource.MAKER_VERIFIED
    if record.get("customer_confirmed_cm") is not None:
        return float(record["customer_confirmed_cm"]), BodyMeasurementSource.CUSTOMER_CONFIRMED
    if record.get("customer_manual_cm") is not None:
        return float(record["customer_manual_cm"]), BodyMeasurementSource.CUSTOMER_MANUAL
    if ai_estimate_can_be_selected(record):
        return float(record["ai_estimate_cm"]), BodyMeasurementSource.AI_ESTIMATE
    return None, BodyMeasurementSource.UNAVAILABLE


def ai_estimate_can_be_selected(record: dict[str, Any]) -> bool:
    return (
        record.get("ai_estimate_cm") is not None
        and record.get("product_action") == "accept_as_ai_estimate"
        and record.get("confidence_tier") != "low_confidence"
        and record.get("source") not in {"manual_user_input_required", "landmark_required", "unavailable"}
    )


def initial_status_for_review(record: dict[str, Any], selected_source: BodyMeasurementSource) -> ProductionStatus:
    if selected_source == BodyMeasurementSource.UNAVAILABLE:
        if record["target"] in {"height", "inseam", "sleeve", "neck"}:
            return ProductionStatus.AWAITING_CUSTOMER_CONFIRMATION
        return ProductionStatus.AWAITING_MAKER_REVIEW
    return ProductionStatus.AWAITING_MAKER_REVIEW


def apply_maker_review_updates(
    payload: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    updated_at: str | None = None,
    allow_revision: bool = False,
) -> dict[str, Any]:
    validate_maker_review_payload(payload, require_ready=False)
    if is_locked(payload) and not allow_revision:
        raise MakerReviewError("Locked maker review cannot be edited without an explicit revision path.")
    timestamp = updated_at or utc_now()
    next_reviews = []
    for review in payload["reviews"]:
        target = review["target"]
        next_review = dict(review)
        if target in updates:
            update = dict(updates[target])
            for field in ("maker_verified_body_cm", "maker_ease_allowance_cm", "maker_notes", "production_status"):
                if field in update:
                    next_review[field] = update[field]
            selected_value, selected_source = select_body_measurement(next_review)
            next_review["selected_body_measurement_cm"] = round_float(selected_value)
            next_review["selected_body_measurement_source"] = selected_source.value
            next_review["final_garment_cm"] = calculate_final_garment_cm(selected_value, next_review.get("maker_ease_allowance_cm"))
            next_review["updated_at"] = timestamp
            if next_review["final_garment_cm"] is not None and next_review.get("production_status") not in {
                ProductionStatus.LOCKED_FOR_PRODUCTION.value,
                ProductionStatus.REVISION_REQUESTED.value,
            }:
                next_review["production_status"] = ProductionStatus.READY_FOR_PRODUCTION.value
        next_reviews.append(next_review)
    next_payload = {**payload, "reviews": next_reviews, "updated_at": timestamp}
    if allow_revision and is_locked(payload):
        next_payload = {
            **next_payload,
            "locked_at": None,
            "locked_by_maker_id": None,
            "reviews": [
                {**review, "production_status": ProductionStatus.REVISION_REQUESTED.value, "locked_at": None, "locked_by_maker_id": None}
                for review in next_payload["reviews"]
            ],
        }
    validate_maker_review_payload(next_payload, require_ready=False)
    return next_payload


def lock_for_production(payload: dict[str, Any], maker_id: str, locked_at: str | None = None) -> dict[str, Any]:
    if not maker_id:
        raise MakerReviewError("maker_id is required to lock production measurements.")
    validate_maker_review_payload(payload, require_ready=True)
    timestamp = locked_at or utc_now()
    locked_reviews = [
        {
            **review,
            "production_status": ProductionStatus.LOCKED_FOR_PRODUCTION.value,
            "locked_at": timestamp,
            "locked_by_maker_id": maker_id,
            "updated_at": timestamp,
        }
        for review in payload["reviews"]
    ]
    locked_payload = {
        **payload,
        "reviews": locked_reviews,
        "locked_at": timestamp,
        "locked_by_maker_id": maker_id,
        "updated_at": timestamp,
    }
    validate_maker_review_payload(locked_payload, require_ready=True)
    return locked_payload


def is_locked(payload: dict[str, Any]) -> bool:
    return bool(payload.get("locked_at")) or any(review.get("production_status") == ProductionStatus.LOCKED_FOR_PRODUCTION.value for review in payload.get("reviews", []))


def calculate_final_garment_cm(body_cm: float | None, ease_cm: float | None) -> float | None:
    if body_cm is None or ease_cm is None:
        return None
    return round_float(float(body_cm) + float(ease_cm))


def validate_maker_review_payload(payload: dict[str, Any], require_ready: bool = False) -> None:
    required = {
        "maker_review_schema_version",
        "review_id",
        "measurement_snapshot_id",
        "maker_id",
        "fit_preference",
        "reviews",
        "created_at",
        "caveats",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise MakerReviewError(f"Maker review payload is missing required fields: {', '.join(missing)}")
    if payload["maker_review_schema_version"] != MAKER_REVIEW_SCHEMA_VERSION:
        raise MakerReviewError(f"Unsupported maker review schema version: {payload['maker_review_schema_version']}")
    if not payload["maker_id"]:
        raise MakerReviewError("maker_id is required.")
    FitPreference(payload["fit_preference"])
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise MakerReviewError("Maker review payload requires a non-empty reviews list.")
    target_names = [review.get("target") for review in reviews]
    if target_names != CONFIRMABLE_TARGETS:
        raise MakerReviewError(f"Maker reviews must use stable target order: {CONFIRMABLE_TARGETS}")
    for review in reviews:
        validate_review_record(review, require_ready=require_ready)


def validate_review_record(review: dict[str, Any], require_ready: bool = False) -> None:
    required = {
        "review_id",
        "measurement_snapshot_id",
        "maker_id",
        "target",
        "ai_estimate_cm",
        "ai_interval_low_cm",
        "ai_interval_high_cm",
        "ai_confidence_tier",
        "product_action",
        "source",
        "customer_confirmed_cm",
        "customer_manual_cm",
        "maker_verified_body_cm",
        "maker_ease_allowance_cm",
        "final_garment_cm",
        "selected_body_measurement_cm",
        "selected_body_measurement_source",
        "fit_preference",
        "maker_notes",
        "production_status",
        "created_at",
        "updated_at",
    }
    missing = sorted(required - set(review))
    if missing:
        raise MakerReviewError(f"Maker review for {review.get('target', '<unknown>')} is missing fields: {', '.join(missing)}")
    target = review["target"]
    if target not in CONFIRMABLE_TARGETS:
        raise MakerReviewError(f"Unsupported maker review target: {target}")
    ProductionStatus(review["production_status"])
    BodyMeasurementSource(review["selected_body_measurement_source"])
    FitPreference(review["fit_preference"])
    for field in ("customer_confirmed_cm", "customer_manual_cm", "maker_verified_body_cm", "selected_body_measurement_cm"):
        if review.get(field) is not None:
            validate_body_value(target, float(review[field]), field)
    if review.get("maker_ease_allowance_cm") is not None:
        validate_ease(float(review["maker_ease_allowance_cm"]))
    if review.get("final_garment_cm") is not None:
        if float(review["final_garment_cm"]) <= 0:
            raise MakerReviewError(f"final_garment_cm for {target} must be positive.")
        validate_final_value(target, float(review["final_garment_cm"]))
    if require_ready:
        validate_ready_for_production(review)


def validate_ready_for_production(review: dict[str, Any]) -> None:
    target = review["target"]
    if review.get("maker_ease_allowance_cm") is None:
        raise MakerReviewError(f"maker_ease_allowance_cm is required before production lock for {target}.")
    if review.get("final_garment_cm") is None:
        raise MakerReviewError(f"final_garment_cm is required before production lock for {target}.")
    source = review["selected_body_measurement_source"]
    if source == BodyMeasurementSource.UNAVAILABLE.value:
        raise MakerReviewError(f"{target} cannot be production-locked without a body measurement source.")
    if review.get("ai_confidence_tier") == "low_confidence" and source == BodyMeasurementSource.AI_ESTIMATE.value:
        raise MakerReviewError(f"Low-confidence AI-only {target} requires maker verification or manual confirmation.")
    if review.get("source") in {"manual_user_input_required", "landmark_required"} and source == BodyMeasurementSource.AI_ESTIMATE.value:
        raise MakerReviewError(f"{target} cannot be finalized from AI alone because it requires user input or landmarks.")
    if target in {"height", "inseam", "sleeve", "neck"} and source == BodyMeasurementSource.AI_ESTIMATE.value:
        raise MakerReviewError(f"{target} cannot be finalized from AI alone.")


def validate_body_value(target: str, value: float, field: str) -> None:
    if value <= 0:
        raise MakerReviewError(f"{field} for {target} must be positive.")
    low, high = REALISTIC_RANGES_CM[target]
    if value < low or value > high:
        raise MakerReviewError(f"{field} for {target} is outside plausible range {low}-{high} cm: {value}")


def validate_ease(value: float) -> None:
    if value < EASE_MIN_CM or value > EASE_MAX_CM:
        raise MakerReviewError(f"maker_ease_allowance_cm is outside plausible range {EASE_MIN_CM}-{EASE_MAX_CM} cm: {value}")


def validate_final_value(target: str, value: float) -> None:
    low, high = REALISTIC_RANGES_CM[target]
    if value < max(1.0, low + EASE_MIN_CM) or value > high + EASE_MAX_CM:
        raise MakerReviewError(f"final_garment_cm for {target} is outside plausible range.")


def build_final_garment_measurements(payload: dict[str, Any]) -> dict[str, Any]:
    validate_maker_review_payload(payload, require_ready=True)
    return {
        "review_id": payload["review_id"],
        "measurement_snapshot_id": payload["measurement_snapshot_id"],
        "maker_id": payload["maker_id"],
        "order_id": payload.get("order_id"),
        "fit_preference": payload["fit_preference"],
        "locked_at": payload.get("locked_at"),
        "locked_by_maker_id": payload.get("locked_by_maker_id"),
        "measurements": [
            {
                "target": review["target"],
                "selected_body_measurement_cm": review["selected_body_measurement_cm"],
                "selected_body_measurement_source": review["selected_body_measurement_source"],
                "maker_ease_allowance_cm": review["maker_ease_allowance_cm"],
                "final_garment_cm": review["final_garment_cm"],
            }
            for review in payload["reviews"]
        ],
    }


def format_maker_review_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 4K Maker Measurement Review",
        "",
        f"Review: `{payload['review_id']}`",
        f"Maker: `{payload['maker_id']}`",
        f"Order: `{payload.get('order_id')}`",
        f"Fit preference: `{payload['fit_preference']}`",
        "",
        "| Target | Body Source | Body cm | Ease cm | Final Garment cm | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for review in payload["reviews"]:
        lines.append(
            f"| {review['target']} | {review['selected_body_measurement_source']} | {format_optional(review['selected_body_measurement_cm'])} | "
            f"{format_optional(review['maker_ease_allowance_cm'])} | {format_optional(review['final_garment_cm'])} | {review['production_status']} |"
        )
    lines.extend(
        [
            "",
            "Body measurements describe the person. Maker ease/allowance converts body measurements into garment measurements.",
            "Ease and allowance are maker-only production decisions and are not exposed in the customer confirmation payload.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def export_sample_maker_review(output_dir: str | Path = DEFAULT_OUTPUT_DIR, created_at: str | None = None) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    customer = export_sample_customer_confirmation(CUSTOMER_CONFIRMATION_OUTPUT_DIR, created_at=created_at)["payload"]
    customer = apply_demo_customer_confirmations(customer, updated_at=created_at)
    payload = build_maker_review_payload(customer, maker_id="demo_maker", customer_confirmation_id="sample_customer_confirmation_phase_4j", created_at=created_at)
    payload = apply_demo_maker_updates(payload, updated_at=created_at)
    locked = lock_for_production(payload, maker_id="demo_maker", locked_at=created_at)
    final = build_final_garment_measurements(locked)
    write_json(output_path / SAMPLE_MAKER_REVIEW_JSON, locked)
    write_json(output_path / SAMPLE_FINAL_GARMENT_JSON, final)
    (output_path / MAKER_REVIEW_SUMMARY_MD).write_text(format_maker_review_summary(locked), encoding="utf-8")
    return {
        "sample_maker_review_payload_json": str(output_path / SAMPLE_MAKER_REVIEW_JSON),
        "sample_final_garment_measurements_json": str(output_path / SAMPLE_FINAL_GARMENT_JSON),
        "maker_review_summary_md": str(output_path / MAKER_REVIEW_SUMMARY_MD),
        "payload": locked,
        "final_garment_measurements": final,
    }


def apply_demo_customer_confirmations(payload: dict[str, Any], updated_at: str | None = None) -> dict[str, Any]:
    from training.measurements.customer_measurement_confirmation import apply_customer_measurement_updates

    return apply_customer_measurement_updates(
        payload,
        {
            "height": {"customer_manual_cm": 172.0},
            "inseam": {"customer_manual_cm": 78.0},
            "sleeve": {"customer_manual_cm": 61.0},
            "neck": {"customer_manual_cm": 38.0},
            "chest": {"customer_confirmed_cm": 105.2},
            "waist": {"customer_confirmed_cm": 88.5},
            "hip": {"customer_confirmed_cm": 106.1},
            "thigh": {"customer_confirmed_cm": 62.3},
            "shoulder": {"customer_manual_cm": 45.0},
            "calf": {"customer_manual_cm": 39.0},
        },
        updated_at=updated_at,
    )


def apply_demo_maker_updates(payload: dict[str, Any], updated_at: str | None = None) -> dict[str, Any]:
    updates = {
        "chest": {"maker_ease_allowance_cm": 8.0},
        "waist": {"maker_ease_allowance_cm": 4.0},
        "hip": {"maker_ease_allowance_cm": 6.0},
        "thigh": {"maker_ease_allowance_cm": 3.0},
        "shoulder": {"maker_ease_allowance_cm": 1.0},
        "calf": {"maker_ease_allowance_cm": 2.0},
        "height": {"maker_ease_allowance_cm": 0.0},
        "inseam": {"maker_ease_allowance_cm": 1.0},
        "sleeve": {"maker_ease_allowance_cm": 1.0},
        "neck": {"maker_ease_allowance_cm": 1.5},
    }
    return apply_maker_review_updates(payload, updates, updated_at=updated_at)


def round_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a sample maker measurement review payload.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = export_sample_maker_review(args.output)
    print(f"Maker review payload: {result['sample_maker_review_payload_json']}")
    print(f"Final garment measurements: {result['sample_final_garment_measurements_json']}")
    print(f"Maker review summary: {result['maker_review_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
