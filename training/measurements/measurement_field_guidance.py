from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.measurement_result_schema import write_json

DEFAULT_OUTPUT_DIR = "artifacts/phase_4l_measurement_field_guidance"
CUSTOMER_GUIDANCE_JSON = "customer_field_guidance.json"
MAKER_GUIDANCE_JSON = "maker_field_guidance.json"
ADMIN_GUIDANCE_JSON = "admin_field_guidance.json"
SUMMARY_MD = "measurement_field_guidance_summary.md"

EASE_FIELD_KEYS = {
    "ease_cm",
    "allowance_cm",
    "maker_ease_allowance_cm",
    "garment_allowance_cm",
    "wearing_ease_cm",
    "design_ease_cm",
}


class FieldGuidanceError(ValueError):
    """Raised when field guidance cannot be found or role access is invalid."""


class RoleVisibility(str, Enum):
    CUSTOMER = "customer"
    MAKER = "maker"
    ADMIN = "admin"
    INTERNAL_ONLY = "internal_only"


class EditableBy(str, Enum):
    CUSTOMER = "customer"
    MAKER = "maker"
    ADMIN = "admin"
    SYSTEM = "system"
    NONE = "none"


@dataclass(frozen=True)
class FieldGuidance:
    field_key: str
    label: str
    role_visibility: tuple[RoleVisibility, ...]
    placeholder: str
    helper_text: str
    info_icon_text: str
    warning_text: str | None
    required: bool
    editable_by: EditableBy
    unit: str | None
    example_value: str | None = None
    validation_hint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role_visibility"] = [role.value for role in self.role_visibility]
        payload["editable_by"] = self.editable_by.value
        return payload


def get_field_guidance(field_key: str, role: str | RoleVisibility) -> dict[str, Any]:
    role_value = coerce_role(role)
    guidance = GUIDANCE_BY_KEY.get(field_key)
    if guidance is None:
        raise FieldGuidanceError(f"Unknown measurement guidance field: {field_key}")
    if role_value not in guidance.role_visibility:
        raise FieldGuidanceError(f"Field '{field_key}' is not visible to role '{role_value.value}'")
    return guidance.to_payload()


def list_guidance_for_role(role: str | RoleVisibility) -> list[dict[str, Any]]:
    role_value = coerce_role(role)
    return [
        guidance.to_payload()
        for key, guidance in sorted(GUIDANCE_BY_KEY.items())
        if role_value in guidance.role_visibility
    ]


def validate_role_visibility(field_key: str, role: str | RoleVisibility) -> bool:
    get_field_guidance(field_key, role)
    return True


def export_guidance_json(role: str | RoleVisibility) -> str:
    payload = {
        "role": coerce_role(role).value,
        "fields": list_guidance_for_role(role),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def reject_customer_ease_fields(payload: Any) -> None:
    if isinstance(payload, dict):
        forbidden = sorted(EASE_FIELD_KEYS & set(payload))
        if forbidden:
            raise FieldGuidanceError(f"Customer payload must not include maker-only ease/allowance fields: {', '.join(forbidden)}")
        for value in payload.values():
            reject_customer_ease_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            reject_customer_ease_fields(item)


def export_sample_guidance(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "customer_field_guidance_json": output_path / CUSTOMER_GUIDANCE_JSON,
        "maker_field_guidance_json": output_path / MAKER_GUIDANCE_JSON,
        "admin_field_guidance_json": output_path / ADMIN_GUIDANCE_JSON,
        "measurement_field_guidance_summary_md": output_path / SUMMARY_MD,
    }
    for role, key in (
        (RoleVisibility.CUSTOMER, "customer_field_guidance_json"),
        (RoleVisibility.MAKER, "maker_field_guidance_json"),
        (RoleVisibility.ADMIN, "admin_field_guidance_json"),
    ):
        paths[key].write_text(export_guidance_json(role), encoding="utf-8")
    paths["measurement_field_guidance_summary_md"].write_text(format_guidance_summary(), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def format_guidance_summary() -> str:
    customer_count = len(list_guidance_for_role(RoleVisibility.CUSTOMER))
    maker_count = len(list_guidance_for_role(RoleVisibility.MAKER))
    admin_count = len(list_guidance_for_role(RoleVisibility.ADMIN))
    return "\n".join(
        [
            "# Phase 4L Measurement Field Guidance",
            "",
            f"Customer fields: `{customer_count}`",
            f"Maker fields: `{maker_count}`",
            f"Admin fields: `{admin_count}`",
            "",
            "Customer guidance describes body measurements and fit preference only.",
            "Maker guidance includes maker-only ease/allowance and final garment measurements.",
            "Admin guidance explains model, calibration, uncertainty, and validation caveats.",
            "",
        ]
    )


def coerce_role(role: str | RoleVisibility) -> RoleVisibility:
    if isinstance(role, RoleVisibility):
        return role
    try:
        return RoleVisibility(str(role))
    except ValueError as exc:
        raise FieldGuidanceError(f"Unknown guidance role: {role}") from exc


def cm_field(
    field_key: str,
    label: str,
    roles: tuple[RoleVisibility, ...],
    helper_text: str,
    info_icon_text: str,
    editable_by: EditableBy,
    required: bool = False,
    warning_text: str | None = None,
    placeholder: str | None = None,
    example_value: str | None = None,
    validation_hint: str | None = "Enter a positive body measurement in centimeters.",
) -> FieldGuidance:
    return FieldGuidance(
        field_key=field_key,
        label=label,
        role_visibility=roles,
        placeholder=placeholder or "cm",
        helper_text=helper_text,
        info_icon_text=info_icon_text,
        warning_text=warning_text,
        required=required,
        editable_by=editable_by,
        unit="cm",
        example_value=example_value,
        validation_hint=validation_hint,
    )


def text_field(
    field_key: str,
    label: str,
    roles: tuple[RoleVisibility, ...],
    helper_text: str,
    info_icon_text: str,
    editable_by: EditableBy,
    required: bool = False,
    warning_text: str | None = None,
    placeholder: str = "",
    example_value: str | None = None,
    validation_hint: str | None = None,
) -> FieldGuidance:
    return FieldGuidance(
        field_key=field_key,
        label=label,
        role_visibility=roles,
        placeholder=placeholder,
        helper_text=helper_text,
        info_icon_text=info_icon_text,
        warning_text=warning_text,
        required=required,
        editable_by=editable_by,
        unit=None,
        example_value=example_value,
        validation_hint=validation_hint,
    )


CUSTOMER_ROLES = (RoleVisibility.CUSTOMER, RoleVisibility.ADMIN)
MAKER_ROLES = (RoleVisibility.MAKER, RoleVisibility.ADMIN)
ADMIN_ROLES = (RoleVisibility.ADMIN, RoleVisibility.INTERNAL_ONLY)

GUIDANCE: list[FieldGuidance] = [
    cm_field(
        "height_cm",
        "Height",
        CUSTOMER_ROLES,
        "Enter your real body height. The AI should not guess height from photos.",
        "Height is required as user input because photo scale alone is not reliable enough for production decisions.",
        EditableBy.CUSTOMER,
        required=True,
        placeholder="Your height in cm",
        example_value="172",
        validation_hint="Required. Enter your standing body height in centimeters.",
    ),
    cm_field(
        "chest_cm",
        "Chest",
        CUSTOMER_ROLES,
        "Review or enter your body chest measurement. Do not add extra room.",
        "This is a body measurement around the chest, not a garment chest width and not a final fit allowance.",
        EditableBy.CUSTOMER,
        example_value="102",
    ),
    cm_field(
        "waist_cm",
        "Waist",
        CUSTOMER_ROLES,
        "Review or enter your body waist measurement. Do not add ease.",
        "This is your body waist measurement. Makers add any garment room later.",
        EditableBy.CUSTOMER,
        example_value="84",
    ),
    cm_field(
        "hip_cm",
        "Hip",
        CUSTOMER_ROLES,
        "Review or enter your body hip measurement at the fullest point.",
        "This is a body measurement, not the finished garment measurement.",
        EditableBy.CUSTOMER,
        example_value="106",
    ),
    cm_field(
        "thigh_cm",
        "Thigh",
        CUSTOMER_ROLES,
        "Review or enter your body thigh measurement. Confirmation is recommended.",
        "Thigh AI intervals were undercovered in Phase 4F, so treat this estimate with extra caution.",
        EditableBy.CUSTOMER,
        warning_text="Thigh estimates have documented synthetic undercoverage risk and may need manual confirmation.",
        example_value="62",
    ),
    cm_field(
        "shoulder_cm",
        "Shoulder",
        CUSTOMER_ROLES,
        "Enter or confirm your body shoulder measurement if requested.",
        "Shoulder may need manual confirmation because the current residual pipeline may not provide this AI target.",
        EditableBy.CUSTOMER,
        example_value="45",
    ),
    cm_field(
        "calf_cm",
        "Calf",
        CUSTOMER_ROLES,
        "Enter or confirm your body calf measurement if requested.",
        "Calf may need manual confirmation before maker review.",
        EditableBy.CUSTOMER,
        example_value="39",
    ),
    cm_field(
        "inseam_cm",
        "Inseam",
        CUSTOMER_ROLES,
        "Measure with tape or follow the app's landmark instructions.",
        "Inseam requires a manual or landmark-based measurement; the AI should not finalize it from photos alone.",
        EditableBy.CUSTOMER,
        required=True,
        example_value="78",
    ),
    cm_field(
        "sleeve_cm",
        "Sleeve",
        CUSTOMER_ROLES,
        "Measure sleeve length manually or follow landmark instructions.",
        "Sleeve length needs a manual/landmark strategy and should not be treated as a direct AI-only estimate.",
        EditableBy.CUSTOMER,
        required=True,
        example_value="61",
    ),
    cm_field(
        "neck_cm",
        "Neck",
        CUSTOMER_ROLES,
        "Measure neck manually with tape if requested.",
        "Neck is a small measurement and currently requires manual or maker confirmation.",
        EditableBy.CUSTOMER,
        required=True,
        example_value="38",
    ),
    text_field(
        "fit_preference",
        "Fit Preference",
        CUSTOMER_ROLES,
        "Choose your preferred wearing feel: snug, regular, relaxed, loose, or add a custom note.",
        "Fit preference is a comfort/style preference, not a measurement and not maker ease.",
        EditableBy.CUSTOMER,
        required=True,
        placeholder="regular",
        example_value="regular",
        validation_hint="Choose one of: snug, regular, relaxed, loose, custom_note.",
    ),
    text_field(
        "customer_notes",
        "Customer Notes",
        CUSTOMER_ROLES,
        "Add body-measurement notes or fit concerns for maker review.",
        "Use notes for context, not for ease/allowance values.",
        EditableBy.CUSTOMER,
        placeholder="Optional note for the maker",
    ),
    cm_field(
        "customer_confirmed_cm",
        "Confirmed Body Measurement",
        CUSTOMER_ROLES,
        "Confirm the body measurement shown by AI or measurement review.",
        "This confirms a body measurement. Do not add extra room for comfort.",
        EditableBy.CUSTOMER,
    ),
    cm_field(
        "customer_manual_cm",
        "Manual Body Measurement",
        CUSTOMER_ROLES,
        "Enter a tape-measured body value when the app requires manual input.",
        "Manual values should be body measurements only, without garment ease.",
        EditableBy.CUSTOMER,
    ),
    cm_field(
        "maker_verified_body_cm",
        "Maker Verified Body Measurement",
        MAKER_ROLES,
        "Enter the maker-confirmed body measurement after review.",
        "This is the maker's verified body measurement before garment ease is added.",
        EditableBy.MAKER,
        example_value="105.2",
    ),
    cm_field(
        "maker_ease_allowance_cm",
        "Maker Ease / Allowance",
        MAKER_ROLES,
        "Enter the extra room added for fit, design, comfort, fabric, and construction.",
        "Maker ease/allowance is maker-only and is not seam allowance.",
        EditableBy.MAKER,
        required=True,
        warning_text="Do not expose this field in customer-facing workflows.",
        example_value="8",
        validation_hint="Maker-only. This is added to the selected body measurement.",
    ),
    cm_field(
        "final_garment_cm",
        "Final Garment Measurement",
        MAKER_ROLES,
        "Review the finished garment measurement calculated for production.",
        "Final garment measurement = selected body measurement + maker ease/allowance.",
        EditableBy.SYSTEM,
        required=True,
        example_value="113.2",
    ),
    text_field(
        "selected_body_measurement_source",
        "Selected Body Measurement Source",
        MAKER_ROLES,
        "Shows whether the body value came from maker verification, customer confirmation, manual input, or AI.",
        "Use this source to audit why a garment measurement was calculated from a specific body value.",
        EditableBy.SYSTEM,
        placeholder="customer_confirmed_cm",
    ),
    text_field(
        "maker_notes",
        "Maker Notes",
        MAKER_ROLES,
        "Add production notes, review notes, or reasons for measurement changes.",
        "Maker notes support audit and dispute review.",
        EditableBy.MAKER,
        placeholder="Production note",
    ),
    text_field(
        "production_status",
        "Production Status",
        MAKER_ROLES,
        "Track draft, review, ready, locked, or revision status.",
        "Locked production measurements cannot be changed without a revision path.",
        EditableBy.MAKER,
        required=True,
        placeholder="awaiting_maker_review",
    ),
    text_field(
        "revision_reason",
        "Revision Reason",
        MAKER_ROLES,
        "Explain why locked production measurements need revision.",
        "A revision reason should be recorded before changing locked garment measurements.",
        EditableBy.MAKER,
        placeholder="Customer requested correction",
    ),
    text_field(
        "locked_for_production",
        "Locked For Production",
        MAKER_ROLES,
        "Indicates whether final garment measurements are locked.",
        "Once locked, final garment measurements and ease cannot change without revision.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "locked_at",
        "Locked At",
        MAKER_ROLES,
        "Timestamp when production measurements were locked.",
        "This timestamp supports audit and dispute review.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "locked_by_maker_id",
        "Locked By Maker",
        MAKER_ROLES,
        "Maker ID that locked production measurements.",
        "This identifies who accepted the final garment measurements.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "measurement_snapshot_id",
        "Measurement Snapshot ID",
        ADMIN_ROLES,
        "Identifier for the persisted Body AI measurement snapshot.",
        "Use this to trace customer confirmation and maker review back to the original measurement result.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "model_version",
        "Model Version",
        ADMIN_ROLES,
        "Version of the model or deterministic estimator that produced the measurement.",
        "Model version is required for audit, rollback, and reproducibility.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "pipeline_version",
        "Pipeline Version",
        ADMIN_ROLES,
        "Version of the measurement pipeline and contract.",
        "Pipeline version explains which inference, confidence, uncertainty, and schema rules were used.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "calibration_version",
        "Calibration Version",
        ADMIN_ROLES,
        "Version of the calibration or uncertainty interval policy.",
        "Calibration version supports later comparison with real-world validation updates.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "synthetic_calibrated_only",
        "Synthetic Calibrated Only",
        ADMIN_ROLES,
        "Marks results validated on synthetic calibrated labels rather than real tape measurements.",
        "This means the result is not real-world production validation.",
        EditableBy.SYSTEM,
        warning_text="Do not treat synthetic-calibrated results as real-world validated.",
    ),
    text_field(
        "real_world_validated",
        "Real-World Validated",
        ADMIN_ROLES,
        "Shows whether this measurement system has been validated against real tape-measured people.",
        "This must remain false until real-world validation exists.",
        EditableBy.SYSTEM,
        warning_text="Must remain false until real-world validation is completed.",
    ),
    text_field(
        "quality_flags",
        "Quality Flags",
        ADMIN_ROLES,
        "Flags that indicate calibration risk, unavailable measurements, or geometry issues.",
        "Quality flags should influence manual confirmation, maker review, or retake decisions.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "confidence_tier",
        "Confidence Tier",
        ADMIN_ROLES,
        "Product-risk/action tier for measurement handling.",
        "Confidence tier is not an accuracy guarantee; it guides confirmation or retake behavior.",
        EditableBy.SYSTEM,
    ),
    cm_field(
        "estimated_error_cm",
        "Estimated Error",
        ADMIN_ROLES,
        "Estimated error range used for product risk handling.",
        "This is calibrated uncertainty from synthetic validation, not guaranteed real-world accuracy.",
        EditableBy.SYSTEM,
    ),
    cm_field(
        "interval_low_cm",
        "Interval Low",
        ADMIN_ROLES,
        "Lower bound of the estimated measurement interval.",
        "Interval values represent estimated uncertainty range around the AI estimate.",
        EditableBy.SYSTEM,
    ),
    cm_field(
        "interval_high_cm",
        "Interval High",
        ADMIN_ROLES,
        "Upper bound of the estimated measurement interval.",
        "Interval values represent estimated uncertainty range around the AI estimate.",
        EditableBy.SYSTEM,
    ),
    text_field(
        "product_action",
        "Product Action",
        ADMIN_ROLES,
        "Recommended action such as accept, confirm, retake, or maker review.",
        "Product action should drive workflow handling, not silently finalize measurements.",
        EditableBy.SYSTEM,
    ),
]

GUIDANCE_BY_KEY = {guidance.field_key: guidance for guidance in GUIDANCE}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export measurement field guidance metadata.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = export_sample_guidance(args.output)
    print(f"Customer guidance: {paths['customer_field_guidance_json']}")
    print(f"Maker guidance: {paths['maker_field_guidance_json']}")
    print(f"Admin guidance: {paths['admin_field_guidance_json']}")
    print(f"Summary: {paths['measurement_field_guidance_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
