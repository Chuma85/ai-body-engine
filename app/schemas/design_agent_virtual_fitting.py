from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


BETA_FITTING_DISCLAIMER = (
    "Virtual fitting preview is beta/internal-preview only and is not a production accuracy claim. "
    "Maker review and approved measurement handling remain required before production."
)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


class DesignSessionStatus(str, Enum):
    CREATED = "created"
    OPTIONS_GENERATED = "options_generated"
    REFINED = "refined"
    FITTING_PREVIEW_READY = "fitting_preview_ready"
    APPROVED = "approved"


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REVISION_REQUESTED = "revision_requested"


class PreviewStatus(str, Enum):
    DEMO_SYNTHETIC_PREVIEW = "demo_synthetic_preview"
    PENDING_RENDER_BACKEND = "pending_render_backend"


class DesignPreferenceInput(BaseModel):
    garment_type: str = Field(..., min_length=1)
    color_palette: list[str] = Field(default_factory=list)
    style_direction: str | None = None
    fit_preference: str | None = None
    occasion: str | None = None
    fabric_preference: str | None = None
    inspiration_notes: str | None = None
    maker_notes: str | None = None


class BodyProfileSnapshot(BaseModel):
    body_profile_snapshot_id: str
    measurement_result_id: str
    scan_id: str | None = None
    mannequin_reference_id: str | None = None
    morphology_tags: list[str] = Field(default_factory=list)
    measurement_summary_cm: dict[str, float] = Field(default_factory=dict)
    source: str = "measurement_result_snapshot"
    synthetic_calibrated_only: bool = True
    real_world_validated: bool = False
    warnings: list[str] = Field(default_factory=list)


class GeneratedDesignOption(BaseModel):
    design_option_id: str
    title: str
    style_description: str
    garment_details: list[str] = Field(default_factory=list)
    color_direction: str
    fit_direction: str
    asset_references: list[str] = Field(default_factory=list)
    suitability_notes: list[str] = Field(default_factory=list)
    confidence_metadata: dict[str, Any] = Field(default_factory=dict)
    warning_metadata: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class DesignRefinementRequest(BaseModel):
    design_option_id: str | None = None
    refinement_prompt: str = Field(..., min_length=1)
    updated_preferences: DesignPreferenceInput | None = None


class VirtualFittingRequest(BaseModel):
    design_option_id: str
    preview_mode: str = "synthetic_demo"
    maker_review_required: bool = True


class VirtualFittingResult(BaseModel):
    fitting_result_id: str
    design_option_id: str
    preview_status: PreviewStatus = PreviewStatus.DEMO_SYNTHETIC_PREVIEW
    preview_asset_references: list[str] = Field(default_factory=list)
    fit_summary: str
    tension_notes: list[str] = Field(default_factory=list)
    looseness_notes: list[str] = Field(default_factory=list)
    caution_notes: list[str] = Field(default_factory=list)
    confidence_metadata: dict[str, Any] = Field(default_factory=dict)
    beta_preview_disclaimer: str = BETA_FITTING_DISCLAIMER
    maker_review_recommendation: str = "Maker review required before production pattern or cutting decisions."
    created_at: datetime = Field(default_factory=utc_now)


class DesignSession(BaseModel):
    design_session_id: str
    user_id: str
    session_reference_id: str | None = None
    order_id: str | None = None
    scan_id: str | None = None
    measurement_result_id: str
    body_profile_snapshot: BodyProfileSnapshot
    preferences: DesignPreferenceInput
    generated_options: list[GeneratedDesignOption] = Field(default_factory=list)
    fitting_results: list[VirtualFittingResult] = Field(default_factory=list)
    selected_design_option_id: str | None = None
    approval_state: ApprovalState = ApprovalState.PENDING
    status: DesignSessionStatus = DesignSessionStatus.CREATED
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProductionBrief(BaseModel):
    production_brief_id: str
    design_session_id: str
    approved_design_option: GeneratedDesignOption
    preferences_summary: DesignPreferenceInput
    measurement_references: dict[str, str | None]
    body_profile_snapshot: BodyProfileSnapshot
    fit_notes: list[str] = Field(default_factory=list)
    maker_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=lambda: [BETA_FITTING_DISCLAIMER])
    approval_state: ApprovalState
    generated_at: datetime = Field(default_factory=utc_now)


class DesignSessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_reference_id: str | None = None
    order_id: str | None = None
    scan_id: str | None = None
    measurement_result_id: str | None = None
    measurement_result: dict[str, Any] | None = None
    body_profile_snapshot: BodyProfileSnapshot | None = None
    preferences: DesignPreferenceInput


class DesignGenerationRequest(BaseModel):
    option_count: int = Field(default=3, ge=1, le=5)
    prompt_context: str | None = None


class DesignApprovalRequest(BaseModel):
    design_option_id: str
    maker_production_notes: str | None = None

