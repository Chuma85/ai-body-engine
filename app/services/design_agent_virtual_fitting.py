from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.schemas.design_agent_virtual_fitting import (
    BETA_FITTING_DISCLAIMER,
    ApprovalState,
    BodyProfileSnapshot,
    DesignApprovalRequest,
    DesignGenerationRequest,
    DesignPreferenceInput,
    DesignRefinementRequest,
    DesignSession,
    DesignSessionCreateRequest,
    DesignSessionStatus,
    GeneratedDesignOption,
    ProductionBrief,
    RendererProviderMode,
    VirtualFittingRequest,
    VirtualFittingResult,
    utc_now,
)
from app.services.fitting_asset_pipeline import (
    DemoSyntheticRendererProvider,
    FittingAssetPipelineService,
    LocalPlaceholderRendererProvider,
    UnavailableRendererProvider,
)
from training.measurements.measurement_snapshot_store import (
    MeasurementSnapshotError,
    validate_measurement_result_payload,
)


class DesignSessionError(ValueError):
    """Raised when the design/fitting workflow receives invalid session state."""


class DesignStylingAgent:
    """Deterministic first-pass design agent contract for demo and test flows."""

    def generate_options(
        self,
        preferences: DesignPreferenceInput,
        body_profile: BodyProfileSnapshot,
        option_count: int = 3,
        prompt_context: str | None = None,
    ) -> list[GeneratedDesignOption]:
        palette = ", ".join(preferences.color_palette) if preferences.color_palette else "maker-selected palette"
        style = preferences.style_direction or "clean custom styling"
        fit = preferences.fit_preference or "balanced custom fit"
        fabric = preferences.fabric_preference or "fabric to be confirmed by maker"
        occasion = preferences.occasion or "general wear"
        morphology_note = suitability_note(body_profile)
        options: list[GeneratedDesignOption] = []
        for index in range(option_count):
            variant = index + 1
            options.append(
                GeneratedDesignOption(
                    design_option_id=f"design_option_{uuid4().hex}",
                    title=f"{style.title()} {preferences.garment_type.title()} Concept {variant}",
                    style_description=(
                        f"A {preferences.garment_type} concept for {occasion}, using {style} cues, "
                        f"{palette}, and {fabric}."
                    ),
                    garment_details=[
                        f"Garment type: {preferences.garment_type}",
                        f"Variant emphasis: {variant_label(variant)}",
                        f"Fabric direction: {fabric}",
                    ],
                    color_direction=palette,
                    fit_direction=fit,
                    asset_references=[f"synthetic://design-options/{preferences.garment_type}/{variant}"],
                    suitability_notes=[morphology_note, "Confirm ease, closures, and construction details with the maker."],
                    confidence_metadata={
                        "mode": "deterministic_demo_design_agent",
                        "body_profile_snapshot_id": body_profile.body_profile_snapshot_id,
                        "prompt_context_used": bool(prompt_context),
                    },
                    warning_metadata=list(body_profile.warnings),
                )
            )
        return options

    def refine_option(
        self,
        base_option: GeneratedDesignOption | None,
        preferences: DesignPreferenceInput,
        body_profile: BodyProfileSnapshot,
        request: DesignRefinementRequest,
    ) -> GeneratedDesignOption:
        source_title = base_option.title if base_option else f"{preferences.garment_type.title()} Concept"
        updated = request.updated_preferences or preferences
        palette = ", ".join(updated.color_palette) if updated.color_palette else "maker-selected palette"
        return GeneratedDesignOption(
            design_option_id=f"design_option_{uuid4().hex}",
            title=f"{source_title} Refinement",
            style_description=f"Refined around: {request.refinement_prompt}",
            garment_details=[
                *(base_option.garment_details if base_option else [f"Garment type: {updated.garment_type}"]),
                f"Refinement request: {request.refinement_prompt}",
            ],
            color_direction=palette,
            fit_direction=updated.fit_preference or preferences.fit_preference or "balanced custom fit",
            asset_references=[f"synthetic://design-options/refinements/{uuid4().hex}"],
            suitability_notes=[
                suitability_note(body_profile),
                "Refinement remains a structured concept until rendered or maker-reviewed.",
            ],
            confidence_metadata={
                "mode": "deterministic_demo_refinement_agent",
                "source_design_option_id": request.design_option_id,
                "body_profile_snapshot_id": body_profile.body_profile_snapshot_id,
            },
            warning_metadata=list(body_profile.warnings),
        )


class VirtualFittingEngine:
    """Service boundary for fitting previews; currently returns safe synthetic contracts."""

    def __init__(self, asset_pipeline: FittingAssetPipelineService | None = None) -> None:
        self.asset_pipeline = asset_pipeline or FittingAssetPipelineService()

    def preview(
        self,
        design_option: GeneratedDesignOption,
        body_profile: BodyProfileSnapshot,
        request: VirtualFittingRequest,
        design_session_id: str | None = None,
    ) -> VirtualFittingResult:
        warnings = list(body_profile.warnings)
        if body_profile.synthetic_calibrated_only:
            warnings.append("Body profile is synthetic-calibrated/internal-preview only.")
        fitting_result_id = f"fitting_result_{uuid4().hex}"
        asset_pipeline = pipeline_for_request(request, self.asset_pipeline)
        asset_result = asset_pipeline.generate_assets(
            design_session_id=design_session_id,
            fitting_result_id=fitting_result_id,
            fitting_request=request,
            body_profile=body_profile,
            design_option=design_option,
        )
        preview_asset_references = [
            asset.asset_uri or asset.image_url or asset.asset_id for asset in asset_result.assets
        ] or [f"synthetic://virtual-fitting/{body_profile.body_profile_snapshot_id}/{design_option.design_option_id}"]
        return VirtualFittingResult(
            fitting_result_id=fitting_result_id,
            design_option_id=design_option.design_option_id,
            preview_asset_references=preview_asset_references,
            fitting_preview_assets=asset_result.assets,
            asset_manifest=asset_result.asset_manifest,
            fit_summary=(
                f"Demo preview prepared for {design_option.title} against body profile "
                f"{body_profile.body_profile_snapshot_id}."
            ),
            tension_notes=[
                "No production cloth simulation has been run in this phase.",
                "Check high-movement areas manually during maker review.",
            ],
            looseness_notes=["Ease and drape are advisory placeholders until a validated renderer is integrated."],
            caution_notes=list(dict.fromkeys([BETA_FITTING_DISCLAIMER, *warnings, *asset_result.warnings])),
            confidence_metadata={
                "mode": request.preview_mode,
                "maker_review_required": request.maker_review_required,
                "real_world_validated": body_profile.real_world_validated,
                "renderer_provider": asset_result.renderer_provider.value,
                "render_status": asset_result.render_status.value,
            },
        )


def pipeline_for_request(
    request: VirtualFittingRequest,
    default_pipeline: FittingAssetPipelineService,
) -> FittingAssetPipelineService:
    if request.renderer_provider is RendererProviderMode.DEMO_SYNTHETIC:
        return default_pipeline
    if request.renderer_provider is RendererProviderMode.UNAVAILABLE:
        return FittingAssetPipelineService(UnavailableRendererProvider())
    if request.renderer_provider is RendererProviderMode.LOCAL_PLACEHOLDER:
        return FittingAssetPipelineService(LocalPlaceholderRendererProvider())
    return FittingAssetPipelineService(UnavailableRendererProvider())


class DesignSessionService:
    def __init__(
        self,
        design_agent: DesignStylingAgent | None = None,
        fitting_engine: VirtualFittingEngine | None = None,
    ) -> None:
        self.design_agent = design_agent or DesignStylingAgent()
        self.fitting_engine = fitting_engine or VirtualFittingEngine()
        self._sessions: dict[str, DesignSession] = {}
        self._production_notes: dict[str, str] = {}

    def create_session(self, request: DesignSessionCreateRequest) -> DesignSession:
        measurement_result_id = resolve_measurement_result_id(request)
        body_profile = request.body_profile_snapshot or body_profile_from_measurement(
            measurement_result_id=measurement_result_id,
            scan_id=request.scan_id,
            measurement_result=request.measurement_result,
        )
        warnings = workflow_warnings(body_profile)
        session = DesignSession(
            design_session_id=f"design_session_{uuid4().hex}",
            user_id=request.user_id,
            session_reference_id=request.session_reference_id,
            order_id=request.order_id,
            scan_id=request.scan_id or body_profile.scan_id,
            measurement_result_id=measurement_result_id,
            body_profile_snapshot=body_profile,
            preferences=request.preferences,
            warnings=warnings,
        )
        self._sessions[session.design_session_id] = session
        return session

    def get_session(self, design_session_id: str) -> DesignSession:
        try:
            return self._sessions[design_session_id]
        except KeyError as exc:
            raise DesignSessionError(f"Unknown design session: {design_session_id}") from exc

    def generate_options(self, design_session_id: str, request: DesignGenerationRequest) -> DesignSession:
        session = self.get_session(design_session_id)
        options = self.design_agent.generate_options(
            preferences=session.preferences,
            body_profile=session.body_profile_snapshot,
            option_count=request.option_count,
            prompt_context=request.prompt_context,
        )
        updated = session_copy(
            session,
            generated_options=[*session.generated_options, *options],
            status=DesignSessionStatus.OPTIONS_GENERATED,
        )
        return self._save(updated)

    def refine_options(self, design_session_id: str, request: DesignRefinementRequest) -> DesignSession:
        session = self.get_session(design_session_id)
        base_option = find_option(session, request.design_option_id) if request.design_option_id else None
        refined = self.design_agent.refine_option(
            base_option=base_option,
            preferences=session.preferences,
            body_profile=session.body_profile_snapshot,
            request=request,
        )
        updated_preferences = request.updated_preferences or session.preferences
        updated = session_copy(
            session,
            preferences=updated_preferences,
            generated_options=[*session.generated_options, refined],
            status=DesignSessionStatus.REFINED,
        )
        return self._save(updated)

    def create_fitting_preview(self, design_session_id: str, request: VirtualFittingRequest) -> DesignSession:
        session = self.get_session(design_session_id)
        option = find_option(session, request.design_option_id)
        result = self.fitting_engine.preview(option, session.body_profile_snapshot, request, session.design_session_id)
        updated = session_copy(
            session,
            selected_design_option_id=option.design_option_id,
            fitting_results=[*session.fitting_results, result],
            status=DesignSessionStatus.FITTING_PREVIEW_READY,
        )
        return self._save(updated)

    def approve_design(self, design_session_id: str, request: DesignApprovalRequest) -> DesignSession:
        session = self.get_session(design_session_id)
        find_option(session, request.design_option_id)
        if request.maker_production_notes:
            self._production_notes[design_session_id] = request.maker_production_notes
        updated = session_copy(
            session,
            selected_design_option_id=request.design_option_id,
            approval_state=ApprovalState.APPROVED,
            status=DesignSessionStatus.APPROVED,
        )
        return self._save(updated)

    def production_brief(self, design_session_id: str) -> ProductionBrief:
        session = self.get_session(design_session_id)
        if session.approval_state != ApprovalState.APPROVED or session.selected_design_option_id is None:
            raise DesignSessionError("Production brief requires an approved design option.")
        option = find_option(session, session.selected_design_option_id)
        fitting = latest_fitting_for_option(session, option.design_option_id)
        fit_notes = []
        if fitting:
            fit_notes.extend([fitting.fit_summary, *fitting.tension_notes, *fitting.looseness_notes])
        return ProductionBrief(
            production_brief_id=f"production_brief_{uuid4().hex}",
            design_session_id=session.design_session_id,
            approved_design_option=option,
            preferences_summary=session.preferences,
            measurement_references={
                "measurement_result_id": session.measurement_result_id,
                "scan_id": session.scan_id,
                "body_profile_snapshot_id": session.body_profile_snapshot.body_profile_snapshot_id,
                "mannequin_reference_id": session.body_profile_snapshot.mannequin_reference_id,
            },
            body_profile_snapshot=session.body_profile_snapshot,
            fit_notes=fit_notes,
            maker_notes=[
                note
                for note in [
                    session.preferences.maker_notes,
                    self._production_notes.get(design_session_id),
                    "Confirm measurements, ease, fabric behavior, closures, and construction before production.",
                ]
                if note
            ],
            warnings=[*session.warnings, *(fitting.caution_notes if fitting else [])],
            approval_state=session.approval_state,
        )

    def _save(self, session: DesignSession) -> DesignSession:
        self._sessions[session.design_session_id] = session
        return session


def resolve_measurement_result_id(request: DesignSessionCreateRequest) -> str:
    if request.measurement_result_id:
        return request.measurement_result_id
    if request.body_profile_snapshot:
        return request.body_profile_snapshot.measurement_result_id
    if request.measurement_result:
        return str(request.measurement_result.get("result_id") or request.measurement_result.get("measurement_result_id") or "")
    raise DesignSessionError("measurement_result_id, body_profile_snapshot, or measurement_result is required.")


def body_profile_from_measurement(
    measurement_result_id: str,
    scan_id: str | None,
    measurement_result: dict[str, Any] | None,
) -> BodyProfileSnapshot:
    if not measurement_result_id:
        raise DesignSessionError("A non-empty measurement result reference is required.")
    measurement_summary: dict[str, float] = {}
    warnings = [
        "Design/fitting workflow is beta/internal-preview only.",
        "No live real-user scan usage is approved beyond current storage/privacy scope.",
    ]
    synthetic_calibrated_only = True
    real_world_validated = False
    if measurement_result:
        try:
            validate_measurement_result_payload(measurement_result)
        except MeasurementSnapshotError as exc:
            raise DesignSessionError(str(exc)) from exc
        metadata = measurement_result["metadata"]
        synthetic_calibrated_only = bool(metadata["synthetic_calibrated_only"])
        real_world_validated = bool(metadata["real_world_validated"])
        scan_id = scan_id or str(measurement_result.get("sample_id") or "")
        for target in measurement_result.get("targets", []):
            estimate = target.get("estimate_cm")
            if estimate is not None:
                measurement_summary[str(target["target"])] = round(float(estimate), 4)
        warnings.extend(measurement_result.get("caveats", []))
    morphology_tags = infer_morphology_tags(measurement_summary)
    return BodyProfileSnapshot(
        body_profile_snapshot_id=f"body_profile_snapshot_{uuid4().hex}",
        measurement_result_id=measurement_result_id,
        scan_id=scan_id,
        mannequin_reference_id=f"mannequin://body-ai/{measurement_result_id}",
        morphology_tags=morphology_tags,
        measurement_summary_cm=measurement_summary,
        synthetic_calibrated_only=synthetic_calibrated_only,
        real_world_validated=real_world_validated,
        warnings=warnings,
    )


def infer_morphology_tags(measurements: dict[str, float]) -> list[str]:
    tags = ["body_ai_snapshot"]
    chest = measurements.get("chest")
    waist = measurements.get("waist")
    hip = measurements.get("hip")
    if chest and waist and chest - waist >= 12:
        tags.append("defined_upper_torso")
    if hip and waist and hip - waist >= 12:
        tags.append("defined_hip_curve")
    if not measurements:
        tags.append("measurement_reference_only")
    return tags


def suitability_note(body_profile: BodyProfileSnapshot) -> str:
    tags = ", ".join(body_profile.morphology_tags) if body_profile.morphology_tags else "profile reference available"
    return f"Anchored to body profile {body_profile.body_profile_snapshot_id} ({tags})."


def variant_label(variant: int) -> str:
    labels = {
        1: "clean base silhouette",
        2: "more expressive styling details",
        3: "maker-friendly construction variant",
        4: "occasion-forward variation",
        5: "fabric-forward variation",
    }
    return labels.get(variant, "custom variation")


def workflow_warnings(body_profile: BodyProfileSnapshot) -> list[str]:
    warnings = list(body_profile.warnings)
    warnings.append(BETA_FITTING_DISCLAIMER)
    if body_profile.synthetic_calibrated_only:
        warnings.append("Body profile is synthetic-calibrated/internal-preview unless separately validated.")
    if body_profile.real_world_validated:
        raise DesignSessionError("Production-validated real user scan handling is not enabled for this phase.")
    return list(dict.fromkeys(warnings))


def find_option(session: DesignSession, design_option_id: str | None) -> GeneratedDesignOption:
    for option in session.generated_options:
        if option.design_option_id == design_option_id:
            return option
    raise DesignSessionError(f"Unknown design option for session {session.design_session_id}: {design_option_id}")


def latest_fitting_for_option(session: DesignSession, design_option_id: str) -> VirtualFittingResult | None:
    matching = [result for result in session.fitting_results if result.design_option_id == design_option_id]
    return matching[-1] if matching else None


def session_copy(session: DesignSession, **updates: Any) -> DesignSession:
    updates["updated_at"] = datetime.now(UTC).replace(microsecond=0)
    if hasattr(session, "model_copy"):
        return session.model_copy(update=updates)
    return session.copy(update=updates)
