from __future__ import annotations

from app.schemas.design_agent_virtual_fitting import (
    BETA_FITTING_DISCLAIMER,
    BodyProfileSnapshot,
    FittingPreviewKind,
    GeneratedDesignOption,
    RenderStatus,
    RendererProviderMode,
    VirtualFittingRequest,
)
from app.services.design_agent_virtual_fitting import VirtualFittingEngine
from app.services.fitting_asset_pipeline import Beta2DConceptRendererProvider, FittingAssetPipelineService


def test_beta_2d_concept_renderer_returns_deterministic_asset_metadata() -> None:
    service = FittingAssetPipelineService(Beta2DConceptRendererProvider())
    result = service.generate_assets(
        design_session_id="session_a",
        fitting_result_id="fitting_a",
        fitting_request=VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
            preview_kind=FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT,
        ),
        body_profile=_body_profile(),
        design_option=_design_option(),
    )

    assert result.renderer_provider is RendererProviderMode.BETA_2D_CONCEPT
    assert result.render_status is RenderStatus.GENERATED
    assert result.preview_kind is FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT
    assert result.beta_disclaimer == BETA_FITTING_DISCLAIMER
    assert result.assets[0].asset_uri == "concept://fitting-preview/session_a/option_a"
    assert result.assets[0].thumbnail_url == "concept://fitting-preview/session_a/option_a/thumbnail"
    assert result.assets[0].quality_metadata["uses_real_scan_media"] is False
    assert result.assets[0].quality_metadata["validated_renderer"] is False
    assert result.assets[0].quality_metadata["concept_metadata"]["garment_type"] == "dress"
    assert result.assets[0].quality_metadata["concept_metadata"]["primary_color"] == "emerald"
    assert result.asset_manifest is not None
    assert result.asset_manifest.asset_ids == [result.assets[0].asset_id]
    assert result.asset_manifest.preview_kind is FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT
    assert "Beta 2D concept preview only." in result.warnings
    assert "Not production-grade cloth simulation." in result.warnings
    assert "Maker review required." in result.warnings


def test_beta_2d_concept_renderer_does_not_require_real_scan_photo_fields() -> None:
    service = FittingAssetPipelineService(Beta2DConceptRendererProvider())

    result = service.generate_assets(
        design_session_id="session_a",
        fitting_result_id="fitting_a",
        fitting_request=VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
        ),
        body_profile=_body_profile_without_scan_photo_refs(),
        design_option=_design_option(),
    )

    assert result.assets
    assert result.assets[0].mannequin_ref is None
    assert result.assets[0].quality_metadata["uses_real_scan_media"] is False
    assert result.assets[0].quality_metadata["concept_metadata"]["mannequin_ref"] is None


def test_virtual_fitting_result_can_use_beta_2d_concept_renderer() -> None:
    result = VirtualFittingEngine().preview(
        _design_option(),
        _body_profile(),
        VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
            preview_kind=FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT,
        ),
        design_session_id="session_a",
    )

    assert result.fitting_preview_assets
    assert result.preview_asset_references == ["concept://fitting-preview/session_a/option_a"]
    assert result.asset_manifest is not None
    assert result.asset_manifest.renderer_provider is RendererProviderMode.BETA_2D_CONCEPT
    assert result.confidence_metadata["renderer_provider"] == "beta_2d_concept"
    assert result.confidence_metadata["render_status"] == "generated"
    assert "Beta 2D concept preview only." in result.caution_notes
    assert "Not production-grade cloth simulation." in result.caution_notes
    assert "Maker review required." in result.caution_notes


def _body_profile() -> BodyProfileSnapshot:
    return BodyProfileSnapshot(
        body_profile_snapshot_id="body_profile_a",
        measurement_result_id="measurement_a",
        scan_id="scan_a",
        mannequin_reference_id="mannequin://body-ai/measurement_a",
        morphology_tags=["body_ai_snapshot", "defined_upper_torso"],
        measurement_summary_cm={"chest": 100.0, "waist": 80.0, "hip": 104.0},
        warnings=["Design/fitting workflow is beta/internal-preview only."],
    )


def _body_profile_without_scan_photo_refs() -> BodyProfileSnapshot:
    return BodyProfileSnapshot(
        body_profile_snapshot_id="body_profile_a",
        measurement_result_id="measurement_a",
        scan_id=None,
        mannequin_reference_id=None,
        morphology_tags=["body_ai_snapshot"],
        measurement_summary_cm={"chest": 100.0, "waist": 80.0},
        warnings=["Design/fitting workflow is beta/internal-preview only."],
    )


def _design_option() -> GeneratedDesignOption:
    return GeneratedDesignOption(
        design_option_id="option_a",
        title="Modern Dress Concept",
        style_description="Structured deterministic test option.",
        garment_details=["Garment type: dress"],
        color_direction="emerald, ivory",
        fit_direction="balanced custom fit",
        asset_references=["synthetic://design-options/dress/1"],
    )
