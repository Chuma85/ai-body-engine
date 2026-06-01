from __future__ import annotations

from app.schemas.design_agent_virtual_fitting import (
    BETA_FITTING_DISCLAIMER,
    BodyProfileSnapshot,
    FittingPreviewAsset,
    FittingPreviewAssetType,
    FittingPreviewKind,
    GeneratedDesignOption,
    RenderStatus,
    RendererProviderMode,
    VirtualFittingRequest,
)
from app.services.design_agent_virtual_fitting import VirtualFittingEngine
from app.services.fitting_asset_pipeline import (
    DemoSyntheticRendererProvider,
    FittingAssetPipelineService,
    UnavailableRendererProvider,
)


def test_fitting_preview_asset_schema_validates_contract() -> None:
    asset = FittingPreviewAsset(
        asset_id="asset_a",
        asset_type=FittingPreviewAssetType.IMAGE,
        preview_kind=FittingPreviewKind.MANNEQUIN_DESIGN_CONCEPT,
        render_status=RenderStatus.DEMO_PLACEHOLDER,
        renderer_provider=RendererProviderMode.DEMO_SYNTHETIC,
        asset_uri="demo://fitting-preview/session_a/option_a",
        body_profile_ref="body_profile_a",
        mannequin_ref="mannequin://body-ai/result_a",
        design_option_ref="option_a",
        garment_layer_refs=["synthetic://design-options/dress/1"],
        warnings=["Beta preview only."],
    )

    assert asset.beta_disclaimer == BETA_FITTING_DISCLAIMER
    assert asset.asset_type is FittingPreviewAssetType.IMAGE
    assert asset.render_status is RenderStatus.DEMO_PLACEHOLDER


def test_demo_synthetic_renderer_returns_deterministic_safe_metadata() -> None:
    service = FittingAssetPipelineService(DemoSyntheticRendererProvider())
    result = service.generate_assets(
        design_session_id="session_a",
        fitting_result_id="fitting_a",
        fitting_request=VirtualFittingRequest(design_option_id="option_a"),
        body_profile=_body_profile(),
        design_option=_design_option(),
    )

    assert result.renderer_provider is RendererProviderMode.DEMO_SYNTHETIC
    assert result.render_status is RenderStatus.DEMO_PLACEHOLDER
    assert result.assets[0].asset_uri == "demo://fitting-preview/session_a/option_a"
    assert result.assets[0].quality_metadata["uses_real_scan_media"] is False
    assert result.asset_manifest is not None
    assert result.asset_manifest.asset_ids == [result.assets[0].asset_id]
    assert "Not production-grade cloth simulation." in result.warnings
    assert "Maker review required." in result.warnings


def test_unavailable_renderer_returns_structured_warnings_and_errors() -> None:
    service = FittingAssetPipelineService(UnavailableRendererProvider())
    result = service.generate_assets(
        design_session_id="session_a",
        fitting_result_id="fitting_a",
        fitting_request=VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.UNAVAILABLE,
        ),
        body_profile=_body_profile(),
        design_option=_design_option(),
    )

    assert result.renderer_provider is RendererProviderMode.UNAVAILABLE
    assert result.render_status is RenderStatus.UNAVAILABLE
    assert result.assets == []
    assert result.errors == ["renderer_unavailable"]
    assert result.asset_manifest is not None
    assert result.asset_manifest.render_status is RenderStatus.UNAVAILABLE


def test_virtual_fitting_result_includes_asset_manifest_and_beta_disclaimer() -> None:
    result = VirtualFittingEngine().preview(
        _design_option(),
        _body_profile(),
        VirtualFittingRequest(design_option_id="option_a"),
        design_session_id="session_a",
    )

    assert result.beta_preview_disclaimer == BETA_FITTING_DISCLAIMER
    assert result.fitting_preview_assets
    assert result.asset_manifest is not None
    assert result.asset_manifest.design_option_ref == "option_a"
    assert result.confidence_metadata["renderer_provider"] == "demo_synthetic"
    assert result.confidence_metadata["render_status"] == "demo_placeholder"
    assert result.fitting_preview_assets[0].quality_metadata["uses_real_scan_media"] is False
    assert "Maker review required." in result.caution_notes


def test_unavailable_renderer_path_is_exposed_through_fitting_engine() -> None:
    result = VirtualFittingEngine().preview(
        _design_option(),
        _body_profile(),
        VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.UNAVAILABLE,
        ),
        design_session_id="session_a",
    )

    assert result.asset_manifest is not None
    assert result.asset_manifest.render_status is RenderStatus.UNAVAILABLE
    assert result.confidence_metadata["renderer_provider"] == "unavailable"
    assert "renderer_unavailable" not in result.preview_asset_references[0]
    assert any("unavailable" in note for note in result.caution_notes)


def _body_profile() -> BodyProfileSnapshot:
    return BodyProfileSnapshot(
        body_profile_snapshot_id="body_profile_a",
        measurement_result_id="measurement_a",
        scan_id="scan_a",
        mannequin_reference_id="mannequin://body-ai/measurement_a",
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
        color_direction="emerald",
        fit_direction="balanced custom fit",
        asset_references=["synthetic://design-options/dress/1"],
    )

