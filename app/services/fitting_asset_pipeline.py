"""Fitting preview asset providers.

The provider boundary is intentionally narrow while renderer capability is staged:
demo/synthetic metadata first, then beta 2D/mannequin concept assets, then higher
fidelity snapshot renderers, and only later validated cloth simulation once
measurement, garment, privacy, storage, and QA gates are complete.
"""

from __future__ import annotations

from typing import Protocol

from app.schemas.design_agent_virtual_fitting import (
    BETA_FITTING_DISCLAIMER,
    BodyProfileSnapshot,
    FittingAssetGenerationRequest,
    FittingAssetGenerationResult,
    FittingAssetManifest,
    FittingPreviewAsset,
    FittingPreviewAssetType,
    FittingPreviewKind,
    GeneratedDesignOption,
    RenderStatus,
    RendererProviderMode,
    VirtualFittingRequest,
)


DEMO_ASSET_WARNINGS = [
    "Beta preview only.",
    "Not production-grade cloth simulation.",
    "Maker review required.",
]

BETA_2D_CONCEPT_WARNINGS = [
    "Beta 2D concept preview only.",
    "Not production-grade cloth simulation.",
    "Maker review required.",
]


class RendererProvider(Protocol):
    mode: RendererProviderMode

    def generate(self, request: FittingAssetGenerationRequest) -> FittingAssetGenerationResult:
        """Generate or describe fitting preview assets for a fitting request."""


class DemoSyntheticRendererProvider:
    mode = RendererProviderMode.DEMO_SYNTHETIC

    def generate(self, request: FittingAssetGenerationRequest) -> FittingAssetGenerationResult:
        session_ref = request.design_session_id or "session_unscoped"
        option_ref = request.design_option.design_option_id
        base_uri = f"demo://fitting-preview/{session_ref}/{option_ref}"
        body_profile = request.body_profile_snapshot
        asset = FittingPreviewAsset(
            asset_id=f"fitting_asset_demo_{session_ref}_{option_ref}",
            asset_type=FittingPreviewAssetType.IMAGE,
            preview_kind=request.preview_kind,
            render_status=RenderStatus.DEMO_PLACEHOLDER,
            renderer_provider=self.mode,
            asset_uri=base_uri,
            image_url=base_uri,
            thumbnail_url=f"{base_uri}/thumbnail",
            manifest_uri=f"{base_uri}/manifest",
            body_profile_ref=body_profile.body_profile_snapshot_id,
            mannequin_ref=body_profile.mannequin_reference_id,
            design_option_ref=option_ref,
            garment_layer_refs=list(request.garment_layer_refs),
            warnings=[*DEMO_ASSET_WARNINGS, *body_profile.warnings],
            confidence_metadata={
                "mode": request.preview_mode,
                "maker_review_required": request.maker_review_required,
                "real_world_validated": body_profile.real_world_validated,
            },
            quality_metadata={
                "validated_renderer": False,
                "uses_real_scan_media": False,
                "graphical_preview_level": "synthetic_demo_metadata",
            },
        )
        manifest = FittingAssetManifest(
            manifest_id=f"fitting_manifest_demo_{session_ref}_{option_ref}",
            preview_kind=request.preview_kind,
            render_status=RenderStatus.DEMO_PLACEHOLDER,
            renderer_provider=self.mode,
            manifest_uri=f"{base_uri}/manifest",
            asset_ids=[asset.asset_id],
            assets=[asset],
            body_profile_ref=body_profile.body_profile_snapshot_id,
            mannequin_ref=body_profile.mannequin_reference_id,
            design_option_ref=option_ref,
            garment_layer_refs=list(request.garment_layer_refs),
            warnings=[*DEMO_ASSET_WARNINGS, *body_profile.warnings],
            confidence_metadata=dict(asset.confidence_metadata),
            quality_metadata=dict(asset.quality_metadata),
        )
        return FittingAssetGenerationResult(
            request_id=request.request_id,
            preview_kind=request.preview_kind,
            render_status=RenderStatus.DEMO_PLACEHOLDER,
            renderer_provider=self.mode,
            assets=[asset],
            asset_manifest=manifest,
            warnings=list(manifest.warnings),
            confidence_metadata=dict(asset.confidence_metadata),
            quality_metadata=dict(asset.quality_metadata),
        )


class LocalPlaceholderRendererProvider(DemoSyntheticRendererProvider):
    mode = RendererProviderMode.LOCAL_PLACEHOLDER


class Beta2DConceptRendererProvider:
    mode = RendererProviderMode.BETA_2D_CONCEPT

    def generate(self, request: FittingAssetGenerationRequest) -> FittingAssetGenerationResult:
        session_ref = request.design_session_id or "session_unscoped"
        option_ref = request.design_option.design_option_id
        body_profile = request.body_profile_snapshot
        concept_uri = f"concept://fitting-preview/{session_ref}/{option_ref}"
        preview_kind = FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT
        concept_metadata = build_2d_concept_metadata(body_profile, request.design_option)
        warnings = [*BETA_2D_CONCEPT_WARNINGS, *body_profile.warnings]
        quality_metadata = {
            "validated_renderer": False,
            "uses_real_scan_media": False,
            "graphical_preview_level": "beta_2d_mannequin_garment_concept",
            "concept_asset_format": "metadata_uri",
            "concept_metadata": concept_metadata,
        }
        confidence_metadata = {
            "mode": request.preview_mode,
            "maker_review_required": request.maker_review_required,
            "real_world_validated": body_profile.real_world_validated,
            "body_profile_snapshot_id": body_profile.body_profile_snapshot_id,
            "mannequin_reference_id": body_profile.mannequin_reference_id,
        }
        asset = FittingPreviewAsset(
            asset_id=f"fitting_asset_beta_2d_{session_ref}_{option_ref}",
            asset_type=FittingPreviewAssetType.MANNEQUIN_OVERLAY,
            preview_kind=preview_kind,
            render_status=RenderStatus.GENERATED,
            renderer_provider=self.mode,
            asset_uri=concept_uri,
            thumbnail_url=f"{concept_uri}/thumbnail",
            manifest_uri=f"{concept_uri}/manifest",
            body_profile_ref=body_profile.body_profile_snapshot_id,
            mannequin_ref=body_profile.mannequin_reference_id,
            design_option_ref=option_ref,
            garment_layer_refs=list(request.garment_layer_refs),
            warnings=warnings,
            confidence_metadata=confidence_metadata,
            quality_metadata=quality_metadata,
        )
        manifest = FittingAssetManifest(
            manifest_id=f"fitting_manifest_beta_2d_{session_ref}_{option_ref}",
            preview_kind=preview_kind,
            render_status=RenderStatus.GENERATED,
            renderer_provider=self.mode,
            manifest_uri=f"{concept_uri}/manifest",
            asset_ids=[asset.asset_id],
            assets=[asset],
            body_profile_ref=body_profile.body_profile_snapshot_id,
            mannequin_ref=body_profile.mannequin_reference_id,
            design_option_ref=option_ref,
            garment_layer_refs=list(request.garment_layer_refs),
            warnings=warnings,
            confidence_metadata=dict(confidence_metadata),
            quality_metadata=dict(quality_metadata),
        )
        return FittingAssetGenerationResult(
            request_id=request.request_id,
            preview_kind=preview_kind,
            render_status=RenderStatus.GENERATED,
            renderer_provider=self.mode,
            assets=[asset],
            asset_manifest=manifest,
            warnings=list(manifest.warnings),
            confidence_metadata=dict(confidence_metadata),
            quality_metadata=dict(quality_metadata),
        )


class UnavailableRendererProvider:
    mode = RendererProviderMode.UNAVAILABLE

    def generate(self, request: FittingAssetGenerationRequest) -> FittingAssetGenerationResult:
        warning = "Renderer provider unavailable; fitting preview asset generation did not run."
        return FittingAssetGenerationResult(
            request_id=request.request_id,
            preview_kind=request.preview_kind,
            render_status=RenderStatus.UNAVAILABLE,
            renderer_provider=self.mode,
            assets=[],
            asset_manifest=FittingAssetManifest(
                manifest_id=f"fitting_manifest_unavailable_{request.request_id}",
                preview_kind=request.preview_kind,
                render_status=RenderStatus.UNAVAILABLE,
                renderer_provider=self.mode,
                body_profile_ref=request.body_profile_snapshot.body_profile_snapshot_id,
                mannequin_ref=request.body_profile_snapshot.mannequin_reference_id,
                design_option_ref=request.design_option.design_option_id,
                garment_layer_refs=list(request.garment_layer_refs),
                warnings=[warning, *DEMO_ASSET_WARNINGS],
                confidence_metadata={
                    "maker_review_required": request.maker_review_required,
                    "real_world_validated": request.body_profile_snapshot.real_world_validated,
                },
                quality_metadata={"validated_renderer": False, "uses_real_scan_media": False},
            ),
            warnings=[warning, *DEMO_ASSET_WARNINGS],
            errors=["renderer_unavailable"],
            confidence_metadata={"maker_review_required": request.maker_review_required},
            quality_metadata={"validated_renderer": False, "uses_real_scan_media": False},
        )


def build_2d_concept_metadata(
    body_profile: BodyProfileSnapshot,
    design_option: GeneratedDesignOption,
) -> dict[str, object]:
    measurements = body_profile.measurement_summary_cm
    garment_type = garment_type_from_design_option(design_option)
    primary_color = design_option.color_direction.split(",")[0].strip() or "maker-selected palette"
    body_shape_hint = " / ".join(body_profile.morphology_tags) if body_profile.morphology_tags else "body profile reference"
    return {
        "concept_style": "2d_mannequin_garment_overlay",
        "garment_type": garment_type,
        "primary_color": primary_color,
        "style_description": design_option.style_description,
        "fit_direction": design_option.fit_direction,
        "body_shape_hint": body_shape_hint,
        "mannequin_ref": body_profile.mannequin_reference_id,
        "measurement_summary_cm": {
            key: measurements[key]
            for key in ("chest", "waist", "hip", "height", "shoulder_width")
            if key in measurements
        },
        "render_notes": [
            "Concept uses body profile/mannequin metadata only.",
            "No real user scan or photo media is used.",
            "No cloth physics simulation is performed.",
        ],
    }


def garment_type_from_design_option(design_option: GeneratedDesignOption) -> str:
    for detail in design_option.garment_details:
        label, separator, value = detail.partition(":")
        if separator and label.strip().lower() == "garment type" and value.strip():
            return value.strip().lower()
    title_parts = design_option.title.lower().split()
    return title_parts[-2] if title_parts[-1:] == ["concept"] and len(title_parts) >= 2 else design_option.title.lower()


class FittingAssetPipelineService:
    def __init__(self, renderer_provider: RendererProvider | None = None) -> None:
        self.renderer_provider = renderer_provider or DemoSyntheticRendererProvider()

    def build_request(
        self,
        *,
        design_session_id: str | None,
        fitting_result_id: str | None,
        fitting_request: VirtualFittingRequest,
        body_profile: BodyProfileSnapshot,
        design_option: GeneratedDesignOption,
    ) -> FittingAssetGenerationRequest:
        return FittingAssetGenerationRequest(
            request_id=f"fitting_asset_request_{fitting_result_id or design_option.design_option_id}",
            design_session_id=design_session_id,
            fitting_result_id=fitting_result_id,
            preview_mode=fitting_request.preview_mode,
            preview_kind=fitting_request.preview_kind,
            renderer_provider=self.renderer_provider.mode,
            body_profile_snapshot=body_profile,
            design_option=design_option,
            garment_layer_refs=list(design_option.asset_references),
            maker_review_required=fitting_request.maker_review_required,
            beta_disclaimer=BETA_FITTING_DISCLAIMER,
        )

    def generate_assets(
        self,
        *,
        design_session_id: str | None,
        fitting_result_id: str | None,
        fitting_request: VirtualFittingRequest,
        body_profile: BodyProfileSnapshot,
        design_option: GeneratedDesignOption,
    ) -> FittingAssetGenerationResult:
        generation_request = self.build_request(
            design_session_id=design_session_id,
            fitting_result_id=fitting_result_id,
            fitting_request=fitting_request,
            body_profile=body_profile,
            design_option=design_option,
        )
        return self.renderer_provider.generate(generation_request)
