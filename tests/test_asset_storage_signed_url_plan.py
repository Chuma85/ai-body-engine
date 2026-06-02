from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.design_agent_virtual_fitting import (
    AssetDeletionState,
    FittingAssetStorageMetadata,
    FittingPreviewAsset,
    FittingPreviewAssetType,
    FittingPreviewKind,
    PrivacyGateStatus,
    RenderStatus,
    RendererProviderMode,
    SignedPreviewUrl,
    StorageProviderKind,
    StoredFittingAssetReference,
)
from app.services.fitting_asset_storage_plan import (
    REAL_MEDIA_BLOCKED_WARNING,
    SIGNED_URL_STUB_WARNING,
    build_asset_retention_plan,
    build_signed_url_metadata_stub,
    build_storage_metadata_stub,
    build_storage_object_key,
    evaluate_asset_privacy_gate,
)
from app.services.fitting_asset_pipeline import Beta2DConceptRendererProvider, FittingAssetPipelineService
from app.schemas.design_agent_virtual_fitting import BodyProfileSnapshot, GeneratedDesignOption, VirtualFittingRequest


FIXED_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def test_storage_metadata_schema_validates_signed_url_expiry() -> None:
    signed_url = SignedPreviewUrl(
        signed_url="https://signed-preview.invalid/cloudflare_r2/dev/session/asset.png",
        signed_url_expires_at=datetime(2026, 6, 1, 12, 15, 0, tzinfo=UTC),
        expires_in_seconds=900,
        generated_at=FIXED_NOW,
        warnings=["Do not log signed preview URLs."],
    )
    metadata = FittingAssetStorageMetadata(
        asset_id="asset_a",
        provider=StorageProviderKind.CLOUDFLARE_R2,
        bucket="private-fitting-preview-beta",
        object_key="development/design-sessions/session_a/fitting-assets/asset_a.png",
        content_type="image/png",
        byte_size=2048,
        checksum="sha256:test",
        signed_url=signed_url,
        retention_expires_at=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        deletion_status=AssetDeletionState.ACTIVE,
        privacy_gate_status=PrivacyGateStatus.SYNTHETIC_DEMO_ALLOWED,
        uses_real_scan_media=False,
    )
    asset = FittingPreviewAsset(
        asset_id="asset_a",
        asset_type=FittingPreviewAssetType.IMAGE,
        preview_kind=FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT,
        render_status=RenderStatus.GENERATED,
        renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
        storage_metadata=metadata,
    )

    assert asset.storage_metadata is not None
    assert asset.storage_metadata.signed_url is not None
    assert asset.storage_metadata.signed_url.expires_in_seconds == 900
    assert asset.storage_metadata.deletion_status is AssetDeletionState.ACTIVE


def test_storage_object_keys_are_deterministic_and_environment_session_scoped() -> None:
    object_key = build_storage_object_key(
        environment="Staging Beta",
        session_id="design/session A",
        asset_id="asset:front png",
        extension=".PNG",
    )

    assert object_key == "staging_beta/design-sessions/design_session_a/fitting-assets/asset_front_png.png"
    assert ".." not in object_key


def test_signed_url_metadata_stub_supports_expiry_without_real_provider_signature() -> None:
    metadata = build_storage_metadata_stub(
        asset_id="asset_a",
        provider=StorageProviderKind.CLOUDFLARE_R2,
        bucket="private-fitting-preview-beta",
        object_key="development/design-sessions/session_a/fitting-assets/asset_a.png",
        content_type="image/png",
        uses_real_scan_media=False,
        privacy_gate_status=PrivacyGateStatus.SYNTHETIC_DEMO_ALLOWED,
    )

    signed = build_signed_url_metadata_stub(metadata, 300, now=FIXED_NOW)

    assert signed.signed_url.startswith("https://signed-preview.invalid/cloudflare_r2/")
    assert signed.signed_url_expires_at == datetime(2026, 6, 1, 12, 5, 0, tzinfo=UTC)
    assert SIGNED_URL_STUB_WARNING in signed.warnings


def test_privacy_gate_blocks_real_scan_media_without_required_controls() -> None:
    evaluation = evaluate_asset_privacy_gate(
        uses_real_scan_media=True,
        consent_available=False,
        deletion_policy_configured=False,
        retention_policy_configured=False,
    )

    assert evaluation.allowed is False
    assert evaluation.privacy_gate_status is PrivacyGateStatus.BLOCKED
    assert evaluation.required_controls == ["consent", "deletion_policy", "retention_policy"]
    assert REAL_MEDIA_BLOCKED_WARNING in evaluation.warnings


def test_privacy_gate_allows_synthetic_demo_assets() -> None:
    evaluation = evaluate_asset_privacy_gate(
        uses_real_scan_media=False,
        consent_available=False,
        deletion_policy_configured=False,
        retention_policy_configured=False,
        access_controls_configured=False,
    )

    assert evaluation.allowed is True
    assert evaluation.privacy_gate_status is PrivacyGateStatus.SYNTHETIC_DEMO_ALLOWED
    assert evaluation.required_controls == []


def test_retention_plan_includes_expiry_and_deletion_state() -> None:
    asset_ref = StoredFittingAssetReference(
        asset_id="asset_a",
        provider=StorageProviderKind.AWS_S3,
        object_key="development/design-sessions/session_a/fitting-assets/asset_a.svg",
        content_type="image/svg+xml",
        preview_kind=FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT,
        render_status=RenderStatus.GENERATED,
        renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
    )

    plan = build_asset_retention_plan(asset_ref, retention_days=14, now=FIXED_NOW)

    assert plan.asset_id == "asset_a"
    assert plan.object_key == "development/design-sessions/session_a/fitting-assets/asset_a.svg"
    assert plan.retention_expires_at == datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    assert plan.deletion_status is AssetDeletionState.ACTIVE


def test_beta_2d_concept_renderer_still_does_not_require_storage() -> None:
    service = FittingAssetPipelineService(Beta2DConceptRendererProvider())
    result = service.generate_assets(
        design_session_id="session_a",
        fitting_result_id="fitting_a",
        fitting_request=VirtualFittingRequest(
            design_option_id="option_a",
            renderer_provider=RendererProviderMode.BETA_2D_CONCEPT,
        ),
        body_profile=_body_profile(),
        design_option=_design_option(),
    )

    assert result.assets[0].asset_uri == "concept://fitting-preview/session_a/option_a"
    assert result.assets[0].storage_metadata is None
    assert result.asset_manifest is not None
    assert result.asset_manifest.stored_asset_references == []
    assert result.assets[0].quality_metadata["uses_real_scan_media"] is False
    assert result.assets[0].quality_metadata["validated_renderer"] is False


def test_storage_plan_service_does_not_reference_network_clients_or_secrets() -> None:
    import app.services.fitting_asset_storage_plan as storage_plan

    source = storage_plan.__loader__.get_source(storage_plan.__name__) or ""
    disallowed_snippets = ["boto3", "requests", "httpx", "supabase", "secret_access_key", "api_key"]

    assert all(snippet not in source.lower() for snippet in disallowed_snippets)


def _body_profile() -> BodyProfileSnapshot:
    return BodyProfileSnapshot(
        body_profile_snapshot_id="body_profile_a",
        measurement_result_id="measurement_a",
        scan_id=None,
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
