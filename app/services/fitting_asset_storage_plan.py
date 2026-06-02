"""Planning helpers for future fitting preview asset storage.

This module does not upload files, connect to object storage, or generate real
provider signatures. It prepares deterministic metadata so future storage work
has a safe contract to implement against.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import re
from typing import Protocol

from pydantic import BaseModel, Field

from app.schemas.design_agent_virtual_fitting import (
    AssetDeletionState,
    FittingAssetStorageMetadata,
    FittingPreviewAsset,
    PrivacyGateStatus,
    SignedPreviewUrl,
    StorageProviderKind,
    StoredFittingAssetReference,
)


SIGNED_URL_STUB_WARNING = "Signed preview URL is a deterministic stub; no real provider signature was generated."
SYNTHETIC_ASSET_WARNING = "Synthetic/demo concept asset does not use real scan or photo media."
REAL_MEDIA_BLOCKED_WARNING = "Real scan/photo-derived fitting assets are blocked until privacy gates pass."


class AssetPrivacyGateEvaluation(BaseModel):
    uses_real_scan_media: bool
    allowed: bool
    privacy_gate_status: PrivacyGateStatus
    required_controls: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AssetRetentionPlan(BaseModel):
    asset_id: str
    object_key: str | None = None
    retention_days: int = Field(..., ge=0)
    retention_expires_at: datetime
    deletion_status: AssetDeletionState
    warnings: list[str] = Field(default_factory=list)


class StorageBackedAsset(Protocol):
    asset_id: str
    object_key: str | None


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def build_storage_object_key(
    environment: str,
    session_id: str,
    asset_id: str,
    extension: str,
) -> str:
    """Build an environment/session-scoped object key for future private storage."""

    normalized_extension = sanitize_extension(extension)
    return "/".join(
        [
            sanitize_component(environment),
            "design-sessions",
            sanitize_component(session_id),
            "fitting-assets",
            f"{sanitize_component(asset_id)}.{normalized_extension}",
        ]
    )


def build_signed_url_metadata_stub(
    asset_ref: FittingAssetStorageMetadata | StoredFittingAssetReference | FittingPreviewAsset,
    expires_in_seconds: int,
    *,
    now: datetime | None = None,
) -> SignedPreviewUrl:
    """Return a safe stub that models signed URL expiry without provider credentials."""

    if expires_in_seconds <= 0:
        raise ValueError("expires_in_seconds must be greater than zero.")
    generated_at = now or utc_now()
    object_key = object_key_for_asset(asset_ref)
    if not object_key:
        raise ValueError("asset_ref must include an object_key before signed URL metadata can be planned.")
    provider = provider_for_asset(asset_ref).value
    expires_at = generated_at + timedelta(seconds=expires_in_seconds)
    safe_key = "/".join(sanitize_component(part) for part in object_key.split("/") if part)
    return SignedPreviewUrl(
        signed_url=f"https://signed-preview.invalid/{provider}/{safe_key}?expires_at={expires_at.isoformat()}",
        signed_url_expires_at=expires_at,
        expires_in_seconds=expires_in_seconds,
        generated_at=generated_at,
        warnings=[
            SIGNED_URL_STUB_WARNING,
            "Do not log signed preview URLs or store them as permanent truth.",
        ],
    )


def evaluate_asset_privacy_gate(
    uses_real_scan_media: bool,
    consent_available: bool,
    deletion_policy_configured: bool,
    *,
    retention_policy_configured: bool = True,
    access_controls_configured: bool = True,
) -> AssetPrivacyGateEvaluation:
    """Evaluate whether a planned asset can proceed through the beta storage gate."""

    if not uses_real_scan_media:
        return AssetPrivacyGateEvaluation(
            uses_real_scan_media=False,
            allowed=True,
            privacy_gate_status=PrivacyGateStatus.SYNTHETIC_DEMO_ALLOWED,
            warnings=[SYNTHETIC_ASSET_WARNING],
        )

    missing_controls: list[str] = []
    if not consent_available:
        missing_controls.append("consent")
    if not deletion_policy_configured:
        missing_controls.append("deletion_policy")
    if not retention_policy_configured:
        missing_controls.append("retention_policy")
    if not access_controls_configured:
        missing_controls.append("access_controls")

    if missing_controls:
        return AssetPrivacyGateEvaluation(
            uses_real_scan_media=True,
            allowed=False,
            privacy_gate_status=PrivacyGateStatus.BLOCKED,
            required_controls=missing_controls,
            warnings=[REAL_MEDIA_BLOCKED_WARNING],
        )

    return AssetPrivacyGateEvaluation(
        uses_real_scan_media=True,
        allowed=True,
        privacy_gate_status=PrivacyGateStatus.APPROVED,
        warnings=["Real scan/photo-derived asset storage still requires maker/customer authorization context."],
    )


def build_asset_retention_plan(
    asset_ref: FittingAssetStorageMetadata | StoredFittingAssetReference | FittingPreviewAsset,
    retention_days: int,
    *,
    now: datetime | None = None,
) -> AssetRetentionPlan:
    """Plan retention metadata for an asset without deleting or uploading anything."""

    if retention_days < 0:
        raise ValueError("retention_days must be zero or greater.")
    planned_at = now or utc_now()
    object_key = object_key_for_asset(asset_ref)
    deletion_status = AssetDeletionState.ACTIVE if object_key else AssetDeletionState.NOT_STORED
    warnings = []
    if not object_key:
        warnings.append("Asset has no object key; retention plan is metadata-only.")
    return AssetRetentionPlan(
        asset_id=asset_ref.asset_id,
        object_key=object_key,
        retention_days=retention_days,
        retention_expires_at=planned_at + timedelta(days=retention_days),
        deletion_status=deletion_status,
        warnings=warnings,
    )


def build_storage_metadata_stub(
    *,
    asset_id: str,
    provider: StorageProviderKind,
    bucket: str | None,
    object_key: str | None,
    content_type: str,
    byte_size: int | None = None,
    checksum: str | None = None,
    uses_real_scan_media: bool = False,
    privacy_gate_status: PrivacyGateStatus = PrivacyGateStatus.PENDING_APPROVAL,
    retention_expires_at: datetime | None = None,
    warnings: list[str] | None = None,
) -> FittingAssetStorageMetadata:
    deletion_status = AssetDeletionState.ACTIVE if object_key else AssetDeletionState.NOT_STORED
    return FittingAssetStorageMetadata(
        asset_id=asset_id,
        provider=provider,
        bucket=bucket,
        object_key=object_key,
        content_type=content_type,
        byte_size=byte_size,
        checksum=checksum,
        retention_expires_at=retention_expires_at,
        deletion_status=deletion_status,
        privacy_gate_status=privacy_gate_status,
        uses_real_scan_media=uses_real_scan_media,
        warnings=list(warnings or []),
    )


def object_key_for_asset(
    asset_ref: FittingAssetStorageMetadata | StoredFittingAssetReference | FittingPreviewAsset,
) -> str | None:
    if isinstance(asset_ref, FittingPreviewAsset):
        return asset_ref.storage_metadata.object_key if asset_ref.storage_metadata else None
    if isinstance(asset_ref, StoredFittingAssetReference):
        return asset_ref.object_key or (
            asset_ref.storage_metadata.object_key if asset_ref.storage_metadata else None
        )
    return asset_ref.object_key


def provider_for_asset(
    asset_ref: FittingAssetStorageMetadata | StoredFittingAssetReference | FittingPreviewAsset,
) -> StorageProviderKind:
    if isinstance(asset_ref, FittingPreviewAsset):
        return asset_ref.storage_metadata.provider if asset_ref.storage_metadata else StorageProviderKind.NOT_STORED
    return asset_ref.provider


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return sanitized or "unknown"


def sanitize_extension(extension: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "", extension.strip().lstrip(".")).lower()
    return sanitized or "bin"
