from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_6f_design_g_asset_storage_signed_url_plan.md"
SCHEMA_PATH = ROOT / "app/schemas/design_agent_virtual_fitting.py"
SERVICE_PATH = ROOT / "app/services/fitting_asset_storage_plan.py"
PIPELINE_PATH = ROOT / "app/services/fitting_asset_pipeline.py"
TEST_PATH = ROOT / "tests/test_asset_storage_signed_url_plan.py"
BETA_TEST_PATH = ROOT / "tests/test_beta_2d_concept_renderer.py"


REQUIRED_DOC_SNIPPETS = [
    "Current Asset State",
    "concept:// beta assets exist",
    "No real generated SVG, PNG, image, or render asset storage is active yet.",
    "Future Real Asset Flow",
    "Storage Provider Options",
    "AWS S3",
    "Cloudflare R2",
    "Supabase Storage",
    "local development storage",
    "Recommended Beta Storage Path",
    "Asset Metadata Contract",
    "Signed URL Safety Rules",
    "Retention/Deletion Policy",
    "Privacy Gates",
    "FashionApp Integration Expectations",
    "Phase 6F-Design-H1",
]

REQUIRED_SCHEMA_SNIPPETS = [
    "class FittingAssetStorageMetadata",
    "class SignedPreviewUrl",
    "class AssetDeletionState",
    "class PrivacyGateStatus",
    "class StorageProviderKind",
    "class StoredFittingAssetReference",
    "storage_metadata: FittingAssetStorageMetadata | None = None",
    "stored_asset_references: list[StoredFittingAssetReference]",
]

REQUIRED_SERVICE_SNIPPETS = [
    "build_storage_object_key",
    "build_signed_url_metadata_stub",
    "evaluate_asset_privacy_gate",
    "build_asset_retention_plan",
    "build_storage_metadata_stub",
    "signed-preview.invalid",
    "REAL_MEDIA_BLOCKED_WARNING",
    "provider signatures",
]

REQUIRED_TEST_SNIPPETS = [
    "test_storage_metadata_schema_validates_signed_url_expiry",
    "test_storage_object_keys_are_deterministic_and_environment_session_scoped",
    "test_signed_url_metadata_stub_supports_expiry_without_real_provider_signature",
    "test_privacy_gate_blocks_real_scan_media_without_required_controls",
    "test_privacy_gate_allows_synthetic_demo_assets",
    "test_retention_plan_includes_expiry_and_deletion_state",
    "test_beta_2d_concept_renderer_still_does_not_require_storage",
    "test_storage_plan_service_does_not_reference_network_clients_or_secrets",
]

DISALLOWED_SECRET_SNIPPETS = [
    "aws_secret_access_key",
    "secret_access_key=",
    "r2_access_key_secret",
    "supabase_service_role_key",
    "-----BEGIN PRIVATE KEY-----",
]

DISALLOWED_PRODUCTION_CLAIMS = [
    "production-grade cloth simulation is already available",
    "production-grade cloth simulation is enabled",
    "production-grade cloth simulation is validated",
    "real-world fit accuracy is guaranteed",
    "real-world fitting accuracy is guaranteed",
    "user-derived assets may use public buckets",
]


def main() -> int:
    required_files = [DOC_PATH, SCHEMA_PATH, SERVICE_PATH, PIPELINE_PATH, TEST_PATH, BETA_TEST_PATH]
    missing_files = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing_files:
        raise SystemExit(f"Missing asset storage signed URL files: {', '.join(missing_files)}")

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    missing_doc = [snippet for snippet in REQUIRED_DOC_SNIPPETS if snippet.lower() not in doc_text.lower()]
    if missing_doc:
        raise SystemExit(f"Asset storage plan doc missing snippets: {', '.join(missing_doc)}")

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    missing_schema = [snippet for snippet in REQUIRED_SCHEMA_SNIPPETS if snippet not in schema_text]
    if missing_schema:
        raise SystemExit(f"Asset storage schema missing snippets: {', '.join(missing_schema)}")

    service_text = SERVICE_PATH.read_text(encoding="utf-8")
    missing_service = [snippet for snippet in REQUIRED_SERVICE_SNIPPETS if snippet not in service_text]
    if missing_service:
        raise SystemExit(f"Asset storage planner service missing snippets: {', '.join(missing_service)}")

    test_text = TEST_PATH.read_text(encoding="utf-8")
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in test_text]
    if missing_tests:
        raise SystemExit(f"Asset storage tests missing snippets: {', '.join(missing_tests)}")

    beta_test_text = BETA_TEST_PATH.read_text(encoding="utf-8")
    if "concept://fitting-preview/session_a/option_a" not in beta_test_text:
        raise SystemExit("Beta concept renderer compatibility test no longer checks concept:// output.")

    safety_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [DOC_PATH, SCHEMA_PATH, SERVICE_PATH, PIPELINE_PATH, TEST_PATH, BETA_TEST_PATH]
    ).lower()
    found_secrets = [snippet for snippet in DISALLOWED_SECRET_SNIPPETS if snippet in safety_text]
    if found_secrets:
        raise SystemExit(f"Found disallowed provider credential snippets: {', '.join(found_secrets)}")

    bad_claims = [claim for claim in DISALLOWED_PRODUCTION_CLAIMS if claim in safety_text]
    if bad_claims:
        raise SystemExit(f"Found disallowed production/accuracy claims: {', '.join(bad_claims)}")

    print("Phase 6F-Design-G asset storage signed URL plan verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
