from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "docs/phase_6f_design_d1_graphical_fitting_asset_pipeline.md",
    "app/schemas/design_agent_virtual_fitting.py",
    "app/services/fitting_asset_pipeline.py",
    "app/services/design_agent_virtual_fitting.py",
    "tests/test_graphical_fitting_asset_pipeline.py",
]

REQUIRED_SCHEMA_NAMES = [
    "FittingPreviewAsset",
    "FittingAssetManifest",
    "FittingAssetGenerationRequest",
    "FittingAssetGenerationResult",
    "RendererProviderMode",
    "RenderStatus",
    "FittingPreviewKind",
]

REQUIRED_SERVICE_SNIPPETS = [
    "class RendererProvider",
    "class DemoSyntheticRendererProvider",
    "class UnavailableRendererProvider",
    "class FittingAssetPipelineService",
    "demo://fitting-preview/",
    "uses_real_scan_media",
]

REQUIRED_TEST_SNIPPETS = [
    "test_fitting_preview_asset_schema_validates_contract",
    "test_demo_synthetic_renderer_returns_deterministic_safe_metadata",
    "test_unavailable_renderer_returns_structured_warnings_and_errors",
    "test_virtual_fitting_result_includes_asset_manifest_and_beta_disclaimer",
]

REQUIRED_DOC_SNIPPETS = [
    "Product Flow",
    "Pipeline Architecture",
    "Asset Contract",
    "Renderer Provider Modes",
    "Beta Safety Constraints",
    "FashionApp Integration Expectations",
    "No production-grade cloth simulation claim is made.",
    "No real user scan/photo use is allowed until privacy, storage, and deletion gates pass.",
    "Customer approval does not automatically start production.",
]

DISALLOWED_CLAIMS = [
    "production-grade cloth simulation is enabled",
    "production-grade cloth simulation is validated",
    "real-world measurement accuracy is guaranteed",
    "customer approval automatically starts production",
]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    schema_text = (ROOT / "app/schemas/design_agent_virtual_fitting.py").read_text(encoding="utf-8")
    missing_schemas = [name for name in REQUIRED_SCHEMA_NAMES if f"class {name}" not in schema_text]
    if missing_schemas:
        raise SystemExit(f"Missing required schema classes: {', '.join(missing_schemas)}")

    result_snippets = ["fitting_preview_assets", "asset_manifest", "preview_asset_references"]
    missing_result_snippets = [snippet for snippet in result_snippets if snippet not in schema_text]
    if missing_result_snippets:
        raise SystemExit(f"VirtualFittingResult missing asset support: {', '.join(missing_result_snippets)}")

    service_text = (ROOT / "app/services/fitting_asset_pipeline.py").read_text(encoding="utf-8")
    missing_service = [snippet for snippet in REQUIRED_SERVICE_SNIPPETS if snippet not in service_text]
    if missing_service:
        raise SystemExit(f"Fitting asset pipeline service missing snippets: {', '.join(missing_service)}")

    workflow_text = (ROOT / "app/services/design_agent_virtual_fitting.py").read_text(encoding="utf-8")
    for required in ("FittingAssetPipelineService", "fitting_preview_assets", "asset_manifest"):
        if required not in workflow_text:
            raise SystemExit(f"Fitting preview workflow missing enrichment snippet: {required}")

    test_text = (ROOT / "tests/test_graphical_fitting_asset_pipeline.py").read_text(encoding="utf-8")
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in test_text]
    if missing_tests:
        raise SystemExit(f"Missing required test coverage snippets: {', '.join(missing_tests)}")

    doc_text = (ROOT / "docs/phase_6f_design_d1_graphical_fitting_asset_pipeline.md").read_text(encoding="utf-8")
    missing_docs = [snippet for snippet in REQUIRED_DOC_SNIPPETS if snippet not in doc_text]
    if missing_docs:
        raise SystemExit(f"Documentation missing required safety/architecture content: {', '.join(missing_docs)}")

    combined_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in REQUIRED_FILES
        if (ROOT / path).suffix in {".py", ".md"}
    ).lower()
    bad_claims = [claim for claim in DISALLOWED_CLAIMS if claim in combined_text]
    if bad_claims:
        raise SystemExit(f"Found disallowed production/accuracy claims: {', '.join(bad_claims)}")

    print("Phase 6F-Design-D1 graphical fitting asset pipeline verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

