from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_6f_design_e_renderer_provider_selection_plan.md"
PIPELINE_PATH = ROOT / "app/services/fitting_asset_pipeline.py"
SCHEMA_PATH = ROOT / "app/schemas/design_agent_virtual_fitting.py"


REQUIRED_DOC_SNIPPETS = [
    "Renderer Options Comparison",
    "2D silhouette/garment overlay renderer",
    "Three.js/WebGL mannequin snapshot renderer",
    "Server-side Blender renderer",
    "External cloth simulation provider",
    "hybrid staged approach",
    "Recommended Beta Renderer Path",
    "Stage 1: 2D Silhouette/Garment Overlay Or Mannequin/Design Concept Renderer",
    "Stage 2: Three.js Or Blender Mannequin Snapshot Renderer",
    "Stage 3: Validated Cloth Simulation Or External Provider",
    "Real Render Service Contract",
    "bodyProfileSnapshot",
    "mannequinRef",
    "selectedDesignOption",
    "garmentMetadata",
    "renderMode",
    "FittingAssetGenerationResult",
    "FittingAssetManifest",
    "Storage And Privacy Gates",
    "signed URLs",
    "Real scan/photo-derived assets cannot be stored",
    "Validation Gates Before Production-Grade Fitting Claims",
    "manual measurement comparison",
    "maker review feedback",
    "garment outcome feedback",
    "visual render QA",
    "error tracking",
    "customer approval tracking",
    "not production-grade cloth simulation",
    "Phase 6F-Design-F1: Beta 2D Mannequin/Garment Concept Renderer in ai-body-engine",
]

REQUIRED_PROVIDER_MODES = [
    "DEMO_SYNTHETIC = \"demo_synthetic\"",
    "LOCAL_PLACEHOLDER = \"local_placeholder\"",
    "EXTERNAL_RENDERER = \"external_renderer\"",
    "UNAVAILABLE = \"unavailable\"",
]

REQUIRED_PIPELINE_SNIPPETS = [
    "DemoSyntheticRendererProvider",
    "LocalPlaceholderRendererProvider",
    "UnavailableRendererProvider",
    "demo://fitting-preview/",
    "Not production-grade cloth simulation.",
]

DISALLOWED_PRODUCTION_CLAIMS = [
    "production-grade cloth simulation is already available",
    "production-grade cloth simulation is enabled",
    "production-grade cloth simulation is validated",
    "validated production-grade renderer is enabled",
    "real-world fit accuracy is guaranteed",
    "real-world measurement accuracy is guaranteed",
    "customer approval automatically starts production",
]


def main() -> int:
    if not DOC_PATH.exists():
        raise SystemExit(f"Missing renderer provider selection plan: {DOC_PATH.relative_to(ROOT)}")

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    doc_text_lower = doc_text.lower()
    missing_doc = [snippet for snippet in REQUIRED_DOC_SNIPPETS if snippet.lower() not in doc_text_lower]
    if missing_doc:
        raise SystemExit(f"Renderer provider selection plan missing snippets: {', '.join(missing_doc)}")

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    missing_modes = [snippet for snippet in REQUIRED_PROVIDER_MODES if snippet not in schema_text]
    if missing_modes:
        raise SystemExit(f"RendererProviderMode missing expected modes: {', '.join(missing_modes)}")

    pipeline_text = PIPELINE_PATH.read_text(encoding="utf-8")
    missing_pipeline = [snippet for snippet in REQUIRED_PIPELINE_SNIPPETS if snippet not in pipeline_text]
    if missing_pipeline:
        raise SystemExit(f"Fitting asset pipeline missing expected provider support: {', '.join(missing_pipeline)}")

    combined_text = "\n".join([doc_text, schema_text, pipeline_text]).lower()
    bad_claims = [claim for claim in DISALLOWED_PRODUCTION_CLAIMS if claim in combined_text]
    if bad_claims:
        raise SystemExit(f"Found disallowed production/accuracy claims: {', '.join(bad_claims)}")

    print("Phase 6F-Design-E renderer provider selection plan verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
