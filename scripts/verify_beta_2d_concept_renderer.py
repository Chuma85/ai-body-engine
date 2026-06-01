from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_6f_design_f1_beta_2d_mannequin_garment_concept_renderer.md"
SCHEMA_PATH = ROOT / "app/schemas/design_agent_virtual_fitting.py"
PIPELINE_PATH = ROOT / "app/services/fitting_asset_pipeline.py"
WORKFLOW_PATH = ROOT / "app/services/design_agent_virtual_fitting.py"
TEST_PATH = ROOT / "tests/test_beta_2d_concept_renderer.py"


REQUIRED_DOC_SNIPPETS = [
    "first real beta visual renderer path",
    "safe 2D mannequin/garment concept preview",
    "mobile-friendly graphical fitting concept",
    "not production-grade cloth simulation",
    "Input Contract",
    "body profile snapshot",
    "mannequin/body morphology reference",
    "selected generated design option",
    "garment type",
    "color palette",
    "style direction",
    "fit preference",
    "render mode/provider",
    "Output Contract",
    "fitting preview asset metadata",
    "asset manifest",
    "concept://fitting-preview/{sessionId}/{optionId}",
    "renderer_provider=beta_2d_concept",
    "renderStatus=generated",
    "previewKind=beta_2d_mannequin_concept",
    "FashionApp Compatibility",
    "Future Upgrade Path",
]

REQUIRED_SCHEMA_SNIPPETS = [
    "BETA_2D_CONCEPT = \"beta_2d_concept\"",
    "BETA_2D_MANNEQUIN_CONCEPT = \"beta_2d_mannequin_concept\"",
]

REQUIRED_PIPELINE_SNIPPETS = [
    "class Beta2DConceptRendererProvider",
    "BETA_2D_CONCEPT_WARNINGS",
    "Beta 2D concept preview only.",
    "Not production-grade cloth simulation.",
    "Maker review required.",
    "concept://fitting-preview/",
    "build_2d_concept_metadata",
    "uses_real_scan_media",
    "validated_renderer",
    "beta_2d_mannequin_garment_concept",
]

REQUIRED_WORKFLOW_SNIPPETS = [
    "Beta2DConceptRendererProvider",
    "RendererProviderMode.BETA_2D_CONCEPT",
]

REQUIRED_TEST_SNIPPETS = [
    "test_beta_2d_concept_renderer_returns_deterministic_asset_metadata",
    "test_beta_2d_concept_renderer_does_not_require_real_scan_photo_fields",
    "test_virtual_fitting_result_can_use_beta_2d_concept_renderer",
    "RendererProviderMode.BETA_2D_CONCEPT",
    "FittingPreviewKind.BETA_2D_MANNEQUIN_CONCEPT",
    "Not production-grade cloth simulation.",
    "Maker review required.",
]

DISALLOWED_CLAIMS = [
    "production-grade cloth simulation is already available",
    "production-grade cloth simulation is enabled",
    "production-grade cloth simulation is validated",
    "real-world fitting accuracy is guaranteed",
    "real-world fit accuracy is guaranteed",
    "uses real user scan/photo media",
]


def main() -> int:
    required_files = [DOC_PATH, SCHEMA_PATH, PIPELINE_PATH, WORKFLOW_PATH, TEST_PATH]
    missing_files = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing_files:
        raise SystemExit(f"Missing beta 2D concept renderer files: {', '.join(missing_files)}")

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    missing_doc = [snippet for snippet in REQUIRED_DOC_SNIPPETS if snippet.lower() not in doc_text.lower()]
    if missing_doc:
        raise SystemExit(f"Beta 2D concept renderer doc missing snippets: {', '.join(missing_doc)}")

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    missing_schema = [snippet for snippet in REQUIRED_SCHEMA_SNIPPETS if snippet not in schema_text]
    if missing_schema:
        raise SystemExit(f"Schema missing beta 2D concept contract snippets: {', '.join(missing_schema)}")

    pipeline_text = PIPELINE_PATH.read_text(encoding="utf-8")
    missing_pipeline = [snippet for snippet in REQUIRED_PIPELINE_SNIPPETS if snippet not in pipeline_text]
    if missing_pipeline:
        raise SystemExit(f"Pipeline missing beta 2D concept renderer snippets: {', '.join(missing_pipeline)}")

    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    missing_workflow = [snippet for snippet in REQUIRED_WORKFLOW_SNIPPETS if snippet not in workflow_text]
    if missing_workflow:
        raise SystemExit(f"Workflow missing beta 2D concept renderer wiring: {', '.join(missing_workflow)}")

    test_text = TEST_PATH.read_text(encoding="utf-8")
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in test_text]
    if missing_tests:
        raise SystemExit(f"Tests missing beta 2D concept renderer coverage: {', '.join(missing_tests)}")

    combined_text = "\n".join([doc_text, schema_text, pipeline_text, workflow_text, test_text]).lower()
    bad_claims = [claim for claim in DISALLOWED_CLAIMS if claim in combined_text]
    if bad_claims:
        raise SystemExit(f"Found disallowed beta renderer claims: {', '.join(bad_claims)}")

    print("Phase 6F-Design-F1 beta 2D concept renderer verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
