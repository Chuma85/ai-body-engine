from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "docs/phase_6f_design_agent_virtual_fitting_foundation.md",
    "app/schemas/design_agent_virtual_fitting.py",
    "app/services/design_agent_virtual_fitting.py",
    "app/api/routes/design_agent_virtual_fitting.py",
    "tests/test_design_agent_virtual_fitting_foundation.py",
]

REQUIRED_SCHEMA_NAMES = [
    "DesignSession",
    "DesignPreferenceInput",
    "GeneratedDesignOption",
    "DesignRefinementRequest",
    "BodyProfileSnapshot",
    "VirtualFittingRequest",
    "VirtualFittingResult",
    "ProductionBrief",
    "ApprovalState",
    "DesignSessionStatus",
]

REQUIRED_ENDPOINT_SNIPPETS = [
    '"/v1/body-ai/design-sessions"',
    '"/{design_session_id}/generate"',
    '"/{design_session_id}/refine"',
    '"/{design_session_id}/fitting-preview"',
    '"/{design_session_id}/approve"',
    '"/{design_session_id}"',
    '"/{design_session_id}/production-brief"',
]

REQUIRED_TEST_SNIPPETS = [
    "test_design_session_can_be_created_from_measurement_context",
    "test_design_generation_returns_structured_options",
    "test_refinement_adds_new_variation",
    "test_fitting_preview_returns_beta_result_contract",
    "test_production_brief_requires_approval_and_contains_handoff_contract",
    "test_measurement_result_contract_remains_compatible_with_design_session",
]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    schema_text = (ROOT / "app/schemas/design_agent_virtual_fitting.py").read_text(encoding="utf-8")
    missing_schemas = [name for name in REQUIRED_SCHEMA_NAMES if f"class {name}" not in schema_text]
    if missing_schemas:
        raise SystemExit(f"Missing required schema classes: {', '.join(missing_schemas)}")

    route_text = (ROOT / "app/api/routes/design_agent_virtual_fitting.py").read_text(encoding="utf-8")
    missing_routes = [snippet for snippet in REQUIRED_ENDPOINT_SNIPPETS if snippet not in route_text]
    if missing_routes:
        raise SystemExit(f"Missing required route snippets: {', '.join(missing_routes)}")

    test_text = (ROOT / "tests/test_design_agent_virtual_fitting_foundation.py").read_text(encoding="utf-8")
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in test_text]
    if missing_tests:
        raise SystemExit(f"Missing required test coverage snippets: {', '.join(missing_tests)}")

    doc_text = (ROOT / "docs/phase_6f_design_agent_virtual_fitting_foundation.md").read_text(encoding="utf-8")
    for required in ("Product Flow", "Beta Safety Language", "Target API Flow", "CUSTOM-FASHION-MARKETPLACE"):
        if required not in doc_text:
            raise SystemExit(f"Documentation missing required section/content: {required}")

    print("Phase 6F-Design-A foundation verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

