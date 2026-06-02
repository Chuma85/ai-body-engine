from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_3t_b3_back_view_realistic_rendering_fix.md"
WRAPPER_PATH = ROOT / "scripts/generate_phase_3t_enhanced_back_view.py"
CONFIG_PATH = ROOT / "synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json"
PY_GENERATOR_PATH = ROOT / "synthetic/generator/generate_dataset.py"
BLENDER_RENDERER_PATH = ROOT / "synthetic/blender/scripts/render_parametric_body.py"
VALIDATOR_PATH = ROOT / "synthetic/validate_synthetic_dataset.py"
MANIFEST_PATH = ROOT / "synthetic/build_dataset_manifest.py"
TEST_PATHS = [
    ROOT / "tests/test_phase_3t_enhanced_back_view_dataset.py",
    ROOT / "tests/test_validate_synthetic_dataset.py",
    ROOT / "tests/test_build_dataset_manifest.py",
]


REQUIRED_DOC_SNIPPETS = [
    "crude and blocky",
    "smoke-only local artifacts",
    "realistic Blender/body mesh rendering path",
    "data/synthetic/phase_3t_enhanced",
    "render_source=blender_body_mesh",
    "quality_tier=training_candidate",
    "renderer_mode=lightweight_smoke",
    "quality_tier=smoke_only",
    "python scripts\\generate_phase_3t_enhanced_back_view.py --overwrite",
    "--smoke-lightweight",
    "No real-world accuracy improvement",
]

REQUIRED_WRAPPER_SNIPPETS = [
    "REALISTIC_MODE = \"blender_realistic\"",
    "SMOKE_MODE = \"lightweight_smoke\"",
    "generate_realistic_blender_dataset",
    "generate_lightweight_smoke_dataset",
    "build_realistic_blender_command",
    "build_blender_command",
    "--smoke-lightweight",
    "--dry-run",
    "include_back_view=True",
    "require_realistic=True",
    "Blender executable was not found",
]

REQUIRED_CONFIG_SNIPPETS = [
    "\"output_dir\": \"data/synthetic/phase_3t_enhanced\"",
    "\"views\": [\"front\", \"side\", \"back\"]",
    "\"base_mesh\"",
    "\"mesh_deformation\"",
    "\"render_realism\"",
    "\"asset_path\": \"assets/body_meshes/base_human_rigged.fbx\"",
]

REQUIRED_METADATA_SNIPPETS = [
    "renderer_mode",
    "render_source",
    "is_smoke_dataset",
    "is_training_candidate",
    "quality_tier",
]

REQUIRED_VALIDATOR_SNIPPETS = [
    "require_realistic",
    "label_row_is_realistic_training_candidate",
    "non_realistic_label_rows",
    "python_silhouette_placeholder",
    "blender_body_mesh",
    "training_candidate",
]

REQUIRED_TEST_SNIPPETS = [
    "test_realistic_blender_mode_is_default_generation_route",
    "test_realistic_blender_command_includes_front_side_back_config_and_output",
    "test_realistic_validation_rejects_lightweight_smoke_output",
    "test_realistic_validation_accepts_blender_training_candidate_metadata",
    "test_manifest_can_require_realistic_training_candidate_metadata",
    "test_manifest_rejects_smoke_metadata_when_realistic_required",
    "test_legacy_front_side_dataset_still_valid_without_back_view",
]

DISALLOWED_CLAIMS = [
    "real-world accuracy improvement is guaranteed",
    "front + side is no longer supported",
    "back view is mandatory",
]


def main() -> int:
    required_files = [DOC_PATH, WRAPPER_PATH, CONFIG_PATH, PY_GENERATOR_PATH, BLENDER_RENDERER_PATH, VALIDATOR_PATH, MANIFEST_PATH, *TEST_PATHS]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing Phase 3T-B3 files: {', '.join(missing)}")

    _require_snippets(DOC_PATH, REQUIRED_DOC_SNIPPETS, "Phase 3T-B3 doc")
    _require_snippets(WRAPPER_PATH, REQUIRED_WRAPPER_SNIPPETS, "enhanced generation wrapper")
    _require_snippets(CONFIG_PATH, REQUIRED_CONFIG_SNIPPETS, "enhanced Blender config")
    _require_snippets(PY_GENERATOR_PATH, REQUIRED_METADATA_SNIPPETS, "lightweight generator metadata")
    _require_snippets(BLENDER_RENDERER_PATH, REQUIRED_METADATA_SNIPPETS, "Blender renderer metadata")
    _require_snippets(VALIDATOR_PATH, REQUIRED_VALIDATOR_SNIPPETS, "dataset validator")
    _require_snippets(MANIFEST_PATH, ["require_realistic"], "manifest builder")

    tests_text = "\n".join(path.read_text(encoding="utf-8") for path in TEST_PATHS)
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in tests_text]
    if missing_tests:
        raise SystemExit(f"Phase 3T-B3 tests missing snippets: {', '.join(missing_tests)}")

    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in required_files)
    bad_claims = [claim for claim in DISALLOWED_CLAIMS if claim.lower() in combined_text.lower()]
    if bad_claims:
        raise SystemExit(f"Found disallowed claims: {', '.join(bad_claims)}")

    tracked_generated = tracked_generated_enhanced_files()
    if tracked_generated:
        raise SystemExit(
            "Generated phase_3t_enhanced files appear tracked: " + ", ".join(tracked_generated[:20])
        )

    print("Phase 3T-B3 back-view realistic rendering fix verification passed.")
    return 0


def _require_snippets(path: Path, snippets: list[str], label: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in snippets if snippet.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"{label} missing snippets: {', '.join(missing)}")


def tracked_generated_enhanced_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "data/synthetic/phase_3t_enhanced"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        path
        for path in result.stdout.splitlines()
        if path.endswith((".png", ".csv", ".json", ".jpg", ".jpeg"))
    ]


if __name__ == "__main__":
    raise SystemExit(main())
