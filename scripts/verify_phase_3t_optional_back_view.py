from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_3t_b_optional_back_view_synthetic_generation.md"
PY_GENERATOR_PATH = ROOT / "synthetic/generator/generate_dataset.py"
BLENDER_RENDERER_PATH = ROOT / "synthetic/blender/scripts/render_parametric_body.py"
VALIDATOR_PATH = ROOT / "synthetic/validate_synthetic_dataset.py"
MANIFEST_PATH = ROOT / "synthetic/build_dataset_manifest.py"
DATASET_LOADER_PATH = ROOT / "training/datasets/synthetic_body_dataset.py"
PHASE_3T_BACK_CONFIG_PATH = ROOT / "synthetic/blender/configs/phase_3t_optional_back_view_config.example.json"

TEST_PATHS = [
    ROOT / "tests/test_synthetic_generator.py",
    ROOT / "tests/test_validate_synthetic_dataset.py",
    ROOT / "tests/test_build_dataset_manifest.py",
    ROOT / "tests/test_synthetic_body_dataset.py",
    ROOT / "tests/test_blender_pipeline_scaffold.py",
]


REQUIRED_DOC_SNIPPETS = [
    "Add optional back-view synthetic generation",
    "Front + side remains the minimum scan set",
    "Front + side + back becomes the enhanced scan set",
    "data/synthetic/phase_3t/",
    "images/back/",
    "sample_000001_back.png",
    "same body sample",
    "back_image_path",
    "has_back",
    "capture_views",
    "minimum_scan_views=front,side",
    "enhanced_scan_views=front,side,back",
    "Back view is optional",
    "No real-world accuracy claim",
]

REQUIRED_GENERATOR_SNIPPETS = [
    "render_back_silhouette",
    "include_back_view",
    "--include-back-view",
    "back_image_path",
    "has_back",
    "capture_views",
    "front,side,back",
]

REQUIRED_RENDERER_SNIPPETS = [
    "views = list(config.get(\"views\") or [\"front\", \"side\"])",
    "unsupported = sorted(set(views) - {\"front\", \"side\", \"back\"})",
    "back_path = view_paths.get(\"back\")",
    "has_back",
    "capture_views",
    "minimum_scan_views",
    "enhanced_scan_views",
    "if view == \"back\"",
]

REQUIRED_VALIDATION_SNIPPETS = [
    "require_back",
    "label_rows_missing_back_images",
    "Back view is optional",
    "has_back",
    "back_image_path",
]

REQUIRED_TEST_SNIPPETS = [
    "test_generate_tiny_synthetic_dataset_with_optional_back_view",
    "test_missing_back_image_fails_when_label_declares_back_view",
    "test_minimum_front_side_dataset_does_not_require_back_view",
    "test_manifest_includes_optional_back_view_metadata_when_available",
    "test_optional_back_view_is_loaded_when_manifest_includes_it",
    "test_phase_3t_b_optional_back_view_config_loads_and_validates",
    "test_phase_3t_b_label_row_marks_enhanced_capture_views",
]

DISALLOWED_CLAIMS = [
    "real-world accuracy improvement is guaranteed",
    "back view is mandatory",
    "front + side is no longer supported",
]


def main() -> int:
    required_files = [
        DOC_PATH,
        PY_GENERATOR_PATH,
        BLENDER_RENDERER_PATH,
        VALIDATOR_PATH,
        MANIFEST_PATH,
        DATASET_LOADER_PATH,
        PHASE_3T_BACK_CONFIG_PATH,
        *TEST_PATHS,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing Phase 3T-B files: {', '.join(missing)}")

    _require_snippets(DOC_PATH, REQUIRED_DOC_SNIPPETS, "Phase 3T-B doc")
    _require_snippets(PY_GENERATOR_PATH, REQUIRED_GENERATOR_SNIPPETS, "Python generator")
    _require_snippets(BLENDER_RENDERER_PATH, REQUIRED_RENDERER_SNIPPETS, "Blender renderer")
    _require_snippets(VALIDATOR_PATH, REQUIRED_VALIDATION_SNIPPETS, "dataset validator")
    _require_snippets(MANIFEST_PATH, ["has_back", "capture_views", "enhanced_scan_views"], "manifest builder")
    _require_snippets(DATASET_LOADER_PATH, ["back_image_path", "has_back", "capture_views"], "dataset loader")
    _require_snippets(PHASE_3T_BACK_CONFIG_PATH, ["phase_3t_b_optional_back_view_smoke_v1", "\"views\": [\"front\", \"side\", \"back\"]"], "Phase 3T-B config")

    tests_text = "\n".join(path.read_text(encoding="utf-8") for path in TEST_PATHS)
    missing_tests = [snippet for snippet in REQUIRED_TEST_SNIPPETS if snippet not in tests_text]
    if missing_tests:
        raise SystemExit(f"Phase 3T-B tests missing snippets: {', '.join(missing_tests)}")

    combined_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [DOC_PATH, PY_GENERATOR_PATH, BLENDER_RENDERER_PATH, VALIDATOR_PATH, MANIFEST_PATH, DATASET_LOADER_PATH, *TEST_PATHS]
    ).lower()
    bad_claims = [claim for claim in DISALLOWED_CLAIMS if claim in combined_text]
    if bad_claims:
        raise SystemExit(f"Found disallowed optional-back claims: {', '.join(bad_claims)}")

    tracked_generated = tracked_phase_3t_generated_files()
    if tracked_generated:
        raise SystemExit(
            "Generated Phase 3T image/data files appear tracked: " + ", ".join(tracked_generated[:20])
        )

    print("Phase 3T-B optional back-view synthetic generation verification passed.")
    return 0


def _require_snippets(path: Path, snippets: list[str], label: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in snippets if snippet.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"{label} missing snippets: {', '.join(missing)}")


def tracked_phase_3t_generated_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "data/synthetic/phase_3t"],
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
