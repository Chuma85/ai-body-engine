from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

DOC_PATH = ROOT / "docs/phase_3t_b2_generate_enhanced_back_view_dataset.md"
WRAPPER_PATH = ROOT / "scripts/generate_phase_3t_enhanced_back_view.py"
VERIFY_B_PATH = ROOT / "scripts/verify_phase_3t_optional_back_view.py"
CONFIG_PATH = ROOT / "synthetic/blender/configs/phase_3t_enhanced_back_view_config.example.json"
VALIDATOR_PATH = ROOT / "synthetic/validate_synthetic_dataset.py"
MANIFEST_PATH = ROOT / "synthetic/build_dataset_manifest.py"
GITIGNORE_PATH = ROOT / ".gitignore"
TEST_PATH = ROOT / "tests/test_phase_3t_enhanced_back_view_dataset.py"


REQUIRED_DOC_SNIPPETS = [
    "Phase 3T-B2 creates a repeatable path",
    "data/synthetic/phase_3t_enhanced",
    "Keep `data/synthetic/phase_3t` intact",
    "sample_000001_front.png",
    "sample_000001_side.png",
    "sample_000001_back.png",
    "same body parameters",
    "python scripts\\generate_phase_3t_enhanced_back_view.py --overwrite",
    "python -m synthetic.validate_synthetic_dataset --dataset data\\synthetic\\phase_3t_enhanced",
    "python -m synthetic.build_dataset_manifest --dataset data\\synthetic\\phase_3t_enhanced --require-back",
    "scripts/configs/docs/source/tests",
    "do not commit large generated image datasets",
    "front + side remains the minimum legacy scan set",
]

REQUIRED_WRAPPER_SNIPPETS = [
    "DEFAULT_OUTPUT_DIR = \"data/synthetic/phase_3t_enhanced\"",
    "DEFAULT_SAMPLE_COUNT = 1000",
    "include_back_view=True",
    "build_dataset_manifest(output_path, require_back=True)",
    "validate_dataset(output_path, require_back=True)",
    "sample_alignment_summary",
    "--overwrite",
    "--smoke",
    "FileExistsError",
]

REQUIRED_CONFIG_SNIPPETS = [
    "\"output_dir\": \"data/synthetic/phase_3t_enhanced\"",
    "\"views\": [\"front\", \"side\", \"back\"]",
    "\"generator_version\": \"phase_3t_b2_enhanced_back_view_v1\"",
]

REQUIRED_TEST_SNIPPETS = [
    "test_phase_3t_enhanced_config_targets_back_view_dataset",
    "test_enhanced_wrapper_generates_smoke_dataset_with_aligned_views",
    "test_enhanced_wrapper_manifest_includes_back_view_metadata",
    "test_enhanced_wrapper_requires_overwrite_for_existing_output",
    "test_legacy_front_side_dataset_still_valid_without_back_view",
    "sample_000001_back.png",
]

DISALLOWED_SNIPPETS = [
    "C:\\Users\\",
    "real-world accuracy improvement is guaranteed",
    "front + side is no longer supported",
    "back view is mandatory",
]


def main() -> int:
    required_files = [
        DOC_PATH,
        WRAPPER_PATH,
        VERIFY_B_PATH,
        CONFIG_PATH,
        VALIDATOR_PATH,
        MANIFEST_PATH,
        GITIGNORE_PATH,
        TEST_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing Phase 3T-B2 files: {', '.join(missing)}")

    _require_snippets(DOC_PATH, REQUIRED_DOC_SNIPPETS, "Phase 3T-B2 doc")
    _require_snippets(WRAPPER_PATH, REQUIRED_WRAPPER_SNIPPETS, "enhanced generation wrapper")
    _require_snippets(CONFIG_PATH, REQUIRED_CONFIG_SNIPPETS, "enhanced Blender config")
    _require_snippets(VALIDATOR_PATH, ["require_back", "label_rows_missing_back_images"], "dataset validator")
    _require_snippets(MANIFEST_PATH, ["has_back", "capture_views", "require_back"], "manifest builder")
    _require_snippets(GITIGNORE_PATH, ["data/synthetic/phase_3t_enhanced/"], ".gitignore")
    _require_snippets(TEST_PATH, REQUIRED_TEST_SNIPPETS, "Phase 3T-B2 tests")

    combined_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [DOC_PATH, WRAPPER_PATH, CONFIG_PATH, VALIDATOR_PATH, MANIFEST_PATH, GITIGNORE_PATH, TEST_PATH]
    )
    bad_snippets = [snippet for snippet in DISALLOWED_SNIPPETS if snippet.lower() in combined_text.lower()]
    if bad_snippets:
        raise SystemExit(f"Found disallowed local path or claim snippets: {', '.join(bad_snippets)}")

    tracked_generated = tracked_generated_enhanced_files()
    if tracked_generated:
        raise SystemExit(
            "Generated phase_3t_enhanced files appear tracked: " + ", ".join(tracked_generated[:20])
        )

    print("Phase 3T-B2 enhanced back-view dataset generation verification passed.")
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
