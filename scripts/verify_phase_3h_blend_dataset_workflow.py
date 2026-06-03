from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs/phase_3h_b_blend_file_dataset_generation.md"
WRAPPER_PATH = ROOT / "scripts/generate_blend_dataset.py"
BLEND_HELPER_PATH = ROOT / "synthetic/blender/blend_dataset.py"
BLENDER_SCRIPT_PATH = ROOT / "synthetic/blender/scripts/render_blend_dataset.py"
TEST_PATH = ROOT / "tests/test_blend_dataset_workflow.py"
ASSET_README_PATH = ROOT / "assets/body_meshes/README.md"

REQUIRED_SNIPPETS = {
    DOC_PATH: [
        "assets/body_meshes/base_body_scene.blend",
        "FrontCam",
        "SideCam",
        "BackCam",
        "synthetic_labels",
        "real_world_validated",
        "variation_source=static_blend_mesh",
    ],
    WRAPPER_PATH: [
        "generate_blend_dataset",
        "--source",
        "--blend-file",
        "validate_generated_blend_dataset",
        "Blender executable was not found",
    ],
    BLEND_HELPER_PATH: [
        "DEFAULT_BLEND_FILE",
        "build_blend_blender_command",
        "BLEND_LABEL_COLUMNS",
        "validate_blend_file_exists",
        "validate_generated_blend_dataset",
    ],
    BLENDER_SCRIPT_PATH: [
        "bpy.ops.wm.open_mainfile",
        "find_required_cameras",
        "FrontCam",
        "shape_keys_safe_range",
        "static_blend_mesh",
        "SYNTHETIC_LABEL_SOURCE",
    ],
    TEST_PATH: [
        "test_blend_dataset_command_includes_source_blend_file_and_cameras",
        "test_blend_dataset_dry_run_validates_config_without_blender",
        "test_missing_blend_file_fails_with_helpful_message",
        "test_validate_blend_dataset_output_structure_and_schema",
        "test_render_blend_dataset_imports_without_bpy",
    ],
    ASSET_README_PATH: [
        "Phase 3H-B Blend Scene Guidance",
        "base_body_scene.blend",
        "variation_source=static_blend_mesh",
    ],
}


def main() -> int:
    missing: list[str] = []
    for path, snippets in REQUIRED_SNIPPETS.items():
        if not path.exists():
            missing.append(f"Missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                missing.append(f"{path.relative_to(ROOT)} missing snippet: {snippet}")

    if missing:
        for item in missing:
            print(item)
        return 1

    print("Phase 3H-B blend-file dataset workflow verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
