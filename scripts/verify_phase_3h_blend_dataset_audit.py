from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs/phase_3h_c_blend_dataset_audit.md"
CLI_PATH = ROOT / "scripts/audit_blend_dataset.py"
AUDIT_PATH = ROOT / "synthetic/blender/blend_dataset_audit.py"
TEST_PATH = ROOT / "tests/test_blend_dataset_audit.py"

REQUIRED_SNIPPETS = {
    DOC_PATH: [
        "sample_contact_sheet.png",
        "label_distribution_summary.csv",
        "flagged_samples.csv",
        "variation_source=shape_keys_safe_range",
        "variation_source=static_blend_mesh",
        "--strict",
        "Do not train",
    ],
    CLI_PATH: [
        "--dataset",
        "--out",
        "--expected-samples",
        "--max-contact-sheet-samples",
        "--strict",
        "audit_blend_dataset",
    ],
    AUDIT_PATH: [
        "audit_report.json",
        "audit_summary.md",
        "sample_contact_sheet.png",
        "label_distribution_summary.csv",
        "flagged_samples.csv",
        "view_difference_score",
        "shape_keys_safe_range",
        "static_blend_mesh",
        "IMPORTANT_MEASUREMENT_COLUMNS",
    ],
    TEST_PATH: [
        "test_audit_fails_cleanly_when_dataset_folder_missing",
        "test_audit_fails_cleanly_when_labels_missing",
        "test_label_schema_validation_reports_missing_columns",
        "test_audit_detects_label_variation",
        "test_audit_writes_requested_outputs_for_small_fake_dataset",
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

    print("Phase 3H-C blend dataset audit verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
