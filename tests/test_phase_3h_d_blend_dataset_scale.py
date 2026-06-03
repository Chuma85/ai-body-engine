from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_phase_3h_d_blend_dataset_scale import (
    build_audit_command,
    build_generation_command,
    dataset_complete,
    discover_blender_executable,
    required_outputs_exist,
    summarize_verification,
)


def test_generation_command_includes_blend_source_and_sample_count() -> None:
    command = build_generation_command(
        blender_executable="blender",
        dataset="data/synthetic/phase_3h_blend_250",
        samples=250,
        seed=42,
        blend_file="assets/body_meshes/base_body_scene.blend",
    )

    assert "scripts/generate_blend_dataset.py" in command
    assert command[command.index("--source") + 1] == "blend"
    assert command[command.index("--blend-file") + 1] == "assets/body_meshes/base_body_scene.blend"
    assert command[command.index("--out") + 1] == "data/synthetic/phase_3h_blend_250"
    assert command[command.index("--samples") + 1] == "250"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--blender-executable") + 1] == "blender"


def test_generation_command_can_request_overwrite() -> None:
    command = build_generation_command(
        blender_executable="blender",
        dataset="data/synthetic/phase_3h_blend_250",
        samples=250,
        seed=42,
        blend_file="assets/body_meshes/base_body_scene.blend",
        overwrite=True,
    )

    assert "--overwrite" in command


def test_audit_command_uses_strict_expected_samples() -> None:
    command = build_audit_command(
        dataset="data/synthetic/phase_3h_blend_250",
        audit_out="artifacts/phase_3h_blend_250_audit",
        samples=250,
    )

    assert "scripts/audit_blend_dataset.py" in command
    assert command[command.index("--expected-samples") + 1] == "250"
    assert "--strict" in command


def test_dataset_complete_checks_labels_metadata_and_image_count(tmp_path: Path) -> None:
    dataset = tmp_path / "phase_3h_blend_250"
    images = dataset / "images"
    images.mkdir(parents=True)
    (dataset / "labels.csv").write_text("sample_id\n", encoding="utf-8")
    (dataset / "metadata.json").write_text("{}\n", encoding="utf-8")
    for index in range(6):
        (images / f"image_{index}.png").write_bytes(b"png")

    complete, problems = dataset_complete(dataset, samples=2)

    assert complete is True
    assert problems == []


def test_dataset_complete_reports_missing_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "phase_3h_blend_250"
    dataset.mkdir()

    complete, problems = dataset_complete(dataset, samples=250)

    assert complete is False
    assert any("labels.csv" in problem for problem in problems)
    assert any("metadata.json" in problem for problem in problems)
    assert any("expected 750 PNG images" in problem for problem in problems)


def test_required_outputs_include_dataset_and_audit_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    audit = tmp_path / "audit"
    dataset.mkdir()
    audit.mkdir()
    for filename in ("labels.csv", "metadata.json"):
        (dataset / filename).write_text("", encoding="utf-8")
    for filename in (
        "audit_report.json",
        "audit_summary.md",
        "sample_contact_sheet.png",
        "label_distribution_summary.csv",
        "flagged_samples.csv",
    ):
        (audit / filename).write_text("", encoding="utf-8")

    complete, missing = required_outputs_exist(dataset, audit)

    assert complete is True
    assert missing == []


def test_summarize_verification_reads_audit_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    audit = tmp_path / "audit"
    images = dataset / "images"
    images.mkdir(parents=True)
    audit.mkdir()
    for index in range(6):
        (images / f"image_{index}.png").write_bytes(b"png")
    report = {
        "row_count": 2,
        "passed": True,
        "warnings": [],
        "errors": [],
        "strict_failures": [],
        "flagged_sample_count": 0,
        "view_sanity": {"passed": True},
        "label_audit": {"variation_exists": True},
        "metadata": {"variation_source": "shape_keys_safe_range", "shape_key_count": 10},
    }
    (audit / "audit_report.json").write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_verification(
        dataset=dataset,
        audit_out=audit,
        samples=2,
        duration_seconds=12.345,
        generated=True,
    )

    assert summary["sample_count"] == 2
    assert summary["actual_image_count"] == 6
    assert summary["expected_image_count"] == 6
    assert summary["strict_audit_passed"] is True
    assert summary["variation_source"] == "shape_keys_safe_range"
    assert summary["shape_key_count"] == 10
    assert summary["duration_seconds"] == 12.35


def test_discover_blender_executable_rejects_missing_explicit_path() -> None:
    assert discover_blender_executable("definitely_missing_blender_executable") is None
