from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import pytest

from scripts.verify_phase_3h_j_mobile_realism_1000 import (
    DEFAULT_BACKGROUND_JITTER,
    DEFAULT_BODY_ROTATION_JITTER,
    DEFAULT_CAMERA_HEIGHT_JITTER,
    DEFAULT_DATASET,
    DEFAULT_DISTANCE_JITTER,
    DEFAULT_LABEL_MEASUREMENT_SCALE,
    DEFAULT_PHONE_FRAMING_JITTER,
    DEFAULT_SAFE_FRAMING_SCALE,
    DEFAULT_SHAPE_KEY_RANGE,
    DEFAULT_SMOKE_DATASET,
    DEFAULT_SAMPLES,
    DEFAULT_LIGHTING_JITTER,
    PHASE_3H_I_METRICS,
    build_generation_command,
    ensure_not_archived_dataset,
    format_summary,
    main,
    validate_phase_3h_j_dataset,
    verify_coupled_1000,
)
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    LABEL_GENERATION_MODE,
    PHASE_3H_J_LABEL_FORMULA_VERSION,
    camera_set_name,
    generate_shape_key_coupled_measurements,
    shape_key_traceability_json,
)


def test_expected_dataset_path_and_generation_command() -> None:
    assert DEFAULT_DATASET == "data/synthetic/phase_3h_j_mobile_realism_1000"
    assert DEFAULT_SMOKE_DATASET == "data/synthetic/phase_3h_j_mobile_realism_smoke"

    command = build_generation_command(
        blender_executable="blender",
        dataset=DEFAULT_DATASET,
        samples=DEFAULT_SAMPLES,
        seed=42,
        blend_file="assets/body_meshes/base_body_scene.blend",
        overwrite=True,
    )

    assert "data/synthetic/phase_3h_j_mobile_realism_1000" in command
    assert "--mobile-realism" in command
    assert "--view-subdirs" in command
    assert "--shape-key-range" in command
    assert str(DEFAULT_SHAPE_KEY_RANGE) in command
    assert "--label-formula-version" in command
    assert PHASE_3H_J_LABEL_FORMULA_VERSION in command
    assert "--label-measurement-scale" in command
    assert str(DEFAULT_LABEL_MEASUREMENT_SCALE) in command
    assert "--safe-framing-scale" in command
    assert str(DEFAULT_SAFE_FRAMING_SCALE) in command
    assert "--distance-jitter" in command
    assert str(DEFAULT_DISTANCE_JITTER) in command
    assert "--camera-height-jitter" in command
    assert str(DEFAULT_CAMERA_HEIGHT_JITTER) in command
    assert "--body-rotation-jitter" in command
    assert str(DEFAULT_BODY_ROTATION_JITTER) in command
    assert "--lighting-jitter" in command
    assert str(DEFAULT_LIGHTING_JITTER) in command
    assert "--background-jitter" in command
    assert str(DEFAULT_BACKGROUND_JITTER) in command
    assert "--phone-framing-jitter" in command
    assert str(DEFAULT_PHONE_FRAMING_JITTER) in command
    assert "_archived_old_mannequin" not in " ".join(command)


def test_metadata_validation_and_view_folders(tmp_path: Path) -> None:
    dataset = _write_phase_3h_j_dataset(tmp_path, sample_count=4)

    summary = validate_phase_3h_j_dataset(dataset, expected_samples=4)

    assert summary["sample_count"] == 4
    assert summary["image_count"] == 12
    assert summary["label_generation_mode"] == LABEL_GENERATION_MODE
    assert summary["label_formula_version"] == PHASE_3H_J_LABEL_FORMULA_VERSION
    assert summary["synthetic_labels"] is True
    assert summary["real_world_validated"] is False
    assert summary["variation_source"] == "shape_keys_safe_range_plus_mobile_realism"
    assert summary["shape_key_count"] == 10
    assert summary["mobile_realism"] is True
    assert summary["mobile_realism_settings"]["distance_jitter"] == DEFAULT_DISTANCE_JITTER
    assert summary["clipping"]["clipped_view_count"] == 0
    assert summary["view_sanity"]["passed"] is True
    assert summary["label_variation"]["variation_exists"] is True
    assert set(summary["view_folders"]) == {"front", "side", "back"}


def test_clipping_failure_blocks_benchmark_gate(tmp_path: Path) -> None:
    dataset = _write_phase_3h_j_dataset(tmp_path, sample_count=2)
    _write_view_image(dataset, "sample_000001", "front", width=46, x_offset=42, touch_boundary=True)

    with pytest.raises(ValueError, match="do not run correlation or training"):
        validate_phase_3h_j_dataset(dataset, expected_samples=2)


def test_expected_sample_and_image_counts_if_dataset_exists(tmp_path: Path) -> None:
    dataset = _write_phase_3h_j_dataset(tmp_path, sample_count=3)

    with pytest.raises(ValueError, match="Expected 4 labels"):
        validate_phase_3h_j_dataset(dataset, expected_samples=4)

    assert validate_phase_3h_j_dataset(dataset, expected_samples=3)["image_count"] == 9


def test_verifier_summary_format_for_smoke_mode() -> None:
    summary = {
        "dataset": DEFAULT_SMOKE_DATASET,
        "sample_count": 25,
        "image_count": 75,
        "train_sample_count": None,
        "test_sample_count": None,
        "model_ranking": [],
        "audit": None,
        "label_generation_mode": LABEL_GENERATION_MODE,
        "label_formula_version": PHASE_3H_J_LABEL_FORMULA_VERSION,
        "mobile_realism": True,
        "mobile_realism_settings": _mobile_settings(),
        "clipping": {"clipped_view_count": 0},
        "best_model": None,
        "overall_mean_mae": None,
        "comparison_to_phase_3h_i": {"available": False, "reason": "Benchmark skipped."},
        "mae_by_target": {},
        "correlation_by_target": {},
        "weak_targets_below_threshold": [],
        "label_variation_warnings": [],
    }

    text = format_summary(summary)

    assert "Dataset: data/synthetic/phase_3h_j_mobile_realism_smoke" in text
    assert "Images: 75" in text
    assert "Mobile realism: True" in text
    assert "Clipped views: 0" in text
    assert "Overall mean MAE: skipped" in text
    assert "Comparison vs Phase 3H-I: unavailable (Benchmark skipped.)" in text


def test_generated_artifacts_remain_ignored() -> None:
    paths = [
        "data/synthetic/phase_3h_j_mobile_realism_1000/labels.csv",
        "data/synthetic/phase_3h_j_mobile_realism_smoke/images/front/sample_000001_front.png",
        "artifacts/phase_3h_j_mobile_realism_1000_audit/audit_report.json",
    ]
    result = subprocess.run(["git", "check-ignore", *paths], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    for path in paths:
        assert path in result.stdout


def test_no_render_mode_reuses_existing_dataset_without_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _write_phase_3h_j_dataset(tmp_path, sample_count=3)

    def fail_generation(*args: object, **kwargs: object) -> None:
        raise AssertionError("no-render mode must not generate Blender images")

    monkeypatch.setattr("scripts.verify_phase_3h_j_mobile_realism_1000.generate_batched_dataset", fail_generation)

    summary = verify_coupled_1000(
        dataset=str(dataset),
        samples=3,
        smoke=True,
        run_benchmark=False,
        no_render=True,
        blender_executable="definitely-not-needed",
    )

    assert summary["sample_count"] == 3
    assert summary["image_count"] == 9
    assert summary["commands"]["generation"] == ["no-render: reused existing dataset; Blender generation skipped"]


def test_reuse_existing_cli_alias_does_not_require_blender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_dataset = tmp_path / "missing_phase_3h_j"

    def fail_generation(*args: object, **kwargs: object) -> None:
        raise AssertionError("reuse-existing mode must not generate Blender images")

    monkeypatch.setattr("scripts.verify_phase_3h_j_mobile_realism_1000.generate_batched_dataset", fail_generation)
    monkeypatch.setattr("scripts.verify_phase_3h_j_mobile_realism_1000.discover_blender_executable", fail_generation)

    assert main(["--reuse-existing", "--dataset", str(missing_dataset), "--samples", "3"]) == 1


def test_archived_old_datasets_are_not_used() -> None:
    ensure_not_archived_dataset(DEFAULT_DATASET)

    with pytest.raises(ValueError, match="must not use archived old datasets"):
        ensure_not_archived_dataset("data/synthetic/_archived_old_mannequin/phase_3h_j_mobile_realism_1000")
    assert PHASE_3H_I_METRICS.endswith("metrics.json")


def _write_phase_3h_j_dataset(tmp_path: Path, sample_count: int) -> Path:
    dataset = tmp_path / "phase_3h_j_mobile_realism_1000"
    for view in ("front", "side", "back"):
        (dataset / "images" / view).mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(1, sample_count + 1):
        sample_id = f"sample_{index:06d}"
        shape_key_values = {
            "BodyHeight": round(0.02 * index, 4),
            "ChestWidth": round(0.01 * index, 4),
            "HipWidth": round(0.015 * index, 4),
        }
        payload = generate_shape_key_coupled_measurements(
            sample_id=sample_id,
            seed=42,
            shape_key_values=shape_key_values,
            shape_key_range=DEFAULT_SHAPE_KEY_RANGE,
            label_formula_version=PHASE_3H_J_LABEL_FORMULA_VERSION,
            label_measurement_scale=DEFAULT_LABEL_MEASUREMENT_SCALE,
        )
        _write_view_image(dataset, sample_id, "front", width=44 + index, x_offset=42)
        _write_view_image(dataset, sample_id, "side", width=24 + index, x_offset=54)
        _write_view_image(dataset, sample_id, "back", width=40 + index, x_offset=36)

        measurements = payload["measurements"]
        factors = payload["factors"]
        rows.append(
            {
                "sample_id": sample_id,
                "front_image": f"images/front/{sample_id}_front.png",
                "side_image": f"images/side/{sample_id}_side.png",
                "back_image": f"images/back/{sample_id}_back.png",
                "height_cm": measurements["height_cm"],
                "chest_cm": measurements["chest_cm"],
                "waist_cm": measurements["waist_cm"],
                "hip_cm": measurements["hip_cm"],
                "shoulder_cm": measurements["shoulder_cm"],
                "inseam_cm": measurements["inseam_cm"],
                "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                "variation_source": "shape_keys_safe_range_plus_mobile_realism",
                "camera_set": camera_set_name(),
                "seed": 42,
                "label_source": "shape_key_coupled_synthetic_formula",
                "synthetic_labels": "true",
                "real_world_validated": "false",
                "label_generation_mode": LABEL_GENERATION_MODE,
                "height_factor": factors["height_factor"],
                "chest_factor": factors["chest_factor"],
                "waist_factor": factors["waist_factor"],
                "hip_factor": factors["hip_factor"],
                "shoulder_factor": factors["shoulder_factor"],
                "inseam_factor": factors["inseam_factor"],
                "torso_width_factor": factors["torso_width_factor"],
                "leg_length_factor": factors["leg_length_factor"],
                "shape_key_values_json": shape_key_traceability_json(shape_key_values),
                "body_shape_profile_id": payload["body_shape_profile_id"],
            }
        )

    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (dataset / "metadata.json").write_text(
        json.dumps(
            {
                "generator_version": "phase_3h_blend_dataset_v1",
                "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                "camera_set": camera_set_name(),
                "sample_count": sample_count,
                "seed": 42,
                "synthetic_labels": True,
                "real_world_validated": False,
                "variation_source": "shape_keys_safe_range_plus_mobile_realism",
                "shape_key_count": 10,
                "label_generation_mode": LABEL_GENERATION_MODE,
                "label_formula_version": PHASE_3H_J_LABEL_FORMULA_VERSION,
                "mobile_realism": True,
                "mobile_realism_settings": _mobile_settings(),
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _mobile_settings() -> dict[str, float | bool]:
    return {
        "mobile_realism": True,
        "distance_jitter": DEFAULT_DISTANCE_JITTER,
        "camera_height_jitter": DEFAULT_CAMERA_HEIGHT_JITTER,
        "body_rotation_jitter": DEFAULT_BODY_ROTATION_JITTER,
        "lighting_jitter": DEFAULT_LIGHTING_JITTER,
        "background_jitter": DEFAULT_BACKGROUND_JITTER,
        "phone_framing_jitter": DEFAULT_PHONE_FRAMING_JITTER,
    }


def _write_view_image(
    dataset: Path,
    sample_id: str,
    view: str,
    *,
    width: int,
    x_offset: int,
    touch_boundary: bool = False,
) -> None:
    image = Image.new("RGB", (128, 160), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    top = 0 if touch_boundary else 24
    draw.rectangle([x_offset, top, x_offset + width, 136], fill=(30, 30, 30))
    if view == "side":
        draw.rectangle([x_offset + 4, max(top + 24, 24), x_offset + width - 4, 112], fill=(80, 80, 80))
    if view == "back":
        draw.rectangle([x_offset - 4, max(top + 10, 10), x_offset + width + 2, 126], fill=(55, 55, 55))
    image.save(dataset / "images" / view / f"{sample_id}_{view}.png")
