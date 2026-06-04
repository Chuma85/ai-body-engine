from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from scripts.verify_phase_3h_i_coupled_1000 import (
    DEFAULT_DATASET,
    DEFAULT_LABEL_MEASUREMENT_SCALE,
    DEFAULT_SAFE_FRAMING_SCALE,
    DEFAULT_SHAPE_KEY_RANGE,
    DEFAULT_SAMPLES,
    PHASE_3H_H_METRICS,
    build_generation_command,
    compare_to_phase_3h_h,
    ensure_not_archived_dataset,
    format_summary,
    validate_phase_3h_i_dataset,
)
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    LABEL_GENERATION_MODE,
    PHASE_3H_I_LABEL_FORMULA_VERSION,
    camera_set_name,
    generate_shape_key_coupled_measurements,
    shape_key_traceability_json,
)


def test_expected_dataset_path_and_generation_command() -> None:
    assert DEFAULT_DATASET == "data/synthetic/phase_3h_i_coupled_1000"

    command = build_generation_command(
        blender_executable="blender",
        dataset=DEFAULT_DATASET,
        samples=DEFAULT_SAMPLES,
        seed=42,
        blend_file="assets/body_meshes/base_body_scene.blend",
        overwrite=True,
    )

    assert "data/synthetic/phase_3h_i_coupled_1000" in command
    assert "--view-subdirs" in command
    assert "--shape-key-range" in command
    assert str(DEFAULT_SHAPE_KEY_RANGE) in command
    assert "--label-formula-version" in command
    assert PHASE_3H_I_LABEL_FORMULA_VERSION in command
    assert "--label-measurement-scale" in command
    assert str(DEFAULT_LABEL_MEASUREMENT_SCALE) in command
    assert "--safe-framing-scale" in command
    assert str(DEFAULT_SAFE_FRAMING_SCALE) in command
    assert "_archived_old_mannequin" not in " ".join(command)


def test_metadata_validation_and_view_folders(tmp_path: Path) -> None:
    dataset = _write_phase_3h_i_dataset(tmp_path, sample_count=4)

    summary = validate_phase_3h_i_dataset(dataset, expected_samples=4)

    assert summary["sample_count"] == 4
    assert summary["image_count"] == 12
    assert summary["label_generation_mode"] == LABEL_GENERATION_MODE
    assert summary["label_formula_version"] == PHASE_3H_I_LABEL_FORMULA_VERSION
    assert summary["synthetic_labels"] is True
    assert summary["real_world_validated"] is False
    assert summary["variation_source"] == "shape_keys_safe_range"
    assert summary["shape_key_count"] == 10
    assert summary["clipping"]["clipped_view_count"] == 0
    assert summary["view_sanity"]["passed"] is True
    assert summary["label_variation"]["variation_exists"] is True
    assert set(summary["view_folders"]) == {"front", "side", "back"}


def test_expected_sample_and_image_counts_if_dataset_exists(tmp_path: Path) -> None:
    dataset = _write_phase_3h_i_dataset(tmp_path, sample_count=3)

    with pytest.raises(ValueError, match="Expected 4 labels"):
        validate_phase_3h_i_dataset(dataset, expected_samples=4)

    assert validate_phase_3h_i_dataset(dataset, expected_samples=3)["image_count"] == 9


def test_verifier_behavior_when_dataset_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required Phase 3H-I dataset path"):
        validate_phase_3h_i_dataset(tmp_path / "missing_dataset", expected_samples=1)


def test_verifier_summary_format() -> None:
    summary = {
        "dataset": DEFAULT_DATASET,
        "sample_count": 1000,
        "image_count": 3000,
        "train_sample_count": 800,
        "test_sample_count": 200,
        "model_ranking": [{"model": "random_forest"}, {"model": "ridge"}],
        "audit": {"passed": True, "warnings_count": 0, "errors_count": 0, "flagged_sample_count": 0},
        "label_generation_mode": LABEL_GENERATION_MODE,
        "label_formula_version": PHASE_3H_I_LABEL_FORMULA_VERSION,
        "clipping": {"clipped_view_count": 0},
        "best_model": "random_forest",
        "overall_mean_mae": 0.7,
        "comparison_to_phase_3h_h": {
            "available": True,
            "overall_status": "improved",
            "overall_delta": -0.2,
            "mae_by_target": {
                target: {"delta": -0.1, "status": "improved"}
                for target in ("height_cm", "chest_cm", "waist_cm", "hip_cm", "shoulder_cm", "inseam_cm")
            },
        },
        "mae_by_target": {
            "height_cm": 1.0,
            "chest_cm": 0.8,
            "waist_cm": 0.9,
            "hip_cm": 0.7,
            "shoulder_cm": 0.4,
            "inseam_cm": 0.6,
        },
        "correlation_by_target": {
            target: {"abs_max_correlation": 0.5, "feature": "front_feature"}
            for target in ("height_cm", "chest_cm", "waist_cm", "hip_cm", "shoulder_cm", "inseam_cm")
        },
        "weak_targets_below_threshold": [],
        "label_variation_warnings": [],
    }

    text = format_summary(summary)

    assert "Phase 3H-I coupled verification passed." in text
    assert "Dataset: data/synthetic/phase_3h_i_coupled_1000" in text
    assert "Model candidates: random_forest, ridge" in text
    assert "Clipped views: 0" in text
    assert "Comparison vs Phase 3H-H: improved (-0.2000 cm)" in text
    assert "Weak targets below 0.25: none" in text


def test_archived_old_datasets_are_not_used() -> None:
    ensure_not_archived_dataset(DEFAULT_DATASET)

    with pytest.raises(ValueError, match="must not use archived old datasets"):
        ensure_not_archived_dataset("data/synthetic/_archived_old_mannequin/phase_3h_i_coupled_1000")


def test_phase_3h_h_comparison_when_prior_report_exists(tmp_path: Path) -> None:
    baseline = tmp_path / "phase_3h_h_metrics.json"
    baseline.write_text(
        json.dumps(
            {
                "best_model": "random_forest",
                "overall_mean_mae": 0.9413,
                "mae_by_target": {
                    "height_cm": 1.5,
                    "chest_cm": 0.9,
                    "waist_cm": 1.0,
                    "hip_cm": 0.8,
                    "shoulder_cm": 0.5,
                    "inseam_cm": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "overall_mean_mae": 0.8,
        "mae_by_target": {
            "height_cm": 1.2,
            "chest_cm": 0.8,
            "waist_cm": 0.9,
            "hip_cm": 0.7,
            "shoulder_cm": 0.4,
            "inseam_cm": 0.8,
        },
    }

    comparison = compare_to_phase_3h_h(metrics, baseline)

    assert comparison["available"] is True
    assert comparison["overall_status"] == "improved"
    assert comparison["mae_by_target"]["height_cm"]["status"] == "improved"
    assert PHASE_3H_H_METRICS.endswith("metrics.json")


def _write_phase_3h_i_dataset(tmp_path: Path, sample_count: int) -> Path:
    dataset = tmp_path / "phase_3h_i_coupled_1000"
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
            label_formula_version=PHASE_3H_I_LABEL_FORMULA_VERSION,
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
                "variation_source": "shape_keys_safe_range",
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
                "variation_source": "shape_keys_safe_range",
                "shape_key_count": 10,
                "label_generation_mode": LABEL_GENERATION_MODE,
                "label_formula_version": PHASE_3H_I_LABEL_FORMULA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _write_view_image(dataset: Path, sample_id: str, view: str, *, width: int, x_offset: int) -> None:
    image = Image.new("RGB", (128, 160), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_offset, 24, x_offset + width, 136], fill=(30, 30, 30))
    if view == "side":
        draw.rectangle([x_offset + 4, 48, x_offset + width - 4, 112], fill=(80, 80, 80))
    if view == "back":
        draw.rectangle([x_offset - 4, 34, x_offset + width + 2, 126], fill=(55, 55, 55))
    image.save(dataset / "images" / view / f"{sample_id}_{view}.png")
