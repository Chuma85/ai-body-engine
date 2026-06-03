from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.train_blend_dataset_baseline import extract_blend_image_features
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    BODY_FACTOR_COLUMNS,
    LABEL_FORMULA_VERSION,
    LABEL_GENERATION_MODE,
    SHAPE_KEY_COUPLED_LABEL_SOURCE,
    blend_generation_config,
    camera_set_name,
    derive_body_factors_from_shape_keys,
    generate_shape_key_coupled_measurements,
    shape_key_label_metadata,
    shape_key_traceability_json,
    validate_factor_label_correlations,
    validate_generated_blend_dataset,
    validate_shape_key_coupled_rows,
)


SHAPE_KEYS_A = {
    "$md-$af-$fe-$yn": 0.02,
    "$md-$af-$ma-$yn": 0.11,
    "$md-$as-$fe-$yn": 0.04,
    "$md-$ca-$ma-$yn": 0.13,
}
SHAPE_KEYS_B = {
    "$md-$af-$fe-$yn": 0.14,
    "$md-$af-$ma-$yn": 0.02,
    "$md-$as-$fe-$yn": 0.12,
    "$md-$ca-$ma-$yn": 0.01,
}


def test_deterministic_factor_derivation_from_shape_keys() -> None:
    first = derive_body_factors_from_shape_keys(SHAPE_KEYS_A, shape_key_range=0.15)
    second = derive_body_factors_from_shape_keys(SHAPE_KEYS_A, shape_key_range=0.15)

    assert first == second
    assert set(first) == set(BODY_FACTOR_COLUMNS)
    assert all(-1.0 <= value <= 1.0 for value in first.values())
    assert first != derive_body_factors_from_shape_keys(SHAPE_KEYS_B, shape_key_range=0.15)


def test_measurement_labels_change_when_body_factors_change() -> None:
    low = generate_shape_key_coupled_measurements(
        sample_id="sample_000001",
        seed=42,
        shape_key_values={name: 0.01 for name in SHAPE_KEYS_A},
        label_noise_cm=0.0,
    )
    high = generate_shape_key_coupled_measurements(
        sample_id="sample_000001",
        seed=42,
        shape_key_values={name: 0.14 for name in SHAPE_KEYS_A},
        label_noise_cm=0.0,
    )

    assert high["measurements"] != low["measurements"]
    assert high["label_generation_mode"] == LABEL_GENERATION_MODE
    assert high["label_formula_version"] == LABEL_FORMULA_VERSION


def test_height_inseam_and_torso_measurements_are_coupled() -> None:
    rows = []
    for index in range(1, 30):
        value = 0.01 + index * 0.004
        shape_keys = {
            "$md-$af-$fe-$yn": value,
            "$md-$af-$ma-$yn": value,
            "$md-$ca-$ma-$yn": value,
            "$md-universal-$ma-$yn-$av$mu-$av$wg": value,
        }
        payload = generate_shape_key_coupled_measurements(
            sample_id=f"sample_{index:06d}",
            seed=42,
            shape_key_values=shape_keys,
            label_noise_cm=0.0,
        )
        row = {
            **{target: str(value) for target, value in payload["measurements"].items()},
            **{factor: str(value) for factor, value in payload["factors"].items()},
            "sample_id": f"sample_{index:06d}",
            "label_generation_mode": LABEL_GENERATION_MODE,
            "synthetic_labels": "true",
            "real_world_validated": "false",
            "shape_key_values_json": shape_key_traceability_json(shape_keys),
        }
        rows.append(row)

    result = validate_factor_label_correlations(rows, min_abs_correlation=0.80)

    assert result["valid"] is True
    assert result["correlations"]["height_cm"] > 0.95
    assert result["correlations"]["inseam_cm"] > 0.95
    assert result["correlations"]["chest_cm"] > 0.80
    assert result["correlations"]["waist_cm"] > 0.80
    assert result["correlations"]["hip_cm"] > 0.80
    assert result["correlations"]["shoulder_cm"] > 0.80


def test_labels_csv_includes_new_traceability_columns(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=6)

    with (dataset / "labels.csv").open("r", newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        rows = list(reader)

    for column in (
        "label_generation_mode",
        "shape_key_values_json",
        "body_shape_profile_id",
        "torso_width_factor",
        "leg_length_factor",
    ):
        assert column in (reader.fieldnames or [])
    assert rows[0]["label_generation_mode"] == LABEL_GENERATION_MODE
    assert json.loads(rows[0]["shape_key_values_json"])
    assert validate_shape_key_coupled_rows(rows) == []


def test_metadata_includes_formula_and_mapping_information() -> None:
    metadata = shape_key_label_metadata(seed=42)
    config = blend_generation_config(seed=42)

    assert metadata["label_generation_mode"] == LABEL_GENERATION_MODE
    assert metadata["label_formula_version"] == LABEL_FORMULA_VERSION
    assert "shape_key_to_factor_mapping" in metadata
    assert "measurement_formula_summary" in metadata
    assert config["label_generation"]["deterministic_seed"] == 42


def test_generated_dataset_validation_accepts_coupled_schema(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=8)

    result = validate_generated_blend_dataset(dataset, expected_samples=8)

    assert result["valid"] is True
    assert result["factor_label_correlation"]["valid"] is True


def test_no_label_leakage_into_image_feature_extraction(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=2)
    with (dataset / "labels.csv").open("r", newline="", encoding="utf-8") as labels_file:
        row = next(csv.DictReader(labels_file))

    features = extract_blend_image_features(row, dataset)

    assert "shape_key_values_json" not in features
    assert all(column not in features for column in BODY_FACTOR_COLUMNS)
    assert "front_raw_bbox_width_ratio" in features


def _write_coupled_dataset(tmp_path: Path, sample_count: int) -> Path:
    dataset = tmp_path / "phase_3h_g_coupled"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    for index in range(1, sample_count + 1):
        sample_id = f"sample_{index:06d}"
        value = 0.01 + index * 0.01
        shape_keys = {
            "$md-$af-$fe-$yn": value,
            "$md-$af-$ma-$yn": value,
            "$md-$ca-$ma-$yn": value,
            "$md-universal-$ma-$yn-$av$mu-$av$wg": value,
        }
        payload = generate_shape_key_coupled_measurements(
            sample_id=sample_id,
            seed=42,
            shape_key_values=shape_keys,
            label_noise_cm=0.0,
        )
        image_names = {
            "front": f"images/{sample_id}_front.png",
            "side": f"images/{sample_id}_side.png",
            "back": f"images/{sample_id}_back.png",
        }
        _write_rect_image(dataset / image_names["front"], rect=(30, 10, 50 + index, 88))
        _write_rect_image(dataset / image_names["side"], rect=(38, 10, 48 + index // 2, 88))
        _write_rect_image(dataset / image_names["back"], rect=(28, 10, 52 + index, 88))
        rows.append(
            {
                "sample_id": sample_id,
                "front_image": image_names["front"],
                "side_image": image_names["side"],
                "back_image": image_names["back"],
                **payload["measurements"],
                "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                "variation_source": "shape_keys_safe_range",
                "camera_set": camera_set_name(),
                "seed": "42",
                "label_source": SHAPE_KEY_COUPLED_LABEL_SOURCE,
                "synthetic_labels": "true",
                "real_world_validated": "false",
                "label_generation_mode": LABEL_GENERATION_MODE,
                **payload["factors"],
                "shape_key_values_json": shape_key_traceability_json(shape_keys),
                "body_shape_profile_id": payload["body_shape_profile_id"],
            }
        )
    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "generator_version": "phase_3h_blend_dataset_v1",
        "source_blend_file": "assets/body_meshes/base_body_scene.blend",
        "camera_set": camera_set_name(),
        "sample_count": sample_count,
        "seed": 42,
        "synthetic_labels": True,
        "real_world_validated": False,
        "variation_source": "shape_keys_safe_range",
        "shape_key_count": 4,
        **shape_key_label_metadata(42),
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _write_rect_image(path: Path, rect: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (96, 112), (50, 50, 50))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 190, 165)
    image.save(path)
