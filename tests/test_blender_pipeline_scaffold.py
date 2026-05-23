from dataclasses import replace
import csv
import importlib

import pytest

from synthetic.blender.utils.blender_command import build_blender_command, format_command
from synthetic.blender.utils.measurement_schema import (
    OPTIONAL_METADATA_COLUMNS,
    REQUIRED_MEASUREMENT_COLUMNS,
    validate_measurement_row,
)
from synthetic.blender.utils.render_config import load_render_config, validate_render_config
from synthetic.generator.generate_dataset import LABEL_COLUMNS
from synthetic.generator.validate_dataset import validate_dataset

CONFIG_PATH = "synthetic/blender/configs/phase_2b_render_config.example.json"
PHASE_2C_CONFIG_PATH = "synthetic/blender/configs/phase_2c_render_config.example.json"
PHASE_2D_CONFIG_PATH = "synthetic/blender/configs/phase_2d_render_config.example.json"
PHASE_2E_CONFIG_PATH = "synthetic/blender/configs/phase_2e_base_mesh_config.example.json"
PHASE_2F_CONFIG_PATH = "synthetic/blender/configs/phase_2f_mesh_variation_config.example.json"


def test_example_render_config_loads_and_validates() -> None:
    config = load_render_config(CONFIG_PATH)

    assert config.generator_version == "phase_2b_blender_scaffold_v1"
    assert config.sample_count == 10
    assert "front" in config.views
    assert "side" in config.views


def test_invalid_sample_count_fails_validation() -> None:
    config = load_render_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="sample_count"):
        validate_render_config(replace(config, sample_count=0))


def test_required_measurement_columns_and_row_validation() -> None:
    row = {column: "value" for column in REQUIRED_MEASUREMENT_COLUMNS}

    assert "front_image_path" in REQUIRED_MEASUREMENT_COLUMNS
    assert "side_image_path" in REQUIRED_MEASUREMENT_COLUMNS
    assert validate_measurement_row(row) is True

    row["height_cm"] = ""
    assert validate_measurement_row(row) is False


def test_build_blender_command_and_formatting() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=CONFIG_PATH,
        dry_run=True,
    )

    assert command[:4] == ["blender", "--background", "--python", "synthetic/blender/scripts/render_parametric_body.py"]
    assert command[-2:] == ["--config", CONFIG_PATH]
    assert format_command(command) == (
        "blender --background --python synthetic/blender/scripts/render_parametric_body.py "
        "-- --config synthetic/blender/configs/phase_2b_render_config.example.json"
    )


def test_render_parametric_body_imports_without_bpy() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    assert hasattr(module, "create_procedural_body")
    assert hasattr(module, "generate_body_parameters")
    assert hasattr(module, "main")


def test_blender_output_dir_resolution_for_relative_paths() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    resolved = module.resolve_output_dir("data/synthetic/phase_2c")

    assert resolved == module.repo_root() / "data" / "synthetic" / "phase_2c"
    assert resolved.is_absolute()


def test_blender_output_dir_resolution_for_absolute_paths(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    resolved = module.resolve_output_dir(str(tmp_path / "phase_2c"))

    assert resolved == (tmp_path / "phase_2c").resolve()


def test_blender_output_dir_resolution_for_windows_paths() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    resolved = module.resolve_output_dir(r"C:\ai-body-engine-test\phase_2c")

    assert resolved.is_absolute()
    assert str(resolved).lower().endswith(r"ai-body-engine-test\phase_2c")


def test_blender_output_dir_resolution_for_posix_paths() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    resolved = module.resolve_output_dir("/tmp/ai-body-engine/phase_2c")

    assert resolved.is_absolute()
    assert "ai-body-engine" in resolved.as_posix()


def test_blender_label_paths_remain_repo_relative() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    absolute_path = module.repo_root() / "data" / "synthetic" / "phase_2c" / "images" / "front" / "sample.png"

    assert module.repo_relative_path(absolute_path) == "data/synthetic/phase_2c/images/front/sample.png"


def test_phase_2c_render_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2C_CONFIG_PATH)

    assert config.generator_version == "phase_2c_blender_procedural_body_v1"
    assert config.output_dir == "data/synthetic/phase_2c"
    assert config.materials is not None
    assert len(config.materials["skin_tones"]) == 4


def test_phase_2c_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2C_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2C_CONFIG_PATH
    assert "render_parametric_body.py" in format_command(command)


def test_phase_2d_render_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2D_CONFIG_PATH)

    assert config.generator_version == "phase_2d_anatomical_procedural_body_v1"
    assert config.output_dir == "data/synthetic/phase_2d"
    assert config.anatomy is not None
    assert config.anatomy["enable_torso_taper"] is True
    assert config.render_quality is not None
    assert config.render_quality["resolution_percentage"] == 70


def test_phase_2d_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2D_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2D_CONFIG_PATH
    assert format_command(command).endswith("--config synthetic/blender/configs/phase_2d_render_config.example.json")


def test_body_shape_adjustments_change_expected_scales() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    params = {"body_shape": "athletic"}

    adjusted = module.apply_body_shape_adjustments(params)

    assert adjusted["shoulder_scale"] > 1
    assert adjusted["chest_scale"] > 1
    assert adjusted["waist_scale"] < 1


def test_optional_metadata_columns_exist_but_are_not_required() -> None:
    row = {column: "value" for column in REQUIRED_MEASUREMENT_COLUMNS}

    assert "skin_tone_id" in OPTIONAL_METADATA_COLUMNS
    assert "anatomy_version" in OPTIONAL_METADATA_COLUMNS
    assert "renderer_mode" in OPTIONAL_METADATA_COLUMNS
    assert "fallback_used" in OPTIONAL_METADATA_COLUMNS
    assert validate_measurement_row(row) is True


def test_phase_2e_base_mesh_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2E_CONFIG_PATH)

    assert config.generator_version == "phase_2e_base_mesh_renderer_v1"
    assert config.output_dir == "data/synthetic/phase_2e"
    assert config.base_mesh is not None
    assert config.base_mesh["enabled"] is True
    assert config.base_mesh["fallback_to_procedural"] is True
    assert config.mesh_deformation is not None
    assert config.mesh_deformation["enabled"] is False


def test_base_mesh_config_is_optional_for_older_configs() -> None:
    config = load_render_config(PHASE_2D_CONFIG_PATH)

    assert config.base_mesh is None
    assert config.mesh_deformation is None


def test_phase_2e_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2E_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2E_CONFIG_PATH
    assert format_command(command).endswith("--config synthetic/blender/configs/phase_2e_base_mesh_config.example.json")


def test_resolve_repo_path_resolves_base_mesh_asset() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    resolved = module.resolve_repo_path("assets/body_meshes/base_human.glb")

    assert resolved == module.repo_root() / "assets" / "body_meshes" / "base_human.glb"
    assert resolved.is_absolute()


def test_missing_base_mesh_fallback_metadata_without_blender_calls() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    assert module.load_base_mesh_if_available(object(), str(module.repo_root() / "assets" / "body_meshes" / "missing.glb"), "glb") is None


def test_phase_2f_mesh_variation_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2F_CONFIG_PATH)

    assert config.generator_version == "phase_2f_mesh_variation_v1"
    assert config.output_dir == "data/synthetic/phase_2f"
    assert config.mesh_deformation is not None
    assert config.mesh_deformation["enabled"] is True
    assert config.mesh_deformation["mode"] == "region_scale_v1"
    assert config.camera is not None
    assert config.camera["auto_frame_body"] is True


def test_phase_2f_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2F_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2F_CONFIG_PATH
    assert format_command(command).endswith("--config synthetic/blender/configs/phase_2f_mesh_variation_config.example.json")


def test_region_scale_factors_are_clamped() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    factors = module.compute_region_scale_factors(
        {
            "deformation_strength": 2.0,
            "shoulder_cm": 120,
            "chest_cm": 230,
            "waist_cm": 20,
            "hip_cm": 240,
            "thigh_cm": 5,
            "calf_cm": 120,
            "sleeve_cm": 10,
            "inseam_cm": 160,
        }
    )

    assert factors["shoulders"] == 1.35
    assert factors["waist"] == 0.75
    assert factors["arms"] == 0.75
    assert all(0.75 <= value <= 1.35 for value in factors.values())


def test_region_scale_factors_respond_to_measurements() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    factors = module.compute_region_scale_factors(
        {
            "deformation_strength": 0.35,
            "shoulder_cm": 52,
            "chest_cm": 112,
            "waist_cm": 74,
            "hip_cm": 108,
            "thigh_cm": 62,
            "calf_cm": 34,
            "sleeve_cm": 66,
            "inseam_cm": 86,
        }
    )

    assert factors["shoulders"] > 1.0
    assert factors["chest"] > 1.0
    assert factors["waist"] < 1.0


def test_measurement_schema_matches_dataset_label_columns() -> None:
    assert REQUIRED_MEASUREMENT_COLUMNS == LABEL_COLUMNS


def test_validate_mock_phase_2c_dataset(tmp_path) -> None:
    output_dir = tmp_path / "phase_2c"
    front_dir = output_dir / "images" / "front"
    side_dir = output_dir / "images" / "side"
    labels_dir = output_dir / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    front_path = front_dir / "sample_000001_front.png"
    side_path = side_dir / "sample_000001_side.png"
    front_path.write_bytes(b"mock-front")
    side_path.write_bytes(b"mock-side")

    labels_csv = labels_dir / "labels.csv"
    row = {
        "sample_id": "sample_000001",
        "front_image_path": front_path.as_posix(),
        "side_image_path": side_path.as_posix(),
        "height_cm": "180",
        "weight_kg": "75",
        "chest_cm": "98",
        "waist_cm": "82",
        "hip_cm": "100",
        "shoulder_cm": "45",
        "inseam_cm": "82",
        "sleeve_cm": "62",
        "neck_cm": "39",
        "thigh_cm": "56",
        "calf_cm": "38",
        "body_shape": "average",
        "generator_version": "phase_2c_blender_procedural_body_v1",
    }

    with labels_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    result = validate_dataset(str(labels_csv))

    assert result["valid"] is True
    assert result["row_count"] == 1
