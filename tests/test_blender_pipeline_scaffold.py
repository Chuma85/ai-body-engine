from dataclasses import replace
import csv
import importlib
import math
import random
from types import SimpleNamespace

import pytest

from synthetic.blender.utils.blender_command import build_blender_command, format_command
from synthetic.blender.utils.deformation_math import clamp, compute_shape_key_targets, normalize_measurement_to_unit
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
PHASE_2G_CONFIG_PATH = "synthetic/blender/configs/phase_2g_rigged_mesh_config.example.json"
PHASE_2V_CONFIG_PATH = "synthetic/blender/configs/phase_2v_controlled_variation_config.example.json"
PHASE_3G_CONFIG_PATH = "synthetic/blender/configs/phase_3g_render_realism_config.example.json"


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
    assert "rigging_enabled" in OPTIONAL_METADATA_COLUMNS
    assert "shape_key_matches" in OPTIONAL_METADATA_COLUMNS
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


def test_phase_2g_rigged_mesh_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2G_CONFIG_PATH)

    assert config.generator_version == "phase_2g_rigged_mesh_pipeline_v1"
    assert config.output_dir == "data/synthetic/phase_2g"
    assert config.base_mesh is not None
    assert config.base_mesh["asset_path"] == "assets/body_meshes/base_human_rigged.fbx"
    assert config.base_mesh["source_front_axis"] == "+X"
    assert config.base_mesh["source_yaw_degrees"] == 30.0
    assert config.base_mesh["fallback_to_static_mesh"] is True
    assert config.rigging is not None
    assert config.rigging["detect_armature"] is True
    assert config.shape_key_mapping is not None
    assert "waist" in config.shape_key_mapping
    assert config.mesh_deformation is not None
    assert config.mesh_deformation["mode"] == "rigged_or_shape_key_v1"
    assert config.variation_controls is not None
    assert config.variation_controls["enabled"] is False
    assert "slim" in config.variation_controls["body_shape_profiles"]


def test_variation_controls_are_backwards_compatible_when_disabled() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    base_ranges = {"height_cm": [150, 205], "weight_kg": [45, 130]}
    config = {
        "body_parameter_ranges": base_ranges,
        "variation_controls": {
            "enabled": False,
            "profile_range_overrides_enabled": True,
            "body_shape_profiles": {
                "slim": {"body_parameter_ranges": {"weight_kg": [45, 75]}},
            },
        },
    }

    assert module.ranges_for_body_shape(base_ranges, "slim", config) == base_ranges


def test_variation_controls_can_apply_profile_ranges_when_enabled() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    base_ranges = {"height_cm": [150, 205], "weight_kg": [45, 130]}
    config = {
        "body_parameter_ranges": base_ranges,
        "variation_controls": {
            "enabled": True,
            "profile_range_overrides_enabled": True,
            "body_shape_profiles": {
                "slim": {"body_parameter_ranges": {"weight_kg": [45, 75]}},
            },
        },
    }

    ranges = module.ranges_for_body_shape(base_ranges, "slim", config)

    assert ranges["height_cm"] == [150, 205]
    assert ranges["weight_kg"] == [45, 75]
    assert base_ranges["weight_kg"] == [45, 130]


def test_variation_controls_select_configured_profiles_when_enabled() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {
        "variation_controls": {
            "enabled": True,
            "body_shape_profiles": {
                "slim": {},
            },
        },
    }

    assert module.select_body_shape(random.Random(1), config) == "slim"


def test_phase_2g_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2G_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2G_CONFIG_PATH
    assert format_command(command).endswith("--config synthetic/blender/configs/phase_2g_rigged_mesh_config.example.json")


def test_phase_2v_controlled_variation_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_2V_CONFIG_PATH)

    assert config.generator_version == "phase_2v_controlled_variation_v1"
    assert config.output_dir == "data/synthetic/phase_2v"
    assert config.sample_count == 1000
    assert config.variation_controls is not None
    assert config.variation_controls["enabled"] is True
    assert config.variation_controls["profile_range_overrides_enabled"] is True
    assert sorted(config.variation_controls["body_shape_profiles"]) == ["average", "broad", "curvy", "slim"]


def test_phase_2v_blender_command_can_be_built() -> None:
    command = build_blender_command(
        blender_executable="blender",
        script_path="synthetic/blender/scripts/render_parametric_body.py",
        config_path=PHASE_2V_CONFIG_PATH,
        dry_run=True,
    )

    assert command[-1] == PHASE_2V_CONFIG_PATH
    assert format_command(command).endswith("--config synthetic/blender/configs/phase_2v_controlled_variation_config.example.json")


def test_render_realism_defaults_are_disabled_for_existing_configs() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = load_render_config(PHASE_2G_CONFIG_PATH)

    controls = module.render_realism_controls(config.__dict__)

    assert config.render_realism is None
    assert controls["enabled"] is False
    assert controls["camera"]["distance_jitter_range"] == [0.0, 0.0]
    assert controls["materials"]["skin_tone_brightness_range"] == [1.0, 1.0]


def test_phase_3g_render_realism_config_loads_and_validates() -> None:
    config = load_render_config(PHASE_3G_CONFIG_PATH)

    assert config.generator_version == "phase_3g_render_realism_v1"
    assert config.output_dir == "data/synthetic/phase_3g_smoke"
    assert config.render_realism is not None
    assert config.render_realism["enabled"] is True
    assert config.render_realism["background"]["brightness_range"] == [0.82, 1.0]
    assert config.render_realism["camera"]["orthographic_scale_jitter_range"] == [1.0, 1.05]


def test_render_realism_resolution_override_is_optional_and_compatible() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {
        "image_width": 768,
        "image_height": 1024,
        "render_realism": {
            "enabled": True,
            "render_resolution": {
                "enabled": True,
                "image_width": 640,
                "image_height": 896,
            },
        },
    }

    module.apply_render_realism_resolution_override(config)

    assert config["image_width"] == 640
    assert config["image_height"] == 896


def test_render_realism_camera_jitter_stays_inside_safe_bounds() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {
        "render_realism": {
            "enabled": True,
            "camera": {
                "distance_jitter_range": [-0.12, 0.12],
                "orthographic_scale_jitter_range": [1.0, 1.05],
                "lateral_offset_range": [-0.035, 0.035],
                "vertical_offset_range": [-0.035, 0.035],
            },
        },
    }

    jitter = module.camera_jitter(config, random.Random(42))

    assert -0.12 <= jitter["distance_delta"] <= 0.12
    assert 1.0 <= jitter["orthographic_scale_multiplier"] <= 1.05
    assert -0.035 <= jitter["lateral_offset"] <= 0.035
    assert -0.035 <= jitter["vertical_offset"] <= 0.035


def test_render_realism_rejects_unsafe_camera_jitter() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {
        "render_realism": {
            "enabled": True,
            "camera": {
                "lateral_offset_range": [-0.5, 0.5],
            },
        },
    }

    with pytest.raises(ValueError, match="lateral_offset_range"):
        module.render_realism_controls(config)


def test_phase_3g_label_row_format_remains_compatible() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = module.load_config(PHASE_3G_CONFIG_PATH)
    front_path = module.repo_root() / "data" / "synthetic" / "phase_3g_smoke" / "images" / "front" / "sample_000001_front.png"
    side_path = module.repo_root() / "data" / "synthetic" / "phase_3g_smoke" / "images" / "side" / "sample_000001_side.png"
    params = {
        "sample_id": "sample_000001",
        "height_cm": 180.0,
        "weight_kg": 75.0,
        "chest_cm": 100.0,
        "waist_cm": 82.0,
        "hip_cm": 101.0,
        "shoulder_cm": 45.0,
        "inseam_cm": 82.0,
        "sleeve_cm": 62.0,
        "neck_cm": 39.0,
        "thigh_cm": 56.0,
        "calf_cm": 38.0,
        "body_shape": "average",
        "skin_tone_id": 0,
        "pose_variation_degrees": 0.0,
    }

    row = module.label_row_for_sample(params, config, front_path, side_path, module.resume_render_metadata(config))

    assert set(row) == set(module.LABEL_COLUMNS)
    assert validate_measurement_row(row) is True
    assert row["render_width"] == 768
    assert row["render_height"] == 1024


def test_deformation_math_clamps_and_normalizes_values() -> None:
    assert clamp(-1, 0, 1) == 0
    assert clamp(2, 0, 1) == 1
    assert normalize_measurement_to_unit(82, 82, 55, 125) == 0.5
    assert normalize_measurement_to_unit(500, 82, 55, 125) == 1.0


def test_shape_key_targets_are_clamped_between_zero_and_one() -> None:
    targets = compute_shape_key_targets(
        {
            "height_cm": 230,
            "weight_kg": 200,
            "chest_cm": 160,
            "waist_cm": 20,
            "hip_cm": 150,
            "shoulder_cm": 80,
            "body_shape": "athletic",
        },
        {
            "height": ["Height"],
            "weight": ["Weight"],
            "chest": ["Chest"],
            "waist": ["Waist"],
            "hips": ["Hips"],
            "shoulders": ["Shoulders"],
            "muscle": ["Muscle"],
            "body_fat": ["Fat"],
        },
    )

    assert targets["height"] == 1.0
    assert targets["waist"] == 0.0
    assert targets["muscle"] == 0.8
    assert all(0.0 <= value <= 1.0 for value in targets.values())


def test_phase_2g_renderer_helpers_are_import_safe_without_bpy() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    assert hasattr(module, "detect_armatures")
    assert hasattr(module, "detect_shape_keys")
    assert hasattr(module, "apply_rigged_mesh_deformation")


def test_phase_2i_fbx_front_axis_rotates_to_canonical_front() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    assert module.CANONICAL_FRONT_AXIS == "-Y"
    assert module.CANONICAL_SIDE_VIEW_AXIS == "-X"
    assert math.isclose(module.horizontal_axis_rotation("+X", module.CANONICAL_FRONT_AXIS), -math.pi / 2)
    assert math.isclose(module.horizontal_axis_rotation("-Y", module.CANONICAL_FRONT_AXIS), 0.0)


def test_phase_2j_source_yaw_calibrates_rigged_mesh_profile() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    rotation = module.horizontal_axis_rotation("+X", module.CANONICAL_FRONT_AXIS) + math.radians(30.0)

    assert math.isclose(rotation, math.radians(-60.0))


def test_phase_2i_camera_transforms_use_true_front_and_side_views() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")

    front_location, front_rotation = module.camera_transform_for_view("front", (0.0, 0.0, 1.5), 4.0)
    side_location, side_rotation = module.camera_transform_for_view("side", (0.0, 0.0, 1.5), 4.0)

    assert front_location == (0.0, -4.0, 1.5)
    assert front_rotation == (math.pi / 2, 0, 0)
    assert side_location == (-4.0, 0.0, 1.5)
    assert side_rotation == (math.pi / 2, 0, -math.pi / 2)


def test_phase_2i_orthographic_scale_keeps_full_body_visible_in_portrait_frame() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    bounds = (-0.5, 0.5, -0.2, 0.2, 0.0, 3.0)
    frame_width, frame_height, frame_depth = module.camera_frame_dimensions(bounds, "front")

    assert frame_width == 1.0
    assert frame_height == 3.0
    assert frame_depth == 0.4
    assert module.orthographic_scale_for_frame(frame_width, frame_height, 768 / 1024, 0.18) == pytest.approx(3.54)


def test_phase_2i_cli_overrides_output_and_sample_count() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {"output_dir": "data/synthetic/phase_2g", "sample_count": 5}

    module.apply_cli_overrides(config, SimpleNamespace(output="data/synthetic/phase_2i", num_samples=2))

    assert config["output_dir"] == "data/synthetic/phase_2i"
    assert config["sample_count"] == 2


def test_phase_2v_batch_label_writer_can_append(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    labels_path = tmp_path / "labels.csv"
    first_row = {column: "" for column in module.LABEL_COLUMNS}
    first_row.update({"sample_id": "sample_000001", "front_image_path": "front1.png", "side_image_path": "side1.png"})
    second_row = {column: "" for column in module.LABEL_COLUMNS}
    second_row.update({"sample_id": "sample_000002", "front_image_path": "front2.png", "side_image_path": "side2.png"})

    module.write_labels_csv([first_row], labels_path)
    module.write_labels_csv([second_row], labels_path, append=True)

    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        rows = list(csv.DictReader(labels_file))

    assert [row["sample_id"] for row in rows] == ["sample_000001", "sample_000002"]


def test_phase_2v_incremental_label_writer_flushes_each_row(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    labels_path = tmp_path / "labels.csv"
    row = {column: "" for column in module.LABEL_COLUMNS}
    row.update({"sample_id": "sample_000001", "front_image_path": "front1.png", "side_image_path": "side1.png"})

    module.append_label_row(labels_path, row)

    assert module.read_labeled_sample_ids(labels_path) == {"sample_000001"}


def test_phase_2v_resume_action_skips_completed_samples(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    front_path = tmp_path / "sample_000001_front.png"
    side_path = tmp_path / "sample_000001_side.png"
    front_path.write_bytes(b"front")
    side_path.write_bytes(b"side")

    action = module.resume_action_for_sample("sample_000001", front_path, side_path, {"sample_000001"})

    assert action == "skip"


def test_phase_2v_resume_action_checkpoints_rendered_pair_without_label(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    front_path = tmp_path / "sample_000001_front.png"
    side_path = tmp_path / "sample_000001_side.png"
    front_path.write_bytes(b"front")
    side_path.write_bytes(b"side")

    action = module.resume_action_for_sample("sample_000001", front_path, side_path, set())

    assert action == "checkpoint_existing_pair"


def test_phase_2v_resume_does_not_create_duplicate_label_rows(tmp_path) -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    labels_path = tmp_path / "labels.csv"
    row = {column: "" for column in module.LABEL_COLUMNS}
    row.update({"sample_id": "sample_000001", "front_image_path": "front1.png", "side_image_path": "side1.png"})

    module.append_label_row(labels_path, row)
    labeled_sample_ids = module.read_labeled_sample_ids(labels_path)
    if "sample_000001" not in labeled_sample_ids:
        module.append_label_row(labels_path, row)

    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        rows = list(csv.DictReader(labels_file))

    assert [label_row["sample_id"] for label_row in rows] == ["sample_000001"]


def test_phase_2v_batch_rng_matches_single_pass_sequence() -> None:
    module = importlib.import_module("synthetic.blender.scripts.render_parametric_body")
    config = {
        "random_seed": 42,
        "body_parameter_ranges": {
            "height_cm": [150, 205],
            "weight_kg": [45, 130],
            "chest_cm": [75, 130],
            "waist_cm": [55, 125],
            "hip_cm": [75, 135],
            "shoulder_cm": [35, 60],
            "inseam_cm": [65, 95],
            "sleeve_cm": [50, 75],
            "neck_cm": [30, 50],
            "thigh_cm": [40, 80],
            "calf_cm": [28, 55],
        },
        "variation_controls": {
            "enabled": True,
            "profile_range_overrides_enabled": True,
            "body_shape_profiles": {
                "slim": {"body_parameter_ranges": {"weight_kg": [45, 78]}},
                "average": {"body_parameter_ranges": {"weight_kg": [55, 95]}},
            },
        },
    }
    single_rng = random.Random(config["random_seed"])
    batch_rng = random.Random(config["random_seed"])

    single_pass = [module.generate_body_parameters(index, single_rng, config) for index in range(1, 6)]
    for skipped_index in range(1, 5):
        module.generate_body_parameters(skipped_index, batch_rng, config)
    batch_sample = module.generate_body_parameters(5, batch_rng, config)

    assert batch_sample == single_pass[-1]


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
