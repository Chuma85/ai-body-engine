from dataclasses import replace
import importlib

import pytest

from synthetic.blender.utils.blender_command import build_blender_command, format_command
from synthetic.blender.utils.measurement_schema import REQUIRED_MEASUREMENT_COLUMNS, validate_measurement_row
from synthetic.blender.utils.render_config import load_render_config, validate_render_config

CONFIG_PATH = "synthetic/blender/configs/phase_2b_render_config.example.json"


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

    assert hasattr(module, "create_parametric_body_placeholder")
    assert hasattr(module, "main")
