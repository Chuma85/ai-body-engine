import shlex


def build_blender_command(
    blender_executable: str,
    script_path: str,
    config_path: str,
    dry_run: bool = False,
) -> list[str]:
    # dry_run intentionally affects execution in the CLI helper, not the command shape.
    _ = dry_run
    return [
        blender_executable,
        "--background",
        "--python",
        script_path,
        "--",
        "--config",
        config_path,
    ]


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
