from __future__ import annotations

import argparse
import subprocess

from synthetic.blender.utils.blender_command import build_blender_command, format_command
from synthetic.blender.utils.render_config import load_render_config

DEFAULT_BLENDER_SCRIPT = "synthetic/blender/scripts/render_parametric_body.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run the Blender synthetic pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--blender-executable", default="blender")
    parser.add_argument("--script-path", default=DEFAULT_BLENDER_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_render_config(args.config)
    command = build_blender_command(
        blender_executable=args.blender_executable,
        script_path=args.script_path,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(format_command(command))
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
