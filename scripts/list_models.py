from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.model_lifecycle import format_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List registered AI Body models.")
    parser.add_argument("--lifecycle-root", default="model_lifecycle")
    args = parser.parse_args(argv)
    print(format_models(args.lifecycle_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
