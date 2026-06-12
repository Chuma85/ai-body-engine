from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.datasets.real_world_training_candidate import format_dataset_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List registered real-world dataset candidates.")
    parser.add_argument("--registry-path", default="dataset_registry/datasets.json")
    args = parser.parse_args(argv)

    print(format_dataset_registry(args.registry_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
