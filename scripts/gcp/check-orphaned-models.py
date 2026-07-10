#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.backup_integrity import check_orphaned_models, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Report model records without artifacts and artifacts without model records.")
    parser.add_argument("--backup-index", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, default=ROOT / "model_lifecycle/vertex_model_registry.json")
    args = parser.parse_args()
    result = check_orphaned_models(load_json(args.backup_index), load_json(args.model_registry))
    print(json.dumps(result, indent=2))
    return 2 if result["model_records_without_artifacts"] or result["artifacts_without_model_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
