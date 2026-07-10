#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.backup_integrity import check_integrity, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Check backup metadata integrity without downloading object contents.")
    parser.add_argument("--backup-index", type=Path, required=True)
    parser.add_argument("--observed-objects", type=Path, required=True, help="JSON object metadata fixture/export with an objects array.")
    parser.add_argument("--model-registry", type=Path, default=ROOT / "model_lifecycle/vertex_model_registry.json")
    parser.add_argument("--training-runs", type=Path, default=ROOT / "model_lifecycle/training_runs.json")
    args = parser.parse_args()
    observed_payload = load_json(args.observed_objects)
    result = check_integrity(load_json(args.backup_index), observed_payload.get("objects", []), model_registry=load_json(args.model_registry), training_runs=load_json(args.training_runs))
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
