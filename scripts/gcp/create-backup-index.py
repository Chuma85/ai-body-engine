#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.backup_integrity import create_backup_index, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a metadata-only backup index from a GCP-B upload manifest.")
    parser.add_argument("--upload-manifest", type=Path, required=True)
    parser.add_argument("--retention-config", type=Path, default=ROOT / "config/google-cloud/retention-policy.yaml")
    parser.add_argument("--model-registry", type=Path, default=ROOT / "model_lifecycle/vertex_model_registry.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retention = yaml.safe_load(args.retention_config.read_text(encoding="utf-8"))
    index = create_backup_index(load_json(args.upload_manifest), retention, load_json(args.model_registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "object_count": index["object_count"], "total_size_bytes": index["total_size_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
