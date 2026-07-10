#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.vertex_model_registry import GoogleVertexRegistryClient, RegistrySettings, register_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a local AI Body Engine candidate in Vertex Model Registry. Dry-run by default.")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON metadata satisfying model-registry-record inputs.")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "model_lifecycle/vertex_model_registry.json")
    parser.add_argument("--lifecycle-root", type=Path, default=REPO_ROOT / "model_lifecycle")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    client = GoogleVertexRegistryClient(RegistrySettings())
    result = register_candidate(metadata, client=client, registry_path=args.registry, lifecycle_root=args.lifecycle_root, execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
