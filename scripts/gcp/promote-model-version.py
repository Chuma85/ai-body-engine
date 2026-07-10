#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.vertex_model_registry import GoogleVertexRegistryClient, RegistrySettings, promote_version


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicitly promote an evaluated Vertex model version. Dry-run by default.")
    parser.add_argument("--model-version-id", required=True)
    parser.add_argument("--approval-identity", required=True)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "model_lifecycle/vertex_model_registry.json")
    parser.add_argument("--lifecycle-root", type=Path, default=REPO_ROOT / "model_lifecycle")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    client = GoogleVertexRegistryClient(RegistrySettings()) if args.execute else None
    result = promote_version(args.model_version_id, approval_identity=args.approval_identity, approval_reference=args.approval_reference, client=client, registry_path=args.registry, lifecycle_root=args.lifecycle_root, execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
