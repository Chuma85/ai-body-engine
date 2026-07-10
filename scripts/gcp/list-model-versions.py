#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.vertex_model_registry import GoogleVertexRegistryClient, RegistrySettings, load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="List local controlled model versions, optionally including live Vertex metadata.")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "model_lifecycle/vertex_model_registry.json")
    parser.add_argument("--include-vertex", action="store_true")
    args = parser.parse_args()
    payload: dict[str, object] = {"local": load_registry(args.registry)}
    if args.include_vertex:
        settings = RegistrySettings()
        payload["vertex"] = GoogleVertexRegistryClient(settings).list_versions(settings.model_name)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
