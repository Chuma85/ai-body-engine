from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.datasets.real_world_training_candidate import ingest_real_world_training_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a FashionApp real-world training candidate export.")
    parser.add_argument("dataset_version", help="Dataset version folder under data/real_world/incoming.")
    parser.add_argument("--incoming-root", default="data/real_world/incoming")
    parser.add_argument("--processed-root", default="data/real_world/validated")
    parser.add_argument("--registry-path", default="dataset_registry/datasets.json")
    parser.add_argument("--import-timestamp", default="1970-01-01T00:00:00Z")
    args = parser.parse_args()

    result = ingest_real_world_training_candidate(
        args.dataset_version,
        incoming_root=args.incoming_root,
        processed_root=args.processed_root,
        registry_path=args.registry_path,
        import_timestamp=args.import_timestamp,
    )
    payload = asdict(result)
    payload["processed_dir"] = str(result.processed_dir)
    print(json.dumps(payload | {
        "incoming_dir": str(result.incoming_dir),
        "validated_dir": str(result.validated_dir),
        "registry_path": str(result.registry_path),
        "report_path": str(result.report_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
