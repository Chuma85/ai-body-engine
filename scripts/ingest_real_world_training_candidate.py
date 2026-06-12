from __future__ import annotations

import argparse
import json

from training.datasets.real_world_training_candidate import ingest_real_world_training_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a FashionApp real-world training candidate export.")
    parser.add_argument("dataset_version", help="Dataset version folder under data/real_world/incoming.")
    parser.add_argument("--incoming-root", default="data/real_world/incoming")
    parser.add_argument("--processed-root", default="data/real_world/processed")
    parser.add_argument("--registry-path", default="data/real_world/dataset_registry.json")
    parser.add_argument("--import-timestamp", default="1970-01-01T00:00:00Z")
    args = parser.parse_args()

    result = ingest_real_world_training_candidate(
        args.dataset_version,
        incoming_root=args.incoming_root,
        processed_root=args.processed_root,
        registry_path=args.registry_path,
        import_timestamp=args.import_timestamp,
    )
    print(json.dumps(result.__dict__ | {
        "incoming_dir": str(result.incoming_dir),
        "processed_dir": str(result.processed_dir),
        "registry_path": str(result.registry_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
