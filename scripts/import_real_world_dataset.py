from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.datasets.real_world_training_candidate import import_real_world_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a reviewed CUSTOM-FASHION-MARKETPLACE dataset export.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default="data/real_world/incoming/dataset_export_manifest.json",
        help="Path to dataset_export_manifest.json.",
    )
    parser.add_argument("--registry-path", default="dataset_registry/datasets.json")
    parser.add_argument("--real-world-root", default="data/real_world")
    parser.add_argument("--report-path", default="reports/dataset_validation_report.json")
    parser.add_argument("--import-timestamp")
    args = parser.parse_args(argv)

    result = import_real_world_dataset(
        args.manifest,
        import_timestamp=args.import_timestamp,
        real_world_root=args.real_world_root,
        registry_path=args.registry_path,
        report_path=args.report_path,
    )
    payload = asdict(result)
    for key in ("incoming_dir", "validated_dir", "registry_path", "report_path"):
        payload[key] = str(payload[key])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
