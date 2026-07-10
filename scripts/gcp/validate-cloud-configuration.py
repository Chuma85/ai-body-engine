#!/usr/bin/env python3
"""Validate checked-in GCP manifests, schemas, and Cloud Build configuration."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    asset = json.loads((ROOT / "config/google-cloud/asset-manifest.example.json").read_text(encoding="utf-8"))
    required_asset_fields = {
        "relative_path", "category", "size_bytes", "file_extension",
        "suggested_google_cloud_destination", "sensitivity_classification", "upload_eligible",
    }
    if asset.get("metadata_only") is not True or asset.get("summary", {}).get("file_count") != len(asset.get("assets", [])):
        raise ValueError("Asset manifest summary or metadata-only marker is invalid.")
    if any(set(item) != required_asset_fields for item in asset["assets"]):
        raise ValueError("Asset manifest contains an invalid per-file record.")

    schema = json.loads((ROOT / "schemas/model-registry-record.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    yaml_paths = [
        ROOT / "config/google-cloud/storage-layout.yaml",
        ROOT / "config/google-cloud/model-registry.yaml",
        *sorted((ROOT / "cloudbuild").glob("*.yaml")),
    ]
    for path in yaml_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"YAML root must be a mapping: {path.relative_to(ROOT)}")

    registry_config = yaml.safe_load((ROOT / "config/google-cloud/model-registry.yaml").read_text(encoding="utf-8"))
    if any(registry_config["policy"].get(key) for key in ("auto_promote", "auto_deploy", "create_endpoint")):
        raise ValueError("Model registry automatic promotion/deployment policy must remain disabled.")

    print(json.dumps({
        "valid": True,
        "asset_records": len(asset["assets"]),
        "json_schemas": 1,
        "yaml_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in yaml_paths],
        "lint": "not_configured",
        "type_check": "not_configured",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
