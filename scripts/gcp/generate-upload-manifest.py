#!/usr/bin/env python3
"""Generate a sanitized, checksum-bearing GCS upload manifest; never upload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/google-cloud/storage-layout.yaml"
DEFAULT_ASSET_MANIFEST = REPO_ROOT / "config/google-cloud/asset-manifest.example.json"
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".pytest-tmp", ".tmp", "node_modules", "build", "dist", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_NAMES = re.compile(r"(?i)(^\.env(?:\..+)?$|credential|service[-_]?account|\.pem$|\.key$|\.p12$|\.pfx$)")
REAL_WORLD_CATEGORIES = {"real-world datasets", "participant images", "verified exports"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML object: {path}")
    return value


def is_under(relative_path: str, roots: list[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(path == PurePosixPath(root) or PurePosixPath(root) in path.parents for root in roots)


def safe_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return not (set(part.lower() for part in path.parts) & FORBIDDEN_PARTS) and not FORBIDDEN_NAMES.search(path.name)


def checksums(path: Path) -> tuple[str, str]:
    sha_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha_digest.update(block)
            md5_digest.update(block)
    return sha_digest.hexdigest(), base64.b64encode(md5_digest.digest()).decode("ascii")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def add_database_backup(entries: list[dict], config: dict, backup_path: Path, stamp: str) -> None:
    if not backup_path.is_file():
        raise FileNotFoundError(f"Approved Railway backup not found: {backup_path}")
    bucket = config["buckets"]["database_backups"]["name"]
    prefix = config["buckets"]["database_backups"]["prefixes"]["railway"]
    sha_value, md5_value = checksums(backup_path)
    entries.append({
        "source_relative_path": None,
        "source_path_from_environment": "AI_BODY_RAILWAY_BACKUP_PATH",
        "source_file_name": backup_path.name,
        "category": "database backup",
        "size_bytes": backup_path.stat().st_size,
        "sha256": sha_value,
        "md5_base64": md5_value,
        "gcs_uri": f"gs://{bucket}/{prefix}/{stamp}/{backup_path.name}",
        "sensitivity_classification": "restricted-sensitive",
        "real_world_data": True,
    })


def build(args: argparse.Namespace) -> tuple[dict, Path]:
    config = load_yaml(args.config)
    with args.asset_manifest.open("r", encoding="utf-8") as handle:
        inventory = json.load(handle)
    if not inventory.get("metadata_only"):
        raise ValueError("Phase GCP-A metadata-only manifest is required")
    if args.include_real_world and not args.approve_real_world:
        raise ValueError("--include-real-world requires --approve-real-world")
    if args.include_real_world and not config["policy"].get("real_world_uploads_enabled", False):
        raise ValueError("Real-world uploads are disabled by storage-layout.yaml policy")
    if args.railway_backup and not args.approve_database_backup:
        raise ValueError("--railway-backup requires --approve-database-backup")
    routes = config["category_routes"]
    entries = []
    exclusions: dict[str, int] = {}
    for asset in inventory["assets"]:
        category = asset["category"]
        relative = asset["relative_path"]
        reason = None
        route = routes.get(category)
        if route is None:
            reason = "category_not_routed"
        elif not safe_path(relative):
            reason = "forbidden_path_or_credential_name"
        elif not is_under(relative, route["approved_roots"]):
            reason = "outside_approved_roots"
        elif category in REAL_WORLD_CATEGORIES and not args.include_real_world:
            reason = "real_world_not_approved"
        elif category == "logs" and not config["policy"].get("logs_upload_enabled", False):
            reason = "logs_disabled_by_policy"
        elif not asset.get("upload_eligible") and category != "logs":
            reason = "phase_gcp_a_ineligible"
        source = REPO_ROOT / Path(relative)
        if reason is None and not source.is_file():
            reason = "source_missing"
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        bucket = config["buckets"][route["bucket"]]["name"]
        prefix = config["buckets"][route["bucket"]]["prefixes"][route["prefix"]]
        sha_value, md5_value = checksums(source) if args.checksums else (None, None)
        entries.append({
            "source_relative_path": relative,
            "category": category,
            "size_bytes": source.stat().st_size,
            "sha256": sha_value,
            "md5_base64": md5_value,
            "gcs_uri": f"gs://{bucket}/{prefix}/{relative}",
            "sensitivity_classification": asset["sensitivity_classification"],
            "real_world_data": category in REAL_WORLD_CATEGORIES,
        })
    stamp = timestamp()
    if args.railway_backup:
        add_database_backup(entries, config, args.railway_backup.resolve(), stamp)
    entries.sort(key=lambda item: item["gcs_uri"])
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": args.project_id,
        "region": config["region"],
        "dry_run": not args.execute,
        "checksums": "sha256" if args.checksums else "disabled",
        "real_world_upload_approved": bool(args.include_real_world and args.approve_real_world),
        "database_backup_approved": bool(args.railway_backup and args.approve_database_backup),
        "summary": {"object_count": len(entries), "total_size_bytes": sum(item["size_bytes"] for item in entries), "excluded_counts": dict(sorted(exclusions.items()))},
        "objects": entries,
    }
    output_dir = REPO_ROOT / config["manifest_output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or output_dir / f"upload-manifest-{stamp}.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, output


def validate_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    objects = manifest.get("objects", [])
    uris = [item["gcs_uri"] for item in objects]
    forbidden = [item for item in objects if not safe_path(item.get("source_relative_path") or item.get("source_file_name") or "")]
    missing_checksums = [item for item in objects if manifest.get("checksums") == "sha256" and (len(item.get("sha256") or "") != 64 or not item.get("md5_base64"))]
    real_world_count = sum(bool(item.get("real_world_data")) for item in objects)
    unapproved_real_world = any(item.get("real_world_data") and item.get("category") != "database backup" for item in objects) and not manifest.get("real_world_upload_approved")
    unapproved_database = any(item.get("category") == "database backup" for item in objects) and not manifest.get("database_backup_approved")
    summary_matches = manifest.get("summary", {}).get("object_count") == len(objects) and manifest.get("summary", {}).get("total_size_bytes") == sum(item["size_bytes"] for item in objects)
    result = {
        "valid": not forbidden and not missing_checksums and len(uris) == len(set(uris)) and summary_matches and not unapproved_real_world and not unapproved_database,
        "dry_run": manifest.get("dry_run"),
        "object_count": len(objects),
        "total_size_bytes": sum(item["size_bytes"] for item in objects),
        "forbidden_path_count": len(forbidden),
        "real_world_object_count": real_world_count,
        "missing_checksum_count": len(missing_checksums),
        "duplicate_destination_count": len(uris) - len(set(uris)),
        "summary_matches": summary_matches,
        "categories": sorted({item["category"] for item in objects}),
    }
    if not result["valid"]:
        raise ValueError(f"Upload manifest validation failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-id")
    parser.add_argument("--validate-manifest", type=Path, help="Validate an existing upload manifest and exit")
    parser.add_argument("--checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-real-world", action="store_true")
    parser.add_argument("--approve-real-world", action="store_true")
    parser.add_argument("--railway-backup", type=Path)
    parser.add_argument("--approve-database-backup", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Mark manifest for an explicitly approved execution")
    args = parser.parse_args()
    if args.validate_manifest:
        print(json.dumps(validate_manifest(args.validate_manifest), indent=2))
        return 0
    if not args.project_id:
        parser.error("--project-id is required when generating a manifest")
    manifest, output = build(args)
    print(json.dumps({"manifest": str(output), "dry_run": manifest["dry_run"], **manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
