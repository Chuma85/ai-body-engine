#!/usr/bin/env python3
"""Inventory local AI Body Engine assets without modifying or uploading them."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "config/google-cloud/asset-manifest.example.json"
DEFAULT_INVENTORY = REPO_ROOT / "docs/google-cloud/AI_BODY_ENGINE_ASSET_INVENTORY.md"

NEVER_UPLOAD_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".pytest-tmp",
    ".tmp", "node_modules", ".mypy_cache", ".ruff_cache", "dist", "build",
}
MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".ckpt", ".h5", ".keras", ".pkl", ".joblib", ".safetensors", ".tflite", ".pb"}
DATASET_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".npy", ".npz", ".arrow", ".feather", ".h5"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".dcm"}
RENDER_EXTENSIONS = IMAGE_EXTENSIONS | {".blend", ".blend1", ".fbx", ".obj", ".glb", ".gltf"}
TEXT_SCAN_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".env", ".sh", ".ps1", ".sql"}
ALL_CATEGORIES = {
    "source code", "synthetic datasets", "real-world datasets", "participant images",
    "verified exports", "training manifests", "evaluation holdouts", "model checkpoints",
    "pretrained models", "candidate models", "promoted models", "evaluation reports",
    "leakage audits", "comparison reports", "generated renders", "logs", "temporary files",
    "caches", "virtual environments", "secrets and credentials", "database dumps",
    "other repository assets",
}

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=]{24,}"),
    "connection_string": re.compile(r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s'\"]+", re.I),
    "credential_assignment": re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|secret|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*[^\s#]{8,}"),
    "google_service_account": re.compile(r'"type"\s*:\s*"service_account"'),
}


def normalized(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def iter_files() -> Iterable[Path]:
    generated_outputs = {DEFAULT_MANIFEST.resolve(), DEFAULT_INVENTORY.resolve()}
    for root, dirs, files in os.walk(REPO_ROOT, followlinks=False):
        if Path(root) == REPO_ROOT and ".git" in dirs:
            dirs.remove(".git")
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if not path.is_symlink() and path.resolve() not in generated_outputs:
                yield path


def git_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def classify(relative_path: str, extension: str) -> str:
    lower = relative_path.lower()
    parts = set(lower.split("/"))
    name = Path(lower).name
    if ".git" in parts:
        return "git metadata"
    if parts & {".venv", "venv"}:
        return "virtual environments"
    if parts & {"__pycache__", ".pytest_cache", ".pytest-tmp", ".mypy_cache", ".ruff_cache"}:
        return "caches"
    if parts & {".tmp", "tmp", "temp", "build", "dist"} or extension in {".tmp", ".temp", ".bak"}:
        return "temporary files"
    if (name.startswith(".env") and name not in {".env.example", ".env.sample", ".env.template"}) or "credential" in name or "service-account" in name or "service_account" in name or extension in {".pem", ".key", ".p12", ".pfx"}:
        return "secrets and credentials"
    if extension in {".sql", ".dump", ".bak"} or name.endswith((".sql.gz", ".dump.gz")):
        return "database dumps"
    if "participant" in lower and extension in IMAGE_EXTENSIONS:
        return "participant images"
    if "holdout" in lower or "test_split" in lower or "evaluation_split" in lower:
        return "evaluation holdouts"
    if "leakage" in lower:
        return "leakage audits"
    if "comparison" in lower or "compare_" in name or "benchmark_leaderboard" in lower:
        return "comparison reports"
    if "verified" in lower and ("export" in lower or extension in DATASET_EXTENSIONS):
        return "verified exports"
    if "manifest" in name or "dataset_registry" in parts or "model_lifecycle" in parts:
        return "training manifests"
    if extension in MODEL_EXTENSIONS or ("model" in name and extension == ".json"):
        if "pretrain" in lower or "foundation" in lower:
            return "pretrained models"
        if "production" in lower or "promoted" in lower:
            return "promoted models"
        if "candidate" in lower:
            return "candidate models"
        return "model checkpoints"
    if "checkpoint" in lower:
        return "model checkpoints"
    if "logs" in parts or extension == ".log":
        return "logs"
    if "report" in lower or "benchmark" in lower or "evaluation" in lower or "metrics" in lower:
        return "evaluation reports"
    if "render" in lower or extension in RENDER_EXTENSIONS:
        return "generated renders"
    if "real_world" in lower or "real-world" in lower:
        return "real-world datasets"
    if "synthetic" in lower or ("data" in parts and extension in DATASET_EXTENSIONS | IMAGE_EXTENSIONS):
        return "synthetic datasets"
    if extension in {".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".ps1", ".toml", ".ini", ".cfg", ".md", ".txt", ".html", ".css", ".yml", ".yaml"}:
        return "source code"
    return "other repository assets"


def sensitivity(category: str, relative_path: str) -> str:
    if category == "secrets and credentials":
        return "restricted-secret"
    if category in {"participant images", "real-world datasets", "verified exports", "database dumps"}:
        return "restricted-sensitive"
    if category in {"evaluation holdouts", "model checkpoints", "pretrained models", "candidate models", "promoted models"}:
        return "confidential"
    if relative_path.startswith(".git/"):
        return "internal"
    return "internal"


def destination(category: str) -> str:
    if category == "source code":
        return "Google Secure Source Manager (GitHub temporary mirror)"
    if category in {"synthetic datasets", "real-world datasets", "participant images", "verified exports", "evaluation holdouts", "training manifests"}:
        return "Private Google Cloud Storage dataset bucket"
    if category in {"model checkpoints", "pretrained models", "candidate models", "promoted models"}:
        return "Private Google Cloud Storage model bucket"
    if category == "database dumps":
        return "Dedicated private Google Cloud Storage backup bucket"
    if category == "secrets and credentials":
        return "Secret Manager (manual secret recreation only; never upload file)"
    if category in {"evaluation reports", "leakage audits", "comparison reports"}:
        return "Private Google Cloud Storage report bucket"
    if category == "generated renders":
        return "Private Google Cloud Storage render bucket"
    return "Do not upload or review manually"


def eligible(category: str, relative_path: str) -> bool:
    parts = set(relative_path.lower().split("/"))
    if parts & NEVER_UPLOAD_PARTS:
        return False
    if category in {"secrets and credentials", "caches", "virtual environments", "temporary files", "git metadata", "logs"}:
        return False
    return True


def scan_sensitive(path: Path, relative_path: str, category: str) -> list[str]:
    findings: set[str] = set()
    lower_name = path.name.lower()
    if lower_name.startswith(".env") and lower_name not in {".env.example", ".env.sample", ".env.template"}:
        findings.add("environment_file")
    if category == "participant images":
        findings.add("participant_photo")
    if category == "database dumps":
        findings.add("database_dump")
    if path.stat().st_size > 2_000_000 or (path.suffix.lower() not in TEXT_SCAN_EXTENSIONS and not lower_name.startswith(".env")):
        return sorted(findings)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return sorted(findings)
    for finding, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            findings.add(finding)
    return sorted(findings)


def render_inventory(manifest: dict, sensitive: list[dict]) -> str:
    lines = [
        "# AI Body Engine Asset Inventory", "",
        "> Generated by `python scripts/gcp/audit-ai-body-assets.py`. Metadata only; no assets were moved, deleted, or uploaded.", "",
        f"- Files inventoried: **{manifest['summary']['file_count']:,}**",
        f"- Total bytes: **{manifest['summary']['total_size_bytes']:,}**",
        f"- Upload-eligible files: **{manifest['summary']['upload_eligible_file_count']:,}**",
        f"- Potentially sensitive findings: **{len(sensitive):,}** (paths and finding types only)", "",
        "## Category totals", "", "| Category | Files | Bytes |", "|---|---:|---:|",
    ]
    for item in manifest["category_summary"]:
        lines.append(f"| {item['category']} | {item['file_count']:,} | {item['total_size_bytes']:,} |")
    lines += ["", "## Largest 25 files", "", "| Relative path | Category | Bytes | Upload eligible |", "|---|---|---:|---|"]
    for item in manifest["largest_files"]:
        lines.append(f"| `{item['relative_path']}` | {item['category']} | {item['size_bytes']:,} | {str(item['upload_eligible']).lower()} |")
    lines += ["", "## Asset extensions", "", f"- Model assets: {', '.join(manifest['extensions']['model_assets']) or 'None found'}", f"- Dataset assets: {', '.join(manifest['extensions']['datasets']) or 'None found'}", f"- Image assets: {', '.join(manifest['extensions']['images']) or 'None found'}", "", "## Sensitive-path audit", ""]
    if sensitive:
        lines += ["Only paths and finding types are recorded; matched values are never emitted.", "", "| Relative path | Finding types | Ignored by Git |", "|---|---|---|"]
        for item in sensitive:
            lines.append(f"| `{item['relative_path']}` | {', '.join(item['finding_types'])} | {str(item['git_ignored']).lower()} |")
    else:
        lines.append("No potentially sensitive files or content patterns were detected by the bounded scanner.")
    lines += ["", "## Never upload", "", "`.git/`, `.venv/`, `venv/`, `**/__pycache__/`, `.pytest_cache/`, `.pytest-tmp/`, `node_modules/`, temporary/build folders, logs, and local credential files are ineligible.", ""]
    return "\n".join(lines)


def build_manifest() -> tuple[dict, list[dict]]:
    assets = []
    sensitive = []
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    model_exts, dataset_exts, image_exts = set(), set(), set()
    for path in iter_files():
        rel = normalized(path)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        ext = path.suffix.lower()
        category = classify(rel, ext)
        upload_eligible = eligible(category, rel)
        item = {
            "relative_path": rel,
            "category": category,
            "size_bytes": size,
            "file_extension": ext,
            "suggested_google_cloud_destination": destination(category),
            "sensitivity_classification": sensitivity(category, rel),
            "upload_eligible": upload_eligible,
        }
        assets.append(item)
        totals[category][0] += 1
        totals[category][1] += size
        if category in {"model checkpoints", "pretrained models", "candidate models", "promoted models"}:
            model_exts.add(ext)
        if category in {"synthetic datasets", "real-world datasets", "verified exports", "evaluation holdouts"} and ext in DATASET_EXTENSIONS:
            dataset_exts.add(ext)
        if ext in IMAGE_EXTENSIONS:
            image_exts.add(ext)
        findings = scan_sensitive(path, rel, category)
        if findings:
            sensitive.append({"relative_path": rel, "finding_types": findings, "git_ignored": git_ignored(rel)})
    assets.sort(key=lambda value: value["relative_path"])
    category_summary = [{"category": key, "file_count": totals[key][0], "total_size_bytes": totals[key][1]} for key in sorted(ALL_CATEGORIES)]
    manifest = {
        "schema_version": "1.0",
        "project_id": "fashionai-501816",
        "primary_region": "northamerica-northeast2",
        "existing_participant_photo_bucket": "fashionai-body-data-uploads",
        "metadata_only": True,
        "excluded_from_manifest": [".git/ (version-control internals; never upload)"],
        "summary": {
            "file_count": len(assets),
            "total_size_bytes": sum(item["size_bytes"] for item in assets),
            "upload_eligible_file_count": sum(item["upload_eligible"] for item in assets),
        },
        "category_summary": category_summary,
        "extensions": {"model_assets": sorted(model_exts), "datasets": sorted(dataset_exts), "images": sorted(image_exts)},
        "largest_files": sorted(assets, key=lambda value: value["size_bytes"], reverse=True)[:25],
        "assets": assets,
    }
    return manifest, sensitive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print upload-eligible metadata without writing or uploading anything")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    manifest, sensitive = build_manifest()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "would_upload": [item for item in manifest["assets"] if item["upload_eligible"]]}, indent=2))
        return 0
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.inventory.write_text(render_inventory(manifest, sensitive), encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest.relative_to(REPO_ROOT)), "inventory": str(args.inventory.relative_to(REPO_ROOT)), "files": manifest["summary"]["file_count"], "bytes": manifest["summary"]["total_size_bytes"], "sensitive_findings": len(sensitive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
