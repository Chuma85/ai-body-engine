from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLEND_FILE = "assets/body_meshes/base_body_scene.blend"
DEFAULT_DATASET = "data/synthetic/phase_3h_blend_250"
DEFAULT_AUDIT_OUT = "artifacts/phase_3h_blend_250_audit"
DEFAULT_SAMPLES = 250
DEFAULT_SEED = 42
REQUIRED_DATASET_FILES = [
    "labels.csv",
    "metadata.json",
]
REQUIRED_AUDIT_FILES = [
    "audit_report.json",
    "audit_summary.md",
    "sample_contact_sheet.png",
    "label_distribution_summary.csv",
    "flagged_samples.csv",
]


def discover_blender_executable(explicit: str | None = None) -> str | None:
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists() or shutil.which(explicit):
            return explicit
        return None

    from_path = shutil.which("blender")
    if from_path:
        return from_path

    foundation_dir = Path("C:/Program Files/Blender Foundation")
    if foundation_dir.exists():
        candidates = sorted(
            foundation_dir.glob("Blender */blender.exe"),
            key=lambda path: path.as_posix(),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return None


def build_generation_command(
    *,
    blender_executable: str,
    dataset: str,
    samples: int,
    seed: int,
    blend_file: str,
    overwrite: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/generate_blend_dataset.py",
        "--source",
        "blend",
        "--blend-file",
        blend_file,
        "--out",
        dataset,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--blender-executable",
        blender_executable,
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def build_audit_command(*, dataset: str, audit_out: str, samples: int) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_blend_dataset.py",
        "--dataset",
        dataset,
        "--out",
        audit_out,
        "--expected-samples",
        str(samples),
        "--strict",
    ]


def dataset_complete(dataset: str | Path, samples: int) -> tuple[bool, list[str]]:
    dataset_path = Path(dataset)
    missing = [str(dataset_path / relative_path) for relative_path in REQUIRED_DATASET_FILES if not (dataset_path / relative_path).exists()]
    images_dir = dataset_path / "images"
    image_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    expected_images = samples * 3
    if image_count != expected_images:
        missing.append(f"expected {expected_images} PNG images, found {image_count}")
    return not missing, missing


def required_outputs_exist(dataset: str | Path, audit_out: str | Path) -> tuple[bool, list[str]]:
    dataset_path = Path(dataset)
    audit_path = Path(audit_out)
    missing = [
        str(dataset_path / relative_path)
        for relative_path in REQUIRED_DATASET_FILES
        if not (dataset_path / relative_path).exists()
    ]
    missing.extend(
        str(audit_path / relative_path)
        for relative_path in REQUIRED_AUDIT_FILES
        if not (audit_path / relative_path).exists()
    )
    return not missing, missing


def summarize_verification(
    *,
    dataset: str | Path,
    audit_out: str | Path,
    samples: int,
    duration_seconds: float,
    generated: bool,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    audit_report_path = Path(audit_out) / "audit_report.json"
    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    actual_image_count = len(list((dataset_path / "images").glob("*.png")))
    expected_image_count = samples * 3
    return {
        "dataset": str(dataset_path),
        "audit_out": str(audit_out),
        "generated": generated,
        "sample_count": report["row_count"],
        "expected_image_count": expected_image_count,
        "actual_image_count": actual_image_count,
        "strict_audit_passed": bool(report["passed"]),
        "warnings_count": len(report["warnings"]),
        "errors_count": len(report["errors"]),
        "strict_failures_count": len(report["strict_failures"]),
        "flagged_sample_count": int(report["flagged_sample_count"]),
        "view_sanity_passed": bool(report["view_sanity"]["passed"]),
        "label_variation_exists": bool(report["label_audit"]["variation_exists"]),
        "variation_source": report["metadata"].get("variation_source"),
        "shape_key_count": report["metadata"].get("shape_key_count"),
        "duration_seconds": round(duration_seconds, 2),
    }


def verify_scaled_blend_dataset(
    *,
    dataset: str = DEFAULT_DATASET,
    audit_out: str = DEFAULT_AUDIT_OUT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    blend_file: str = DEFAULT_BLEND_FILE,
    blender_executable: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    blender = discover_blender_executable(blender_executable)
    if blender is None:
        raise RuntimeError(
            "Blender executable was not found. Install Blender, add blender to PATH, or pass "
            "--blender-executable with the full blender.exe path. Manual generation command: "
            f"python scripts/generate_blend_dataset.py --source blend --blend-file {blend_file} "
            f"--out {dataset} --samples {samples} --seed {seed}"
        )

    start_time = time.perf_counter()
    dataset_path = Path(dataset)
    generated = False
    if dataset_path.exists() and any(dataset_path.iterdir()):
        complete, problems = dataset_complete(dataset_path, samples)
        if not complete:
            raise RuntimeError(
                f"Dataset already exists but is incomplete: {dataset_path}. "
                f"Problems: {'; '.join(problems)}. Pass --overwrite to regenerate."
            )
        if overwrite:
            generation_command = build_generation_command(
                blender_executable=blender,
                dataset=dataset,
                samples=samples,
                seed=seed,
                blend_file=blend_file,
                overwrite=True,
            )
            subprocess.run(generation_command, cwd=ROOT, check=True)
            generated = True
    else:
        generation_command = build_generation_command(
            blender_executable=blender,
            dataset=dataset,
            samples=samples,
            seed=seed,
            blend_file=blend_file,
            overwrite=False,
        )
        subprocess.run(generation_command, cwd=ROOT, check=True)
        generated = True

    audit_command = build_audit_command(dataset=dataset, audit_out=audit_out, samples=samples)
    subprocess.run(audit_command, cwd=ROOT, check=True)
    complete, missing = required_outputs_exist(dataset, audit_out)
    if not complete:
        raise RuntimeError("Missing required verification outputs: " + "; ".join(missing))

    summary = summarize_verification(
        dataset=dataset,
        audit_out=audit_out,
        samples=samples,
        duration_seconds=time.perf_counter() - start_time,
        generated=generated,
    )
    if summary["sample_count"] != samples:
        raise RuntimeError(f"Expected {samples} samples, got {summary['sample_count']}.")
    if summary["actual_image_count"] != summary["expected_image_count"]:
        raise RuntimeError(
            f"Expected {summary['expected_image_count']} images, got {summary['actual_image_count']}."
        )
    if not summary["strict_audit_passed"]:
        raise RuntimeError("Strict audit did not pass. See audit_report.json for details.")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-D scaled blend dataset verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Audit artifacts: {summary['audit_out']}",
        f"Generated this run: {summary['generated']}",
        f"Samples: {summary['sample_count']}",
        f"Images: {summary['actual_image_count']} / {summary['expected_image_count']}",
        f"Strict audit passed: {summary['strict_audit_passed']}",
        f"Warnings: {summary['warnings_count']}",
        f"Errors: {summary['errors_count']}",
        f"Strict failures: {summary['strict_failures_count']}",
        f"Flagged samples: {summary['flagged_sample_count']}",
        f"Front/side/back sanity passed: {summary['view_sanity_passed']}",
        f"Label variation exists: {summary['label_variation_exists']}",
        f"Variation source: {summary['variation_source']}",
        f"Shape key count: {summary['shape_key_count']}",
        f"Runtime seconds: {summary['duration_seconds']}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and strictly audit the Phase 3H-D 250-sample blend dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--blend-file", default=DEFAULT_BLEND_FILE)
    parser.add_argument("--blender-executable", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_scaled_blend_dataset(
            dataset=args.dataset,
            audit_out=args.audit_out,
            samples=args.samples,
            seed=args.seed,
            blend_file=args.blend_file,
            blender_executable=args.blender_executable,
            overwrite=args.overwrite,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-D scaled blend dataset verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
