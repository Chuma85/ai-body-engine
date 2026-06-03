from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_blend_label_visual_correlation import DEFAULT_MIN_ABS_CORRELATION
from scripts.verify_phase_3h_d_blend_dataset_scale import DEFAULT_BLEND_FILE, discover_blender_executable
from synthetic.blender.blend_dataset import (
    BODY_FACTOR_COLUMNS,
    LABEL_GENERATION_MODE,
    validate_factor_label_correlations,
    validate_shape_key_coupled_rows,
)

DEFAULT_DATASET = "data/synthetic/phase_3h_g_coupled_smoke"
DEFAULT_AUDIT_OUT = "artifacts/phase_3h_g_coupled_smoke_audit"
DEFAULT_CORRELATION_OUT = "artifacts/phase_3h_g_label_visual_correlation"
DEFAULT_SAMPLES = 25
DEFAULT_SEED = 42
DEFAULT_LABEL_NOISE_CM = 0.05
DEFAULT_FACTOR_MIN_ABS_CORRELATION = 0.70
TARGET_COLUMNS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
]
TRACEABILITY_COLUMNS = [
    "label_generation_mode",
    "synthetic_labels",
    "real_world_validated",
    "height_factor",
    "chest_factor",
    "waist_factor",
    "hip_factor",
    "shoulder_factor",
    "inseam_factor",
    "torso_width_factor",
    "leg_length_factor",
    "shape_key_values_json",
    "body_shape_profile_id",
]


def verify_shape_key_label_coupling(
    *,
    dataset: str = DEFAULT_DATASET,
    audit_out: str = DEFAULT_AUDIT_OUT,
    correlation_out: str = DEFAULT_CORRELATION_OUT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    blend_file: str = DEFAULT_BLEND_FILE,
    blender_executable: str | None = None,
    label_noise_cm: float = DEFAULT_LABEL_NOISE_CM,
    factor_min_abs_correlation: float = DEFAULT_FACTOR_MIN_ABS_CORRELATION,
) -> dict[str, Any]:
    blender = discover_blender_executable(blender_executable)
    if blender is None:
        raise RuntimeError(
            "Blender executable was not found. Install Blender, add blender to PATH, or pass "
            "--blender-executable with the full blender.exe path."
        )

    generation_command = build_generation_command(
        blender_executable=blender,
        dataset=dataset,
        samples=samples,
        seed=seed,
        blend_file=blend_file,
        label_noise_cm=label_noise_cm,
    )
    subprocess.run(generation_command, cwd=REPO_ROOT, check=True)

    audit_command = build_audit_command(dataset=dataset, audit_out=audit_out, samples=samples)
    subprocess.run(audit_command, cwd=REPO_ROOT, check=True)

    correlation_command = build_correlation_command(dataset=dataset, out=correlation_out)
    subprocess.run(correlation_command, cwd=REPO_ROOT, check=True)

    rows = read_labels(Path(dataset) / "labels.csv")
    traceability_errors = validate_traceability(rows)
    if traceability_errors:
        raise ValueError("Traceability validation failed: " + "; ".join(traceability_errors))
    coupling_errors = validate_shape_key_coupled_rows(rows)
    if coupling_errors:
        raise ValueError("Shape-key coupled row validation failed: " + "; ".join(coupling_errors))
    factor_correlation = validate_factor_label_correlations(rows, min_abs_correlation=factor_min_abs_correlation)
    if not factor_correlation["valid"]:
        raise ValueError(
            "Weak factor-to-label correlations: " + ", ".join(factor_correlation["weak_targets"])
        )
    label_correlation = summarize_label_pair_correlations(rows)
    label_correlation_errors = validate_label_to_label_correlation(label_correlation)
    if label_correlation_errors:
        raise ValueError("Label-to-label correlation validation failed: " + "; ".join(label_correlation_errors))

    audit_report = json.loads((Path(audit_out) / "audit_report.json").read_text(encoding="utf-8"))
    correlation_report = json.loads((Path(correlation_out) / "correlation_report.json").read_text(encoding="utf-8"))
    metadata = json.loads((Path(dataset) / "metadata.json").read_text(encoding="utf-8"))
    strongest_visual = correlation_report["strongest_visual_correlation_by_target"]
    remaining_weak_targets = correlation_report["weakly_learnable_targets"]
    return {
        "dataset": dataset,
        "audit_out": audit_out,
        "correlation_out": correlation_out,
        "sample_count": len(rows),
        "image_count": len(list((Path(dataset) / "images").glob("*.png"))),
        "audit_passed": bool(audit_report.get("passed")),
        "label_generation_mode": metadata.get("label_generation_mode"),
        "label_formula_version": metadata.get("label_formula_version"),
        "traceability_columns": TRACEABILITY_COLUMNS,
        "factor_correlations": factor_correlation["correlations"],
        "label_pair_correlations": label_correlation,
        "strongest_visual_correlation_by_target": strongest_visual,
        "remaining_weak_targets": remaining_weak_targets,
        "generation_command": " ".join(generation_command),
        "audit_command": " ".join(audit_command),
        "correlation_command": " ".join(correlation_command),
    }


def build_generation_command(
    *,
    blender_executable: str,
    dataset: str,
    samples: int,
    seed: int,
    blend_file: str,
    label_noise_cm: float,
) -> list[str]:
    return [
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
        "--label-noise-cm",
        str(label_noise_cm),
        "--overwrite",
    ]


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


def build_correlation_command(*, dataset: str, out: str) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_blend_label_visual_correlation.py",
        "--dataset",
        dataset,
        "--out",
        out,
        "--target-columns",
        *TARGET_COLUMNS,
        "--min-abs-correlation",
        str(DEFAULT_MIN_ABS_CORRELATION),
    ]


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))


def validate_traceability(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["labels.csv has no rows"]
    columns = set(rows[0])
    missing_columns = [column for column in TRACEABILITY_COLUMNS if column not in columns]
    if missing_columns:
        errors.append("labels.csv missing traceability columns: " + ", ".join(missing_columns))
    for row in rows:
        sample_id = row.get("sample_id", "<missing sample_id>")
        if row.get("label_generation_mode") != LABEL_GENERATION_MODE:
            errors.append(f"{sample_id}: label_generation_mode is not {LABEL_GENERATION_MODE}")
        if row.get("synthetic_labels") != "true":
            errors.append(f"{sample_id}: synthetic_labels must be true")
        if row.get("real_world_validated") != "false":
            errors.append(f"{sample_id}: real_world_validated must be false")
        for column in BODY_FACTOR_COLUMNS:
            if row.get(column, "") == "":
                errors.append(f"{sample_id}: missing {column}")
        if not row.get("shape_key_values_json"):
            errors.append(f"{sample_id}: missing shape_key_values_json")
    return errors


def summarize_label_pair_correlations(rows: list[dict[str, str]]) -> dict[str, float | None]:
    pairs = {
        "height_cm:inseam_cm": ("height_cm", "inseam_cm"),
        "chest_cm:waist_cm": ("chest_cm", "waist_cm"),
        "chest_cm:hip_cm": ("chest_cm", "hip_cm"),
        "waist_cm:hip_cm": ("waist_cm", "hip_cm"),
        "chest_cm:shoulder_cm": ("chest_cm", "shoulder_cm"),
    }
    return {name: pearson(rows, left, right) for name, (left, right) in pairs.items()}


def validate_label_to_label_correlation(correlations: dict[str, float | None]) -> list[str]:
    errors: list[str] = []
    for pair, value in correlations.items():
        if value is None or value < 0.20:
            errors.append(f"{pair} correlation is too low: {value}")
        if value is not None and value > 0.995:
            errors.append(f"{pair} correlation is nearly identical: {value}")
    return errors


def pearson(rows: list[dict[str, str]], left: str, right: str) -> float | None:
    left_values = [float(row[left]) for row in rows]
    right_values = [float(row[right]) for row in rows]
    if len(left_values) < 2:
        return None
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_delta = [value - left_mean for value in left_values]
    right_delta = [value - right_mean for value in right_values]
    left_norm = sum(value * value for value in left_delta) ** 0.5
    right_norm = sum(value * value for value in right_delta) ** 0.5
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / (left_norm * right_norm)


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-G shape-key label coupling verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Samples: {summary['sample_count']}",
        f"Images: {summary['image_count']}",
        f"Strict audit passed: {summary['audit_passed']}",
        f"Label generation mode: {summary['label_generation_mode']}",
        f"Label formula version: {summary['label_formula_version']}",
        "Strongest factor correlation per target:",
    ]
    for target, value in summary["factor_correlations"].items():
        lines.append(f"  {target}: {float(value):.4f}")
    lines.append("Strongest visual correlation per target:")
    for target, row in summary["strongest_visual_correlation_by_target"].items():
        lines.append(f"  {target}: {float(row['abs_max_correlation']):.4f} via {row['feature']}")
    lines.append("Remaining weak visual targets: " + (", ".join(summary["remaining_weak_targets"]) or "none"))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Phase 3H-G shape-key coupled label generation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--correlation-out", default=DEFAULT_CORRELATION_OUT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--blend-file", default=DEFAULT_BLEND_FILE)
    parser.add_argument("--blender-executable", default=None)
    parser.add_argument("--label-noise-cm", type=float, default=DEFAULT_LABEL_NOISE_CM)
    parser.add_argument("--factor-min-abs-correlation", type=float, default=DEFAULT_FACTOR_MIN_ABS_CORRELATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_shape_key_label_coupling(
            dataset=args.dataset,
            audit_out=args.audit_out,
            correlation_out=args.correlation_out,
            samples=args.samples,
            seed=args.seed,
            blend_file=args.blend_file,
            blender_executable=args.blender_executable,
            label_noise_cm=args.label_noise_cm,
            factor_min_abs_correlation=args.factor_min_abs_correlation,
        )
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-G shape-key label coupling verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
