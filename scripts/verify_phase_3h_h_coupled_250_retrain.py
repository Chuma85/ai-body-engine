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

from scripts.verify_phase_3h_d_blend_dataset_scale import DEFAULT_BLEND_FILE, discover_blender_executable
from synthetic.blender.blend_dataset import (
    BODY_FACTOR_COLUMNS,
    LABEL_FORMULA_VERSION,
    LABEL_GENERATION_MODE,
    validate_shape_key_coupled_rows,
)

DEFAULT_DATASET = "data/synthetic/phase_3h_h_coupled_250"
DEFAULT_AUDIT_OUT = "artifacts/phase_3h_h_coupled_250_audit"
DEFAULT_CORRELATION_OUT = "artifacts/phase_3h_h_label_visual_correlation"
DEFAULT_TRAINING_OUT = "artifacts/phase_3h_h_blend_baseline"
DEFAULT_SAMPLES = 250
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_ABS_CORRELATION = 0.25
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
    "shape_key_values_json",
    "height_factor",
    "chest_factor",
    "waist_factor",
    "hip_factor",
    "shoulder_factor",
    "inseam_factor",
    "torso_width_factor",
    "leg_length_factor",
]
PHASE_3H_E_BASELINE = {
    "dataset": "data/synthetic/phase_3h_blend_250",
    "best_model": "random_forest",
    "overall_mean_mae": 13.1239,
    "mae_by_target": {
        "height_cm": 14.0280,
        "chest_cm": 15.7985,
        "waist_cm": 17.3995,
        "hip_cm": 17.9524,
        "shoulder_cm": 6.1045,
        "inseam_cm": 7.4608,
    },
}


def verify_coupled_250_retrain(
    *,
    dataset: str = DEFAULT_DATASET,
    audit_out: str = DEFAULT_AUDIT_OUT,
    correlation_out: str = DEFAULT_CORRELATION_OUT,
    training_out: str = DEFAULT_TRAINING_OUT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    blend_file: str = DEFAULT_BLEND_FILE,
    blender_executable: str | None = None,
    overwrite: bool = True,
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
        overwrite=overwrite,
    )
    subprocess.run(generation_command, cwd=REPO_ROOT, check=True)
    dataset_summary = validate_coupled_dataset_outputs(dataset, expected_samples=samples)

    audit_command = build_audit_command(dataset=dataset, audit_out=audit_out, samples=samples)
    subprocess.run(audit_command, cwd=REPO_ROOT, check=True)
    audit_summary = read_audit_summary(Path(audit_out) / "audit_report.json")

    correlation_command = build_correlation_command(dataset=dataset, out=correlation_out)
    subprocess.run(correlation_command, cwd=REPO_ROOT, check=True)
    correlation_report = json.loads((Path(correlation_out) / "correlation_report.json").read_text(encoding="utf-8"))

    audit_report = str(Path(audit_out) / "audit_report.json")
    training_command = build_training_command(
        dataset=dataset,
        out=training_out,
        seed=seed,
        test_size=test_size,
        audit_report=audit_report,
    )
    subprocess.run(training_command, cwd=REPO_ROOT, check=True)
    training_metrics = json.loads((Path(training_out) / "metrics.json").read_text(encoding="utf-8"))
    comparison = compare_metrics_to_phase_3h_e(training_metrics)

    return {
        "dataset": dataset,
        "audit_out": audit_out,
        "correlation_out": correlation_out,
        "training_out": training_out,
        "sample_count": dataset_summary["sample_count"],
        "image_count": dataset_summary["image_count"],
        "label_generation_mode": dataset_summary["label_generation_mode"],
        "label_formula_version": dataset_summary["label_formula_version"],
        "synthetic_labels": dataset_summary["synthetic_labels"],
        "real_world_validated": dataset_summary["real_world_validated"],
        "audit": audit_summary,
        "strongest_visual_correlation_by_target": correlation_report["strongest_visual_correlation_by_target"],
        "weak_targets_below_threshold": correlation_report["weakly_learnable_targets"],
        "best_model": training_metrics["best_model"],
        "overall_mean_mae": training_metrics["overall_mean_mae"],
        "mae_by_target": training_metrics["mae_by_target"],
        "comparison_to_phase_3h_e": comparison,
        "commands": {
            "generation": " ".join(generation_command),
            "audit": " ".join(audit_command),
            "correlation": " ".join(correlation_command),
            "training": " ".join(training_command),
        },
    }


def build_generation_command(
    *,
    blender_executable: str,
    dataset: str,
    samples: int,
    seed: int,
    blend_file: str,
    overwrite: bool,
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


def build_training_command(
    *,
    dataset: str,
    out: str,
    seed: int,
    test_size: float,
    audit_report: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_blend_dataset_baseline.py",
        "--dataset",
        dataset,
        "--out",
        out,
        "--seed",
        str(seed),
        "--test-size",
        str(test_size),
        "--target-columns",
        *TARGET_COLUMNS,
        "--strict-audit-required",
        "--audit-report",
        audit_report,
    ]


def validate_coupled_dataset_outputs(dataset: str | Path, expected_samples: int) -> dict[str, Any]:
    dataset_path = Path(dataset)
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    for path in (dataset_path, labels_path, metadata_path, images_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required coupled dataset path: {path}")

    rows = read_labels(labels_path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} labels, found {len(rows)}")
    image_count = len(list(images_dir.glob("*.png")))
    expected_images = expected_samples * 3
    if image_count != expected_images:
        raise ValueError(f"Expected {expected_images} PNG images, found {image_count}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_columns = [column for column in TRACEABILITY_COLUMNS if column not in rows[0]]
    if missing_columns:
        raise ValueError("labels.csv missing traceability columns: " + ", ".join(missing_columns))
    row_errors = validate_shape_key_coupled_rows(rows)
    if row_errors:
        raise ValueError("Shape-key coupled label validation failed: " + "; ".join(row_errors))
    for column in BODY_FACTOR_COLUMNS:
        if column not in rows[0]:
            raise ValueError(f"labels.csv missing factor column: {column}")
    if metadata.get("label_generation_mode") != LABEL_GENERATION_MODE:
        raise ValueError(f"metadata label_generation_mode is not {LABEL_GENERATION_MODE}")
    if metadata.get("label_formula_version") != LABEL_FORMULA_VERSION:
        raise ValueError(f"metadata label_formula_version is not {LABEL_FORMULA_VERSION}")
    if metadata.get("synthetic_labels") is not True:
        raise ValueError("metadata synthetic_labels must be true")
    if metadata.get("real_world_validated") is not False:
        raise ValueError("metadata real_world_validated must be false")

    return {
        "sample_count": len(rows),
        "image_count": image_count,
        "label_generation_mode": metadata["label_generation_mode"],
        "label_formula_version": metadata["label_formula_version"],
        "synthetic_labels": metadata["synthetic_labels"],
        "real_world_validated": metadata["real_world_validated"],
    }


def read_audit_summary(audit_report_path: Path) -> dict[str, Any]:
    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if report.get("strict") is not True:
        raise ValueError(f"Audit report was not strict: {audit_report_path}")
    if report.get("passed") is not True:
        raise ValueError(f"Strict audit did not pass: {audit_report_path}")
    return {
        "passed": bool(report.get("passed")),
        "warnings_count": len(report.get("warnings", [])),
        "errors_count": len(report.get("errors", [])),
        "strict_failures_count": len(report.get("strict_failures", [])),
        "flagged_sample_count": int(report.get("flagged_sample_count", 0)),
    }


def compare_metrics_to_phase_3h_e(metrics: dict[str, Any]) -> dict[str, Any]:
    new_overall = float(metrics["overall_mean_mae"])
    old_overall = float(PHASE_3H_E_BASELINE["overall_mean_mae"])
    target_comparisons = {}
    for target, old_value in PHASE_3H_E_BASELINE["mae_by_target"].items():
        new_value = float(metrics["mae_by_target"][target])
        delta = new_value - float(old_value)
        target_comparisons[target] = {
            "phase_3h_e_mae": float(old_value),
            "phase_3h_h_mae": new_value,
            "delta": delta,
            "status": "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged",
        }
    overall_delta = new_overall - old_overall
    return {
        "phase_3h_e_best_model": PHASE_3H_E_BASELINE["best_model"],
        "phase_3h_e_overall_mean_mae": old_overall,
        "phase_3h_h_overall_mean_mae": new_overall,
        "overall_delta": overall_delta,
        "overall_status": "improved" if overall_delta < 0 else "worsened" if overall_delta > 0 else "unchanged",
        "mae_by_target": target_comparisons,
    }


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-H coupled 250 retraining verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Samples: {summary['sample_count']}",
        f"Images: {summary['image_count']}",
        f"Strict audit passed: {summary['audit']['passed']}",
        f"Label generation mode: {summary['label_generation_mode']}",
        f"Best model: {summary['best_model']}",
        f"Overall mean MAE: {summary['overall_mean_mae']:.4f}",
        (
            "Comparison vs Phase 3H-E: "
            f"{summary['comparison_to_phase_3h_e']['overall_status']} "
            f"({summary['comparison_to_phase_3h_e']['overall_delta']:.4f} cm)"
        ),
        "MAE by target:",
    ]
    for target, value in summary["mae_by_target"].items():
        delta = summary["comparison_to_phase_3h_e"]["mae_by_target"][target]["delta"]
        status = summary["comparison_to_phase_3h_e"]["mae_by_target"][target]["status"]
        lines.append(f"  {target}: {float(value):.4f} ({status}, delta={delta:.4f})")
    lines.append("Strongest visual correlation per target:")
    for target, row in summary["strongest_visual_correlation_by_target"].items():
        lines.append(f"  {target}: {float(row['abs_max_correlation']):.4f} via {row['feature']}")
    lines.append("Weak targets below 0.25: " + (", ".join(summary["weak_targets_below_threshold"]) or "none"))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate coupled 250-sample Blender dataset and retrain baselines.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--correlation-out", default=DEFAULT_CORRELATION_OUT)
    parser.add_argument("--training-out", default=DEFAULT_TRAINING_OUT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--blend-file", default=DEFAULT_BLEND_FILE)
    parser.add_argument("--blender-executable", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_coupled_250_retrain(
            dataset=args.dataset,
            audit_out=args.audit_out,
            correlation_out=args.correlation_out,
            training_out=args.training_out,
            samples=args.samples,
            seed=args.seed,
            test_size=args.test_size,
            blend_file=args.blend_file,
            blender_executable=args.blender_executable,
            overwrite=not args.no_overwrite,
        )
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-H coupled 250 retraining verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
