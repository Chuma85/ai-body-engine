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

from scripts.train_blend_dataset_baseline import (
    DEFAULT_DATASET,
    DEFAULT_OUT,
    DEFAULT_SEED,
    DEFAULT_TARGET_COLUMNS,
    DEFAULT_TEST_SIZE,
    REQUIRED_OUTPUTS,
    build_training_command,
    default_audit_report_for_dataset,
)


def confirm_dataset_and_audit(dataset: str | Path, audit_report: str | Path | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    for path in (labels_path, metadata_path, images_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset path: {path}")

    audit_path = Path(audit_report) if audit_report is not None else default_audit_report_for_dataset(dataset_path)
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing audit report: {audit_path}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit_payload.get("passed")):
        raise ValueError(f"Audit report did not pass: {audit_path}")
    image_count = len(list(images_dir.glob("*.png")))
    return {
        "dataset": str(dataset_path),
        "audit_report": str(audit_path),
        "sample_count": int(audit_payload.get("row_count", 0)),
        "image_count": image_count,
        "strict_audit_passed": bool(audit_payload.get("passed")),
    }


def verify_expected_artifacts(out: str | Path) -> dict[str, Any]:
    output_dir = Path(out)
    missing = [str(output_dir / filename) for filename in REQUIRED_OUTPUTS if not (output_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3H-E artifacts: " + "; ".join(missing))

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    required_metric_keys = {
        "best_model",
        "dataset",
        "image_count",
        "mae_by_target",
        "model_ranking",
        "models",
        "overall_mean_mae",
        "real_world_validated",
        "sample_count",
        "target_columns",
    }
    missing_keys = sorted(required_metric_keys.difference(metrics))
    if missing_keys:
        raise ValueError("metrics.json missing required keys: " + ", ".join(missing_keys))
    if metrics["real_world_validated"] is not False:
        raise ValueError("metrics.json must set real_world_validated=false")

    predictions_rows = _read_csv(output_dir / "predictions.csv")
    if not predictions_rows:
        raise ValueError("predictions.csv has no prediction rows")
    ranking_rows = _read_csv(output_dir / "model_ranking.csv")
    if not ranking_rows:
        raise ValueError("model_ranking.csv has no ranking rows")
    ranks = [int(row["rank"]) for row in ranking_rows]
    if ranks != sorted(ranks) or ranks[0] != 1:
        raise ValueError("model_ranking.csv does not rank models from 1")

    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "prediction_rows": len(predictions_rows),
        "ranked_models": [row["model"] for row in ranking_rows],
    }


def verify_phase_3h_e_blend_baseline(
    *,
    dataset: str = DEFAULT_DATASET,
    out: str = DEFAULT_OUT,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    target_columns: list[str] | None = None,
    audit_report: str | None = None,
) -> dict[str, Any]:
    targets = target_columns or [*DEFAULT_TARGET_COLUMNS]
    dataset_summary = confirm_dataset_and_audit(dataset, audit_report)
    command = build_training_command(
        dataset=dataset,
        out=out,
        seed=seed,
        test_size=test_size,
        target_columns=targets,
        strict_audit_required=True,
        audit_report=audit_report,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    artifact_summary = verify_expected_artifacts(out)
    metrics = artifact_summary["metrics"]
    return {
        **dataset_summary,
        "training_command": " ".join(command),
        "output_dir": artifact_summary["output_dir"],
        "best_model": metrics["best_model"],
        "overall_mean_mae": metrics["overall_mean_mae"],
        "mae_by_target": metrics["mae_by_target"],
        "prediction_rows": artifact_summary["prediction_rows"],
        "ranked_models": artifact_summary["ranked_models"],
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-E blend baseline verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Audit report: {summary['audit_report']}",
        f"Output: {summary['output_dir']}",
        f"Samples: {summary['sample_count']}",
        f"Images: {summary['image_count']}",
        f"Strict audit passed: {summary['strict_audit_passed']}",
        f"Best model: {summary['best_model']}",
        f"Overall mean MAE: {summary['overall_mean_mae']:.4f}",
        f"Prediction rows: {summary['prediction_rows']}",
        f"Ranked models: {', '.join(summary['ranked_models'])}",
        "MAE by target:",
    ]
    for target, mae in summary["mae_by_target"].items():
        lines.append(f"  {target}: {mae:.4f}")
    lines.append(f"Training command: {summary['training_command']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Phase 3H-E blend baseline training artifacts.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--target-columns", nargs="+", default=[*DEFAULT_TARGET_COLUMNS])
    parser.add_argument("--audit-report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_phase_3h_e_blend_baseline(
            dataset=args.dataset,
            out=args.out,
            seed=args.seed,
            test_size=args.test_size,
            target_columns=args.target_columns,
            audit_report=args.audit_report,
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-E blend baseline verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    raise SystemExit(main())
