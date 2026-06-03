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

from scripts.audit_blend_label_visual_correlation import (
    DEFAULT_MIN_ABS_CORRELATION,
    DEFAULT_OUT,
    REQUIRED_OUTPUTS,
    build_audit_command,
)
from scripts.train_blend_dataset_baseline import DEFAULT_DATASET, DEFAULT_TARGET_COLUMNS


def verify_phase_3h_f_label_visual_correlation(
    *,
    dataset: str = DEFAULT_DATASET,
    out: str = DEFAULT_OUT,
    target_columns: list[str] | None = None,
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
) -> dict[str, Any]:
    targets = target_columns or [*DEFAULT_TARGET_COLUMNS]
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
    command = build_audit_command(
        dataset=dataset,
        out=out,
        target_columns=targets,
        min_abs_correlation=min_abs_correlation,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return verify_outputs(out, command)


def verify_outputs(out: str | Path, command: list[str]) -> dict[str, Any]:
    output_dir = Path(out)
    missing = [str(output_dir / filename) for filename in REQUIRED_OUTPUTS if not (output_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3H-F artifacts: " + "; ".join(missing))

    report = json.loads((output_dir / "correlation_report.json").read_text(encoding="utf-8"))
    required_keys = {
        "dataset",
        "sample_count",
        "target_columns",
        "strongest_visual_correlation_by_target",
        "weakly_learnable_targets",
        "suspicious_label_behavior",
        "suspicious_visual_feature_behavior",
        "recommended_next_action",
    }
    missing_keys = sorted(required_keys.difference(report))
    if missing_keys:
        raise ValueError("correlation_report.json missing required keys: " + ", ".join(missing_keys))

    csv_rows = {
        "feature_label_correlation.csv": _read_csv(output_dir / "feature_label_correlation.csv"),
        "target_correlation_matrix.csv": _read_csv(output_dir / "target_correlation_matrix.csv"),
        "visual_feature_summary.csv": _read_csv(output_dir / "visual_feature_summary.csv"),
        "label_summary.csv": _read_csv(output_dir / "label_summary.csv"),
        "top_features_by_target.csv": _read_csv(output_dir / "top_features_by_target.csv"),
    }
    empty = [name for name, rows in csv_rows.items() if not rows]
    if empty:
        raise ValueError("Correlation CSVs have no rows: " + ", ".join(empty))

    return {
        "output_dir": str(output_dir),
        "training_free_audit_command": " ".join(command),
        "dataset": report["dataset"],
        "sample_count": report["sample_count"],
        "strongest_visual_correlation_by_target": report["strongest_visual_correlation_by_target"],
        "weakly_learnable_targets": report["weakly_learnable_targets"],
        "flagged_targets": report["flagged_targets"],
        "recommended_next_action": report["recommended_next_action"],
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3H-F label visual correlation verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Output: {summary['output_dir']}",
        f"Samples: {summary['sample_count']}",
        "Strongest visual correlation per target:",
    ]
    for target, row in summary["strongest_visual_correlation_by_target"].items():
        lines.append(f"  {target}: {float(row['abs_max_correlation']):.4f} via {row['feature']}")
    lines.append("Flagged targets/features:")
    if summary["flagged_targets"]:
        for row in summary["flagged_targets"]:
            lines.append(f"  {row['category']} {row['target_or_feature']}: {row['detail']} ({row['value']})")
    else:
        lines.append("  none")
    lines.append("Recommended next action: " + summary["recommended_next_action"])
    lines.append("Audit command: " + summary["training_free_audit_command"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Phase 3H-F label visual correlation audit artifacts.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--target-columns", nargs="+", default=[*DEFAULT_TARGET_COLUMNS])
    parser.add_argument("--min-abs-correlation", type=float, default=DEFAULT_MIN_ABS_CORRELATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_phase_3h_f_label_visual_correlation(
            dataset=args.dataset,
            out=args.out,
            target_columns=args.target_columns,
            min_abs_correlation=args.min_abs_correlation,
        )
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-F label visual correlation verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    raise SystemExit(main())
