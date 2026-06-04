from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.features.image_silhouette_features import (
    create_foreground_mask,
    extract_image_features,
    load_rgb_image,
)

DEFAULT_DATASET = "data/synthetic/phase_3h_blend_250"
DEFAULT_OUT = "artifacts/phase_3h_e_blend_baseline"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TARGET_COLUMNS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
]
VIEW_COLUMNS = {
    "front": "front_image",
    "side": "side_image",
    "back": "back_image",
}
REQUIRED_OUTPUTS = [
    "metrics.json",
    "metrics_summary.md",
    "predictions.csv",
    "train_test_split.json",
    "feature_summary.csv",
    "model_ranking.csv",
    "best_model.joblib",
    "experiment_metadata.json",
]
FEATURE_EXTRACTOR_VERSION = "phase_3h_e_front_side_back_silhouette_v1"
SYNTHETIC_ONLY_WARNING = (
    "Synthetic-only Blender dataset results are not real-world validated and are not production tailoring accuracy."
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def default_audit_report_for_dataset(dataset: str | Path) -> Path:
    dataset_path = Path(dataset)
    return Path("artifacts") / f"{dataset_path.name}_audit" / "audit_report.json"


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))


def validate_blend_dataset(
    dataset: str | Path,
    target_columns: list[str],
    *,
    strict_audit_required: bool = False,
    audit_report: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    rows = read_labels(labels_path)
    if not rows:
        raise ValueError(f"labels.csv has no sample rows: {labels_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_columns = ["sample_id", *VIEW_COLUMNS.values(), *target_columns]
    missing_columns = [column for column in required_columns if column not in rows[0]]
    if missing_columns:
        raise ValueError("labels.csv missing required columns: " + ", ".join(missing_columns))

    numeric_values = {target: [] for target in target_columns}
    image_paths: dict[str, dict[str, Path]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        image_paths[sample_id] = {}
        for view, column in VIEW_COLUMNS.items():
            relative_path = row.get(column, "")
            if not relative_path:
                raise ValueError(f"{sample_id}: missing {column} value")
            image_path = dataset_path / relative_path
            if not image_path.exists():
                raise FileNotFoundError(f"{sample_id}: missing {view} image file: {image_path}")
            image_paths[sample_id][view] = image_path
        for target in target_columns:
            try:
                value = float(row[target])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{sample_id}: non-numeric target {target}={row.get(target)!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{sample_id}: non-finite target {target}={row.get(target)!r}")
            numeric_values[target].append(value)

    identical_targets = [
        target
        for target, values in numeric_values.items()
        if len({round(value, 6) for value in values}) <= 1
    ]
    if identical_targets:
        raise ValueError("Target columns have no label variation: " + ", ".join(identical_targets))

    audit_payload: dict[str, Any] | None = None
    audit_report_path = Path(audit_report) if audit_report is not None else default_audit_report_for_dataset(dataset_path)
    if strict_audit_required:
        if not audit_report_path.exists():
            raise FileNotFoundError(f"Missing strict audit report: {audit_report_path}")
        audit_payload = json.loads(audit_report_path.read_text(encoding="utf-8"))
        if not bool(audit_payload.get("passed")):
            raise ValueError(f"Strict audit did not pass: {audit_report_path}")
        if audit_payload.get("strict") is not True:
            raise ValueError(f"Audit report was not produced in strict mode: {audit_report_path}")

    image_count = sum(1 for _ in images_dir.rglob("*.png"))
    return {
        "dataset": str(dataset_path),
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
        "audit_report": str(audit_report_path),
        "rows": rows,
        "metadata": metadata,
        "audit": audit_payload,
        "image_paths": image_paths,
        "sample_count": len(rows),
        "image_count": image_count,
        "target_columns": target_columns,
        "label_variation_exists": True,
        "real_world_validated": bool(metadata.get("real_world_validated")),
        "synthetic_labels": bool(metadata.get("synthetic_labels", True)),
    }


def extract_blend_image_features(sample: dict[str, str], dataset: str | Path) -> dict[str, float]:
    dataset_path = Path(dataset)
    features: dict[str, float] = {}
    for view, column in VIEW_COLUMNS.items():
        image_path = dataset_path / sample[column]
        view_features = extract_image_features(image_path, view)
        features.update(view_features)
        features.update(extract_projection_features(image_path, view))
    features.update(extract_combined_view_features(features))
    return features


def extract_projection_features(image_path: str | Path, prefix: str, bins: int = 8) -> dict[str, float]:
    image = load_rgb_image(image_path)
    mask = create_foreground_mask(image)
    row_widths = mask.sum(axis=1).astype(np.float64)
    column_heights = mask.sum(axis=0).astype(np.float64)
    active_row_widths = row_widths[row_widths > 0]
    active_column_heights = column_heights[column_heights > 0]
    image_height, image_width = mask.shape
    features = {
        f"{prefix}_projection_row_width_mean": _safe_mean(active_row_widths) / float(image_width),
        f"{prefix}_projection_row_width_std": _safe_std(active_row_widths) / float(image_width),
        f"{prefix}_projection_row_width_max": float(row_widths.max()) / float(image_width),
        f"{prefix}_projection_column_height_mean": _safe_mean(active_column_heights) / float(image_height),
        f"{prefix}_projection_column_height_std": _safe_std(active_column_heights) / float(image_height),
        f"{prefix}_projection_column_height_max": float(column_heights.max()) / float(image_height),
    }
    for index, value in enumerate(_binned_nonzero_density(row_widths, bins=bins)):
        features[f"{prefix}_horizontal_projection_bin_{index:02d}"] = value
    for index, value in enumerate(_binned_nonzero_density(column_heights, bins=bins)):
        features[f"{prefix}_vertical_projection_bin_{index:02d}"] = value
    return features


def extract_combined_view_features(features: dict[str, float]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for metric in (
        "raw_bbox_width_ratio",
        "raw_bbox_height_ratio",
        "raw_mask_area_ratio",
        "bbox_aspect_ratio",
        "projection_row_width_mean",
        "projection_column_height_mean",
    ):
        front = features[f"front_{metric}"]
        side = features[f"side_{metric}"]
        back = features[f"back_{metric}"]
        combined[f"front_side_{metric}_ratio"] = _safe_ratio(front, side)
        combined[f"front_back_{metric}_ratio"] = _safe_ratio(front, back)
        combined[f"side_back_{metric}_ratio"] = _safe_ratio(side, back)
        combined[f"mean_{metric}"] = float(np.mean([front, side, back]))
    combined["front_side_back_area_proxy"] = (
        features["front_raw_mask_area_ratio"]
        * features["side_raw_mask_area_ratio"]
        * features["back_raw_mask_area_ratio"]
    )
    combined["front_side_back_bbox_volume_proxy"] = (
        features["front_raw_bbox_width_ratio"]
        * features["side_raw_bbox_width_ratio"]
        * features["back_raw_bbox_height_ratio"]
    )
    return combined


def build_feature_matrix(rows: list[dict[str, str]], dataset: str | Path) -> tuple[list[str], np.ndarray]:
    feature_rows = [extract_blend_image_features(row, dataset) for row in rows]
    feature_names = sorted(feature_rows[0])
    matrix = np.asarray([[float(feature_row[name]) for name in feature_names] for feature_row in feature_rows], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return feature_names, matrix


def build_target_matrix(rows: list[dict[str, str]], target_columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row[target]) for target in target_columns] for row in rows],
        dtype=np.float64,
    )


def deterministic_train_test_split(sample_count: int, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    if sample_count < 5:
        raise ValueError(f"Need at least 5 samples for train/test split; got {sample_count}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1; got {test_size}.")
    indices = list(range(sample_count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_count = max(1, int(round(sample_count * test_size)))
    train_count = sample_count - test_count
    if train_count < 2:
        raise ValueError(f"Need at least 2 training samples; got {train_count}.")
    test_indices = sorted(indices[:test_count])
    train_indices = sorted(indices[test_count:])
    return train_indices, test_indices


def available_regressors(seed: int) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise RuntimeError("Phase 3H-E baseline training requires scikit-learn for the requested regressors.") from exc

    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "knn": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str]) -> dict[str, Any]:
    absolute_errors = np.abs(y_pred - y_true)
    squared_errors = np.square(y_pred - y_true)
    mae_by_target = {
        target: float(absolute_errors[:, index].mean())
        for index, target in enumerate(target_columns)
    }
    rmse_by_target = {
        target: float(np.sqrt(squared_errors[:, index].mean()))
        for index, target in enumerate(target_columns)
    }
    r2_by_target = {
        target: _r2_or_none(y_true[:, index], y_pred[:, index])
        for index, target in enumerate(target_columns)
    }
    return {
        "overall_mean_mae": float(np.mean(list(mae_by_target.values()))),
        "mae_by_target": mae_by_target,
        "rmse_by_target": rmse_by_target,
        "r2_by_target": r2_by_target,
    }


def rank_models(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "rank": 0,
                "model": model_name,
                "overall_mean_mae": metrics["test"]["overall_mean_mae"],
            }
            for model_name, metrics in metrics_by_model.items()
        ),
        key=lambda row: (row["overall_mean_mae"], row["model"]),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def train_blend_dataset_baseline(
    dataset: str | Path = DEFAULT_DATASET,
    out: str | Path = DEFAULT_OUT,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    target_columns: list[str] | None = None,
    strict_audit_required: bool = False,
    audit_report: str | Path | None = None,
) -> dict[str, Any]:
    targets = target_columns or [*DEFAULT_TARGET_COLUMNS]
    validation = validate_blend_dataset(
        dataset,
        targets,
        strict_audit_required=strict_audit_required,
        audit_report=audit_report,
    )
    rows = validation["rows"]
    feature_names, feature_matrix = build_feature_matrix(rows, dataset)
    target_matrix = build_target_matrix(rows, targets)
    train_indices, test_indices = deterministic_train_test_split(len(rows), test_size, seed)
    x_train = feature_matrix[train_indices]
    x_test = feature_matrix[test_indices]
    y_train = target_matrix[train_indices]
    y_test = target_matrix[test_indices]

    metrics_by_model: dict[str, dict[str, Any]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    models = available_regressors(seed)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = np.asarray(model.predict(x_train), dtype=np.float64)
        test_pred = np.asarray(model.predict(x_test), dtype=np.float64)
        predictions_by_model[model_name] = test_pred
        metrics_by_model[model_name] = {
            "train": evaluate_predictions(y_train, train_pred, targets),
            "test": evaluate_predictions(y_test, test_pred, targets),
        }

    ranking = rank_models(metrics_by_model)
    best_model_name = str(ranking[0]["model"])
    best_model = models[best_model_name]
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "seed": seed,
        "test_size": test_size,
        "train_sample_ids": [rows[index]["sample_id"] for index in train_indices],
        "test_sample_ids": [rows[index]["sample_id"] for index in test_indices],
    }
    experiment_metadata = {
        "phase": "3H-E",
        "dataset": str(Path(dataset)),
        "audit_report": validation["audit_report"],
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "target_columns": targets,
        "sample_count": validation["sample_count"],
        "image_count": validation["image_count"],
        "seed": seed,
        "test_size": test_size,
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "synthetic_only": True,
        "synthetic_labels": validation["synthetic_labels"],
        "real_world_validated": False,
        "warning": SYNTHETIC_ONLY_WARNING,
        "variation_source": validation["metadata"].get("variation_source"),
        "shape_key_count": validation["metadata"].get("shape_key_count"),
    }
    metrics = {
        **experiment_metadata,
        "best_model": best_model_name,
        "overall_mean_mae": metrics_by_model[best_model_name]["test"]["overall_mean_mae"],
        "mae_by_target": metrics_by_model[best_model_name]["test"]["mae_by_target"],
        "models": metrics_by_model,
        "model_ranking": ranking,
    }

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "train_test_split.json", split_payload)
    write_json(output_dir / "experiment_metadata.json", experiment_metadata)
    write_feature_summary(output_dir / "feature_summary.csv", feature_names, x_train)
    write_model_ranking(output_dir / "model_ranking.csv", ranking)
    write_predictions(
        output_dir / "predictions.csv",
        rows,
        test_indices,
        y_test,
        predictions_by_model[best_model_name],
        targets,
        best_model_name,
    )
    write_metrics_summary(output_dir / "metrics_summary.md", metrics)
    write_best_model(output_dir / "best_model.joblib", best_model, metrics, feature_names)
    return {
        "metrics": metrics,
        "metrics_path": str(output_dir / "metrics.json"),
        "output_dir": str(output_dir),
        "training_command": build_training_command(
            dataset=str(dataset),
            out=str(out),
            seed=seed,
            test_size=test_size,
            target_columns=targets,
            strict_audit_required=strict_audit_required,
            audit_report=str(audit_report) if audit_report is not None else None,
        ),
    }


def write_best_model(path: Path, model: Any, metrics: dict[str, Any], feature_names: list[str]) -> None:
    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_columns": metrics["target_columns"],
            "best_model": metrics["best_model"],
            "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
            "synthetic_only": True,
            "real_world_validated": False,
        },
        path,
    )


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    test_indices: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    model_name: str,
) -> None:
    fieldnames = ["sample_id", "model"]
    for target in target_columns:
        fieldnames.extend([f"{target}_actual", f"{target}_predicted", f"{target}_absolute_error"])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_index in enumerate(test_indices):
            output_row: dict[str, Any] = {
                "sample_id": rows[sample_index]["sample_id"],
                "model": model_name,
            }
            for target_index, target in enumerate(target_columns):
                actual = float(y_true[row_index, target_index])
                predicted = float(y_pred[row_index, target_index])
                output_row[f"{target}_actual"] = f"{actual:.6f}"
                output_row[f"{target}_predicted"] = f"{predicted:.6f}"
                output_row[f"{target}_absolute_error"] = f"{abs(predicted - actual):.6f}"
            writer.writerow(output_row)


def write_feature_summary(path: Path, feature_names: list[str], x_train: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["feature", "train_mean", "train_std", "train_min", "train_max"])
        writer.writeheader()
        for index, feature in enumerate(feature_names):
            values = x_train[:, index]
            writer.writerow(
                {
                    "feature": feature,
                    "train_mean": f"{float(values.mean()):.10f}",
                    "train_std": f"{float(values.std()):.10f}",
                    "train_min": f"{float(values.min()):.10f}",
                    "train_max": f"{float(values.max()):.10f}",
                }
            )


def write_model_ranking(path: Path, ranking: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["rank", "model", "overall_mean_mae"])
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "model": row["model"],
                    "overall_mean_mae": f"{float(row['overall_mean_mae']):.6f}",
                }
            )


def write_metrics_summary(path: Path, metrics: dict[str, Any]) -> None:
    best = metrics["best_model"]
    best_metrics = metrics["models"][best]["test"]
    lines = [
        "# Phase 3H-E Blender Dataset Baseline Training",
        "",
        f"- Dataset: `{metrics['dataset']}`",
        f"- Sample count: `{metrics['sample_count']}`",
        f"- Image count: `{metrics['image_count']}`",
        f"- Training sample count: `{metrics['train_sample_count']}`",
        f"- Test sample count: `{metrics['test_sample_count']}`",
        f"- Target columns: `{', '.join(metrics['target_columns'])}`",
        f"- Best model: `{best}`",
        f"- Overall mean MAE: `{best_metrics['overall_mean_mae']:.4f}`",
        "- Synthetic-only: `true`",
        "- real_world_validated=false",
        f"- Warning: {SYNTHETIC_ONLY_WARNING}",
        "",
        "## MAE Per Measurement",
    ]
    for target, mae in best_metrics["mae_by_target"].items():
        lines.append(f"- {target}: `{mae:.4f}`")
    lines.extend(
        [
            "",
            "## Model Ranking",
        ]
    )
    for row in metrics["model_ranking"]:
        lines.append(f"- {row['rank']}. {row['model']}: `{row['overall_mean_mae']:.4f}` overall mean MAE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_training_command(
    *,
    dataset: str,
    out: str,
    seed: int,
    test_size: float,
    target_columns: list[str],
    strict_audit_required: bool,
    audit_report: str | None = None,
) -> list[str]:
    command = [
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
        *target_columns,
    ]
    if strict_audit_required:
        command.append("--strict-audit-required")
    if audit_report:
        command.extend(["--audit-report", audit_report])
    return command


def format_training_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "Phase 3H-E blend baseline training complete.",
        f"Dataset: {metrics['dataset']}",
        f"Output: {result['output_dir']}",
        f"Samples: {metrics['sample_count']}",
        f"Images: {metrics['image_count']}",
        f"Train/Test: {metrics['train_sample_count']}/{metrics['test_sample_count']}",
        f"Best model: {metrics['best_model']}",
        f"Overall mean MAE: {metrics['overall_mean_mae']:.4f}",
        "MAE by target:",
    ]
    for target, mae in metrics["mae_by_target"].items():
        lines.append(f"  {target}: {mae:.4f}")
    lines.append(f"Warning: {SYNTHETIC_ONLY_WARNING}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 3H-E baseline regressors on the audited Blender dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--target-columns", nargs="+", default=[*DEFAULT_TARGET_COLUMNS])
    parser.add_argument("--strict-audit-required", action="store_true")
    parser.add_argument("--audit-report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_blend_dataset_baseline(
        dataset=args.dataset,
        out=args.out,
        seed=args.seed,
        test_size=args.test_size,
        target_columns=args.target_columns,
        strict_audit_required=args.strict_audit_required,
        audit_report=args.audit_report,
    )
    print(format_training_summary(result))
    return 0


def _binned_nonzero_density(values: np.ndarray, bins: int) -> list[float]:
    chunks = np.array_split(values, bins)
    return [float(chunk[chunk > 0].mean() / max(values.max(), 1.0)) if np.any(chunk > 0) else 0.0 for chunk in chunks]


def _r2_or_none(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    denominator = float(np.square(y_true - y_true.mean()).sum())
    if denominator <= 1e-12:
        return None
    numerator = float(np.square(y_true - y_pred).sum())
    return 1.0 - numerator / denominator


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(values.std()) if values.size else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
