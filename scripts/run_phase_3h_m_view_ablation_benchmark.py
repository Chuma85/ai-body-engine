from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_blend_dataset_baseline import (
    available_regressors,
    deterministic_train_test_split,
    evaluate_predictions,
    extract_combined_view_features,
    extract_image_features,
    extract_projection_features,
    rank_models,
    read_labels,
)

DEFAULT_DATASET = "data/synthetic/phase_3h_j_mobile_realism_1000"
DEFAULT_OUTPUT = "artifacts/phase_3h_m_view_ablation_benchmark"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
EXPECTED_SAMPLES = 1000
VIEWS = ("front", "side", "back")
VIEW_COMBINATIONS = {
    "front": ("front",),
    "side": ("side",),
    "back": ("back",),
    "front_side": ("front", "side"),
    "front_back": ("front", "back"),
    "side_back": ("side", "back"),
    "front_side_back": ("front", "side", "back"),
}
TARGET_COLUMNS = ("height_cm", "chest_cm", "waist_cm", "hip_cm", "shoulder_cm", "inseam_cm")
ARCHIVED_DATASET_MARKERS = ("_archived_old_mannequin", "archived_old_mannequin", "old_mannequin")
SYNTHETIC_ONLY_WARNING = (
    "Synthetic-only dataset results are not real-world validated and are not production tailoring accuracy."
)


def run_view_ablation_benchmark(
    *,
    dataset: str | Path = DEFAULT_DATASET,
    out: str | Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_dir = Path(out)
    validation = validate_dataset(dataset_path, expected_samples=EXPECTED_SAMPLES)
    rows = validation["rows"]
    train_indices, test_indices = deterministic_train_test_split(len(rows), test_size, seed)
    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    y_train = build_target_matrix(train_rows)
    y_test = build_target_matrix(test_rows)
    feature_cache: dict[tuple[str, str], dict[str, float]] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for combination_name, views in VIEW_COMBINATIONS.items():
        feature_names, x_train = build_feature_matrix(train_rows, dataset_path, views, feature_cache)
        eval_feature_names, x_test = build_feature_matrix(test_rows, dataset_path, views, feature_cache)
        if feature_names != eval_feature_names:
            raise ValueError(f"{combination_name} train/evaluation feature schema mismatch.")
        results[combination_name] = train_and_evaluate_combination(
            combination_name=combination_name,
            views=views,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_count=len(train_rows),
            test_count=len(test_rows),
            seed=seed,
            feature_count=len(feature_names),
        )

    comparison = compare_view_combinations(results)
    summary = {
        "phase": "phase_3h_m_view_ablation_benchmark",
        "dataset": dataset_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "sample_count": len(rows),
        "image_count": validation["image_count"],
        "seed": seed,
        "test_size": test_size,
        "train_sample_count": len(train_rows),
        "test_sample_count": len(test_rows),
        "target_columns": list(TARGET_COLUMNS),
        "view_combinations": {name: list(views) for name, views in VIEW_COMBINATIONS.items()},
        "results": results,
        "comparison": comparison,
        "best_overall_combination": comparison["best_overall_combination"],
        "synthetic_labels": validation["synthetic_labels"],
        "real_world_validated": False,
        "warning": SYNTHETIC_ONLY_WARNING,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "summary.json", comparison)
    write_comparison_csv(output_dir / "comparison.csv", results, comparison)
    return summary


def validate_dataset(dataset_path: Path, *, expected_samples: int) -> dict[str, Any]:
    ensure_not_archived_dataset(dataset_path)
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    view_dirs = {view: images_dir / view for view in VIEWS}
    for path in (dataset_path, labels_path, metadata_path, images_dir, *view_dirs.values()):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3H-M dataset input path: {path}")
    rows = read_labels(labels_path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} labels in {labels_path}, found {len(rows)}")
    required_columns = ["sample_id", *[f"{view}_image" for view in VIEWS], *TARGET_COLUMNS]
    missing_columns = [column for column in required_columns if column not in rows[0]]
    if missing_columns:
        raise ValueError(f"{labels_path} missing columns: {', '.join(missing_columns)}")
    missing_images = missing_images_for_rows(dataset_path, rows)
    if missing_images:
        raise FileNotFoundError("Missing referenced images: " + "; ".join(missing_images[:10]))
    image_count = sum(1 for _ in images_dir.rglob("*.png"))
    expected_images = expected_samples * len(VIEWS)
    if image_count != expected_images:
        raise ValueError(f"Expected {expected_images} PNGs in {images_dir}, found {image_count}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "rows": rows,
        "image_count": image_count,
        "synthetic_labels": bool(metadata.get("synthetic_labels", True)),
    }


def missing_images_for_rows(dataset_path: Path, rows: list[dict[str, str]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for view in VIEWS:
            relative = row.get(f"{view}_image", "")
            if not relative or not (dataset_path / relative).exists():
                missing.append(f"{row.get('sample_id', '<unknown>')}:{view}:{relative}")
    return missing


def build_feature_matrix(
    rows: list[dict[str, str]],
    dataset_path: Path,
    views: tuple[str, ...],
    feature_cache: dict[tuple[str, str], dict[str, float]],
) -> tuple[list[str], np.ndarray]:
    feature_rows = [features_for_row(row, dataset_path, views, feature_cache) for row in rows]
    feature_names = sorted(feature_rows[0])
    matrix = np.asarray(
        [[float(feature_row[name]) for name in feature_names] for feature_row in feature_rows],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return feature_names, matrix


def features_for_row(
    row: dict[str, str],
    dataset_path: Path,
    views: tuple[str, ...],
    feature_cache: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for view in views:
        features.update(view_features_for_row(row, dataset_path, view, feature_cache))
    features.update(extract_subset_combined_view_features(features, views))
    return features


def view_features_for_row(
    row: dict[str, str],
    dataset_path: Path,
    view: str,
    feature_cache: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    cache_key = (row["sample_id"], view)
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    image_path = dataset_path / row[f"{view}_image"]
    features = {
        **extract_image_features(image_path, view),
        **extract_projection_features(image_path, view),
    }
    feature_cache[cache_key] = features
    return features


def extract_subset_combined_view_features(features: dict[str, float], views: tuple[str, ...]) -> dict[str, float]:
    if views == ("front", "side", "back"):
        return extract_combined_view_features(features)
    if len(views) < 2:
        return {}
    combined: dict[str, float] = {}
    for metric in (
        "raw_bbox_width_ratio",
        "raw_bbox_height_ratio",
        "raw_mask_area_ratio",
        "bbox_aspect_ratio",
        "projection_row_width_mean",
        "projection_column_height_mean",
    ):
        values = [features[f"{view}_{metric}"] for view in views]
        combined[f"{'_'.join(views)}_{metric}_mean"] = float(np.mean(values))
        for left_index, left_view in enumerate(views):
            for right_view in views[left_index + 1 :]:
                combined[f"{left_view}_{right_view}_{metric}_ratio"] = safe_ratio(
                    features[f"{left_view}_{metric}"],
                    features[f"{right_view}_{metric}"],
                )
    return combined


def build_target_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[target]) for target in TARGET_COLUMNS] for row in rows],
        dtype=np.float64,
    )


def train_and_evaluate_combination(
    *,
    combination_name: str,
    views: tuple[str, ...],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_count: int,
    test_count: int,
    seed: int,
    feature_count: int,
) -> dict[str, Any]:
    models = available_regressors(seed)
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = np.asarray(model.predict(x_train), dtype=np.float64)
        test_pred = np.asarray(model.predict(x_test), dtype=np.float64)
        metrics_by_model[model_name] = {
            "train": evaluate_predictions(y_train, train_pred, list(TARGET_COLUMNS)),
            "test": evaluate_predictions(y_test, test_pred, list(TARGET_COLUMNS)),
        }
    ranking = rank_models(metrics_by_model)
    best_model = str(ranking[0]["model"])
    best_metrics = metrics_by_model[best_model]["test"]
    return {
        "view_combination": combination_name,
        "views": list(views),
        "train_sample_count": train_count,
        "test_sample_count": test_count,
        "feature_count": feature_count,
        "best_model": best_model,
        "overall_mean_mae": best_metrics["overall_mean_mae"],
        "mae_by_target": best_metrics["mae_by_target"],
        "models": metrics_by_model,
        "model_ranking": ranking,
    }


def compare_view_combinations(results: dict[str, Any]) -> dict[str, Any]:
    baseline = results["front_side_back"]
    ranked = sorted(
        (
            {
                "rank": 0,
                "view_combination": name,
                "best_model": result["best_model"],
                "overall_mean_mae": result["overall_mean_mae"],
                "delta_vs_front_side_back": float(
                    result["overall_mean_mae"] - baseline["overall_mean_mae"]
                ),
            }
            for name, result in results.items()
        ),
        key=lambda row: (row["overall_mean_mae"], row["view_combination"]),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    best_by_target = {
        target: min(
            (
                {
                    "view_combination": name,
                    "mae": float(result["mae_by_target"][target]),
                    "delta_vs_front_side_back": float(
                        result["mae_by_target"][target] - baseline["mae_by_target"][target]
                    ),
                }
                for name, result in results.items()
            ),
            key=lambda row: (row["mae"], row["view_combination"]),
        )
        for target in TARGET_COLUMNS
    }
    front_side = results["front_side"]
    front_back = results["front_back"]
    side_back = results["side_back"]
    back_only = results["back"]
    adding_back_target_deltas = {
        target: float(baseline["mae_by_target"][target] - front_side["mae_by_target"][target])
        for target in TARGET_COLUMNS
    }
    return {
        "ranked_view_combinations": ranked,
        "best_overall_combination": ranked[0]["view_combination"],
        "best_by_target": best_by_target,
        "front_side_back_overall_mae": baseline["overall_mean_mae"],
        "front_side_overall_mae": front_side["overall_mean_mae"],
        "adding_back_to_front_side_delta": float(
            baseline["overall_mean_mae"] - front_side["overall_mean_mae"]
        ),
        "back_view_improves_front_side": baseline["overall_mean_mae"] < front_side["overall_mean_mae"],
        "adding_back_to_front_side_target_deltas": adding_back_target_deltas,
        "targets_improved_by_adding_back": [
            target for target, delta in adding_back_target_deltas.items() if delta < 0.0
        ],
        "targets_worsened_by_adding_back": [
            target for target, delta in adding_back_target_deltas.items() if delta > 0.0
        ],
        "adding_side_to_front_back_delta": float(
            baseline["overall_mean_mae"] - front_back["overall_mean_mae"]
        ),
        "adding_front_to_side_back_delta": float(
            baseline["overall_mean_mae"] - side_back["overall_mean_mae"]
        ),
        "back_only_overall_mae": back_only["overall_mean_mae"],
        "back_only_rank": next(
            row["rank"] for row in ranked if row["view_combination"] == "back"
        ),
        "target_deltas_vs_front_side_back": {
            name: {
                target: float(result["mae_by_target"][target] - baseline["mae_by_target"][target])
                for target in TARGET_COLUMNS
            }
            for name, result in results.items()
        },
    }


def write_comparison_csv(path: Path, results: dict[str, Any], comparison: dict[str, Any]) -> None:
    fieldnames = [
        "view_combination",
        "views",
        "best_model",
        "overall_mean_mae",
        "delta_vs_front_side_back",
        *TARGET_COLUMNS,
    ]
    deltas = {
        row["view_combination"]: row["delta_vs_front_side_back"]
        for row in comparison["ranked_view_combinations"]
    }
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow(
                {
                    "view_combination": name,
                    "views": ",".join(result["views"]),
                    "best_model": result["best_model"],
                    "overall_mean_mae": f"{float(result['overall_mean_mae']):.6f}",
                    "delta_vs_front_side_back": f"{float(deltas[name]):.6f}",
                    **{
                        target: f"{float(result['mae_by_target'][target]):.6f}"
                        for target in TARGET_COLUMNS
                    },
                }
            )


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-9:
        return 0.0
    return float(numerator) / float(denominator)


def ensure_not_archived_dataset(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if any(marker in normalized for marker in ARCHIVED_DATASET_MARKERS):
        raise ValueError(f"Phase 3H-M must not use archived old mannequin datasets: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3H-M view ablation benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_view_ablation_benchmark(
            dataset=args.dataset,
            out=args.out,
            seed=args.seed,
            test_size=args.test_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Phase 3H-M view ablation benchmark failed: {exc}")
        return 1
    print("Phase 3H-M view ablation benchmark complete.")
    print(f"Dataset: {summary['dataset']}")
    print(f"Output: {summary['output_dir']}")
    print(f"Best overall view combination: {summary['best_overall_combination']}")
    for name, result in summary["results"].items():
        print(
            f"{name}: {result['best_model']} "
            f"overall MAE={float(result['overall_mean_mae']):.4f}"
        )
    print(
        "Adding back to front+side delta: "
        f"{summary['comparison']['adding_back_to_front_side_delta']:.4f}"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    raise SystemExit(main())
