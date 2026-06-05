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

from scripts.build_phase_3h_k_curriculum_manifest import (
    DEFAULT_OUTPUT as DEFAULT_MANIFEST_DIR,
    MANIFEST_COLUMNS,
    TARGET_COLUMNS,
    ensure_not_archived_dataset,
)
from scripts.train_blend_dataset_baseline import (
    available_regressors,
    evaluate_predictions,
    extract_combined_view_features,
    extract_image_features,
    extract_projection_features,
    rank_models,
)

DEFAULT_OUTPUT = "artifacts/phase_3h_l_curriculum_benchmark"
DEFAULT_SEED = 42
SYNTHETIC_ONLY_WARNING = (
    "Synthetic-only Blender dataset results are not real-world validated and are not production tailoring accuracy."
)
STRATEGIES = {
    "clean_only": "clean_train_manifest.csv",
    "mobile_realism_only": "mobile_realism_train_manifest.csv",
    "mixed_curriculum": "mixed_curriculum_manifest.csv",
}
EVALUATION_MANIFEST = "evaluation_manifest.csv"
REQUIRED_MANIFESTS = [*STRATEGIES.values(), EVALUATION_MANIFEST]


def run_curriculum_benchmark(
    *,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    out: str | Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    manifest_root = Path(manifest_dir)
    output_dir = Path(out)
    manifests = load_required_manifests(manifest_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache: dict[str, dict[str, float]] = {}
    evaluation_rows = manifests[EVALUATION_MANIFEST]
    eval_feature_names, eval_x = build_feature_matrix_from_manifest(evaluation_rows, feature_cache)
    y_eval = build_target_matrix(evaluation_rows)

    strategy_results: dict[str, Any] = {}
    for strategy_name, manifest_name in STRATEGIES.items():
        train_rows = manifests[manifest_name]
        feature_names, x_train = build_feature_matrix_from_manifest(train_rows, feature_cache)
        if feature_names != eval_feature_names:
            raise ValueError(f"{strategy_name} feature schema differs from evaluation schema.")
        y_train = build_target_matrix(train_rows)
        strategy_results[strategy_name] = train_and_evaluate_strategy(
            strategy_name=strategy_name,
            train_rows=train_rows,
            x_train=x_train,
            y_train=y_train,
            evaluation_rows=evaluation_rows,
            x_eval=eval_x,
            y_eval=y_eval,
            seed=seed,
        )

    comparison = compare_strategies(strategy_results)
    summary = {
        "phase": "phase_3h_l_curriculum_benchmark",
        "manifest_dir": manifest_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "seed": seed,
        "evaluation": {
            "manifest": (manifest_root / EVALUATION_MANIFEST).as_posix(),
            "row_count": len(evaluation_rows),
            "dataset_source": sorted({row["dataset_source"] for row in evaluation_rows}),
        },
        "training_row_counts": {
            strategy: len(manifests[manifest_name])
            for strategy, manifest_name in STRATEGIES.items()
        },
        "feature_count": len(eval_feature_names),
        "strategies": strategy_results,
        "comparison": comparison,
        "best_strategy": comparison["best_strategy"],
        "synthetic_labels": True,
        "real_world_validated": False,
        "warning": SYNTHETIC_ONLY_WARNING,
    }

    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "summary.json", comparison)
    write_comparison_csv(output_dir / "comparison.csv", strategy_results)
    return summary


def load_required_manifests(manifest_root: Path) -> dict[str, list[dict[str, str]]]:
    if not manifest_root.exists():
        raise FileNotFoundError(f"Missing Phase 3H-K curriculum manifest folder: {manifest_root}")
    manifests: dict[str, list[dict[str, str]]] = {}
    for manifest_name in REQUIRED_MANIFESTS:
        manifest_path = manifest_root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing required Phase 3H-K manifest: {manifest_path}")
        rows = read_manifest(manifest_path)
        if not rows:
            raise ValueError(f"Manifest has no rows: {manifest_path}")
        missing_columns = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
        if missing_columns:
            raise ValueError(f"{manifest_path} missing columns: {', '.join(missing_columns)}")
        for row in rows:
            ensure_not_archived_dataset(row["dataset_path"])
            for view in ("front_image", "side_image", "back_image"):
                image_path = Path(row[view])
                if not image_path.exists():
                    raise FileNotFoundError(f"{row['curriculum_sample_id']}: missing referenced image: {image_path}")
        manifests[manifest_name] = rows
    return manifests


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def build_feature_matrix_from_manifest(
    rows: list[dict[str, str]],
    feature_cache: dict[str, dict[str, float]],
) -> tuple[list[str], np.ndarray]:
    feature_rows = [features_for_manifest_row(row, feature_cache) for row in rows]
    feature_names = sorted(feature_rows[0])
    matrix = np.asarray(
        [[float(feature_row[name]) for name in feature_names] for feature_row in feature_rows],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return feature_names, matrix


def features_for_manifest_row(row: dict[str, str], feature_cache: dict[str, dict[str, float]]) -> dict[str, float]:
    cache_key = row["curriculum_sample_id"]
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    features: dict[str, float] = {}
    for view, column in (("front", "front_image"), ("side", "side_image"), ("back", "back_image")):
        image_path = Path(row[column])
        features.update(extract_image_features(image_path, view))
        features.update(extract_projection_features(image_path, view))
    features.update(extract_combined_view_features(features))
    feature_cache[cache_key] = features
    return features


def build_target_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[target]) for target in TARGET_COLUMNS] for row in rows],
        dtype=np.float64,
    )


def train_and_evaluate_strategy(
    *,
    strategy_name: str,
    train_rows: list[dict[str, str]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluation_rows: list[dict[str, str]],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    models = available_regressors(seed)
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = np.asarray(model.predict(x_train), dtype=np.float64)
        eval_pred = np.asarray(model.predict(x_eval), dtype=np.float64)
        metrics_by_model[model_name] = {
            "train": evaluate_predictions(y_train, train_pred, list(TARGET_COLUMNS)),
            "evaluation": evaluate_predictions(y_eval, eval_pred, list(TARGET_COLUMNS)),
            "test": evaluate_predictions(y_eval, eval_pred, list(TARGET_COLUMNS)),
        }
    ranking = rank_models(metrics_by_model)
    best_model = str(ranking[0]["model"])
    best_metrics = metrics_by_model[best_model]["evaluation"]
    return {
        "strategy": strategy_name,
        "training_row_count": len(train_rows),
        "evaluation_row_count": len(evaluation_rows),
        "training_sources": sorted({row["dataset_source"] for row in train_rows}),
        "evaluation_sources": sorted({row["dataset_source"] for row in evaluation_rows}),
        "best_model": best_model,
        "overall_mean_mae": best_metrics["overall_mean_mae"],
        "mae_by_target": best_metrics["mae_by_target"],
        "models": metrics_by_model,
        "model_ranking": ranking,
    }


def compare_strategies(strategy_results: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        (
            {
                "rank": 0,
                "strategy": strategy,
                "best_model": result["best_model"],
                "overall_mean_mae": result["overall_mean_mae"],
            }
            for strategy, result in strategy_results.items()
        ),
        key=lambda row: (row["overall_mean_mae"], row["strategy"]),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    clean = strategy_results["clean_only"]
    mobile = strategy_results["mobile_realism_only"]
    mixed = strategy_results["mixed_curriculum"]
    target_deltas = {
        target: {
            "mixed_minus_clean": float(mixed["mae_by_target"][target] - clean["mae_by_target"][target]),
            "mixed_minus_mobile": float(mixed["mae_by_target"][target] - mobile["mae_by_target"][target]),
        }
        for target in TARGET_COLUMNS
    }
    significantly_worse_targets = [
        target
        for target, deltas in target_deltas.items()
        if deltas["mixed_minus_mobile"] > 0.25
    ]
    return {
        "ranked_strategies": ranked,
        "best_strategy": ranked[0]["strategy"],
        "mixed_improves_vs_clean": mixed["overall_mean_mae"] < clean["overall_mean_mae"],
        "mixed_improves_vs_mobile_only": mixed["overall_mean_mae"] < mobile["overall_mean_mae"],
        "overall_deltas": {
            "mobile_minus_clean": float(mobile["overall_mean_mae"] - clean["overall_mean_mae"]),
            "mixed_minus_clean": float(mixed["overall_mean_mae"] - clean["overall_mean_mae"]),
            "mixed_minus_mobile": float(mixed["overall_mean_mae"] - mobile["overall_mean_mae"]),
        },
        "target_deltas": target_deltas,
        "significantly_worse_targets_vs_mobile_only": significantly_worse_targets,
    }


def write_comparison_csv(path: Path, strategy_results: dict[str, Any]) -> None:
    fieldnames = ["strategy", "best_model", "overall_mean_mae", *TARGET_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, result in strategy_results.items():
            writer.writerow(
                {
                    "strategy": strategy,
                    "best_model": result["best_model"],
                    "overall_mean_mae": f"{float(result['overall_mean_mae']):.6f}",
                    **{
                        target: f"{float(result['mae_by_target'][target]):.6f}"
                        for target in TARGET_COLUMNS
                    },
                }
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3H-L curriculum benchmark on existing manifests.")
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_curriculum_benchmark(
            manifest_dir=args.manifest_dir,
            out=args.out,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Phase 3H-L curriculum benchmark failed: {exc}")
        return 1
    print("Phase 3H-L curriculum benchmark complete.")
    print(f"Output: {summary['output_dir']}")
    print(f"Evaluation rows: {summary['evaluation']['row_count']}")
    print(f"Best strategy: {summary['best_strategy']}")
    for strategy, result in summary["strategies"].items():
        print(
            f"{strategy}: {result['best_model']} "
            f"overall MAE={float(result['overall_mean_mae']):.4f}"
        )
    print(f"Mixed improves vs mobile-only: {summary['comparison']['mixed_improves_vs_mobile_only']}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    raise SystemExit(main())
