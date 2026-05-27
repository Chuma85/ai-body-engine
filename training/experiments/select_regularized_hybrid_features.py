from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.analyze_feature_drift import feature_drift_group
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.train_baseline_measurements import TARGET_COLUMNS, _mean, _require_enough_samples
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix, train_ridge_regressor

SUMMARY_FILENAME = "summary.json"
REPORT_FILENAME = "report.md"
RESULTS_FILENAME = "results.csv"
PER_TARGET_RESULTS_FILENAME = "per_target_results.csv"
FEATURE_SELECTION_FILENAME = "feature_selection.json"
MODEL_IMPORTANCE_FILENAME = "model_importance.csv"

DEFAULT_FEATURE_CONFIGS = [
    "normalized_shape",
    "raw_scale_camera",
    "combined_hybrid",
    "combined_hybrid_without_offsets",
    "combined_hybrid_without_area_ratios",
    "selected_low_drift_features",
]
DEFAULT_MODEL_TYPES = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]
SKLEARN_MODEL_TYPES = {"elasticnet", "lasso", "random_forest", "gradient_boosting", "hist_gradient_boosting"}


def run_hybrid_feature_selection_benchmark(
    datasets: list[str | Path],
    output_dir: str | Path,
    feature_configs: list[str] | None = None,
    model_types: list[str] | None = None,
    drift_csv: str | Path | None = None,
    drift_threshold: float = 1.0,
    ridge_alpha: float = 30.0,
    elasticnet_alpha: float = 0.05,
    elasticnet_l1_ratio: float = 0.35,
    random_state: int = 42,
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one dataset is required.")

    selected_feature_configs = feature_configs or DEFAULT_FEATURE_CONFIGS
    selected_model_types = model_types or DEFAULT_MODEL_TYPES
    for config_name in selected_feature_configs:
        validate_feature_config(config_name)
    for model_type in selected_model_types:
        validate_model_type(model_type)

    drift_scores = load_drift_scores(drift_csv) if drift_csv else {}
    all_feature_names = get_feature_names()
    selected_features = {
        config_name: select_feature_names(
            all_feature_names,
            config_name,
            drift_scores=drift_scores,
            drift_threshold=drift_threshold,
        )
        for config_name in selected_feature_configs
    }
    for config_name, names in selected_features.items():
        if not names:
            raise ValueError(f"Feature config '{config_name}' selected zero features.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []

    for dataset_root in datasets:
        dataset_path = Path(dataset_root)
        split_samples = load_split_samples(dataset_path)
        targets_by_split = {split: _target_matrix(samples, TARGET_COLUMNS) for split, samples in split_samples.items()}
        dataset_name = dataset_path.name

        for config_name in selected_feature_configs:
            feature_names = selected_features[config_name]
            features_by_split = {
                split: extract_sample_feature_matrix(samples, feature_names)
                for split, samples in split_samples.items()
            }
            for model_type in selected_model_types:
                if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
                    skipped_runs.append(
                        {
                            "dataset": dataset_name,
                            "feature_config": config_name,
                            "model_type": model_type,
                            "reason": "scikit-learn is not available",
                        }
                    )
                    continue
                try:
                    trained = train_selected_model(
                        model_type,
                        features_by_split["train"],
                        targets_by_split["train"],
                        feature_names,
                        ridge_alpha=ridge_alpha,
                        elasticnet_alpha=elasticnet_alpha,
                        elasticnet_l1_ratio=elasticnet_l1_ratio,
                        random_state=random_state,
                    )
                    predictions_by_split = {
                        split: predict_selected_model(trained, features)
                        for split, features in features_by_split.items()
                    }
                except Exception as error:  # pragma: no cover - exercised by real optional sklearn failures
                    skipped_runs.append(
                        {
                            "dataset": dataset_name,
                            "feature_config": config_name,
                            "model_type": model_type,
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                metrics = build_metrics(targets_by_split, predictions_by_split)
                run_row = {
                    "dataset": dataset_name,
                    "dataset_path": str(dataset_path),
                    "feature_config": config_name,
                    "model_type": model_type,
                    "feature_count": len(feature_names),
                    "train_mae": metrics["train"]["overall_mae"],
                    "val_mae": metrics["val"]["overall_mae"],
                    "test_mae": metrics["test"]["overall_mae"],
                    "worst_target": worst_target(metrics["test"]["mae_by_target"]),
                    "worst_target_mae": max(metrics["test"]["mae_by_target"].values()),
                }
                run_rows.append(run_row)
                per_target_rows.extend(
                    build_per_target_rows(dataset_name, config_name, model_type, metrics["test"]["mae_by_target"])
                )
                importance_rows.extend(
                    build_importance_rows(dataset_name, config_name, model_type, feature_names, trained["importance"])
                )

    summary = build_summary(
        run_rows,
        per_target_rows,
        selected_features,
        skipped_runs,
        drift_csv=drift_csv,
        drift_threshold=drift_threshold,
    )
    paths = {
        "summary_path": output_path / SUMMARY_FILENAME,
        "report_path": output_path / REPORT_FILENAME,
        "results_path": output_path / RESULTS_FILENAME,
        "per_target_results_path": output_path / PER_TARGET_RESULTS_FILENAME,
        "feature_selection_path": output_path / FEATURE_SELECTION_FILENAME,
        "model_importance_path": output_path / MODEL_IMPORTANCE_FILENAME,
    }
    write_json(paths["summary_path"], summary)
    paths["report_path"].write_text(format_report(summary), encoding="utf-8")
    write_csv(paths["results_path"], run_rows)
    write_csv(paths["per_target_results_path"], per_target_rows)
    write_json(paths["feature_selection_path"], build_feature_selection_payload(selected_features, drift_scores, drift_threshold))
    write_csv(paths["model_importance_path"], importance_rows)

    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def load_split_samples(dataset_path: Path) -> dict[str, list[dict[str, Any]]]:
    datasets = {
        "train": SyntheticBodyDataset(dataset_path, split="train"),
        "val": SyntheticBodyDataset(dataset_path, split="val"),
        "test": SyntheticBodyDataset(dataset_path, split="test"),
    }
    _require_enough_samples(datasets["train"], datasets["val"], datasets["test"])
    return {split: list(dataset) for split, dataset in datasets.items()}


def load_drift_scores(drift_csv: str | Path | None) -> dict[str, float]:
    if drift_csv is None:
        return {}
    path = Path(drift_csv)
    if not path.exists():
        raise FileNotFoundError(f"Feature drift CSV does not exist: {path}")
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            feature = row.get("feature", "")
            if not feature:
                continue
            score = float(row.get("mean_abs_drift", 0.0))
            scores[feature] = max(scores.get(feature, 0.0), score)
    return scores


def select_feature_names(
    feature_names: list[str],
    feature_config: str,
    drift_scores: dict[str, float] | None = None,
    drift_threshold: float = 1.0,
) -> list[str]:
    validate_feature_config(feature_config)
    scores = drift_scores or {}
    if feature_config == "normalized_shape":
        selected = [name for name in feature_names if feature_drift_group(name) == "normalized_shape"]
    elif feature_config == "raw_scale_camera":
        selected = [name for name in feature_names if feature_drift_group(name) == "raw_scale_camera"]
    elif feature_config == "combined_hybrid":
        selected = list(feature_names)
    elif feature_config == "combined_hybrid_without_offsets":
        selected = [name for name in feature_names if not is_offset_feature(name)]
    elif feature_config == "combined_hybrid_without_area_ratios":
        selected = [name for name in feature_names if not is_area_ratio_feature(name)]
    elif feature_config == "selected_low_drift_features":
        selected = [
            name
            for name in feature_names
            if scores.get(name, 0.0) <= drift_threshold and not is_offset_feature(name)
        ]
    else:
        raise AssertionError(f"Unhandled feature config: {feature_config}")
    return sorted(selected, key={name: index for index, name in enumerate(feature_names)}.__getitem__)


def is_offset_feature(feature_name: str) -> bool:
    return "_crop_offset_" in feature_name or feature_name.endswith("_crop_offset_x") or feature_name.endswith("_crop_offset_y")


def is_area_ratio_feature(feature_name: str) -> bool:
    if feature_name in {"front_to_side_area_ratio"}:
        return True
    return any(
        token in feature_name
        for token in (
            "raw_mask_area",
            "foreground_area_ratio",
            "area_to_height",
            "area_to_bbox",
            "torso_area_ratio",
            "upper_body_area_ratio",
            "lower_body_area_ratio",
            "upper_to_lower_area_ratio",
            "area_product",
            "area_to_height_proxy",
        )
    )


def train_selected_model(
    model_type: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    feature_names: list[str],
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    random_state: int,
) -> dict[str, Any]:
    validate_model_type(model_type)
    if model_type == "ridge":
        model = train_ridge_regressor(train_features, train_targets, feature_names, TARGET_COLUMNS, ridge_alpha)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        return {
            "model_type": model_type,
            "model": model,
            "importance": linear_group_importance(feature_names, coefficients),
        }

    sklearn = require_sklearn()
    if model_type in {"elasticnet", "lasso"}:
        alpha = elasticnet_alpha
        l1_ratio = 1.0 if model_type == "lasso" else elasticnet_l1_ratio
        estimator = sklearn["MultiOutputRegressor"](
            sklearn["ElasticNet"](alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state)
        )
    elif model_type == "random_forest":
        estimator = sklearn["RandomForestRegressor"](
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )
    elif model_type == "gradient_boosting":
        estimator = sklearn["MultiOutputRegressor"](
            sklearn["GradientBoostingRegressor"](random_state=random_state, max_depth=3, n_estimators=80)
        )
    elif model_type == "hist_gradient_boosting":
        estimator = sklearn["MultiOutputRegressor"](
            sklearn["HistGradientBoostingRegressor"](random_state=random_state, max_iter=80, max_leaf_nodes=15)
        )
    else:
        raise AssertionError(f"Unhandled model type: {model_type}")

    feature_means = train_features.mean(axis=0)
    feature_stds = np.where(train_features.std(axis=0) < 1e-8, 1.0, train_features.std(axis=0))
    use_standardized = model_type in {"elasticnet", "lasso"}
    fit_features = (train_features - feature_means) / feature_stds if use_standardized else train_features
    estimator.fit(fit_features, train_targets)
    return {
        "model_type": model_type,
        "estimator": estimator,
        "use_standardized": use_standardized,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "importance": sklearn_group_importance(model_type, feature_names, estimator),
    }


def predict_selected_model(trained: dict[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    model_type = trained["model_type"]
    if model_type == "ridge":
        model = trained["model"]
        feature_means = np.asarray(model["feature_means"], dtype=np.float64)
        feature_stds = np.asarray(model["feature_stds"], dtype=np.float64)
        intercepts = np.asarray(model["intercepts"], dtype=np.float64)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        standardized = (feature_matrix - feature_means) / feature_stds
        return standardized @ coefficients + intercepts
    features = feature_matrix
    if trained.get("use_standardized"):
        features = (feature_matrix - trained["feature_means"]) / trained["feature_stds"]
    return np.asarray(trained["estimator"].predict(features), dtype=np.float64)


def build_metrics(targets_by_split: dict[str, np.ndarray], predictions_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split, targets in targets_by_split.items():
        predictions = predictions_by_split[split]
        errors = np.abs(predictions - targets)
        mae_by_target = {
            target: float(errors[:, index].mean())
            for index, target in enumerate(TARGET_COLUMNS)
        }
        metrics[split] = {
            "overall_mae": _mean(list(mae_by_target.values())),
            "mae_by_target": mae_by_target,
        }
    return metrics


def build_per_target_rows(
    dataset_name: str,
    feature_config: str,
    model_type: str,
    mae_by_target: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset_name,
            "feature_config": feature_config,
            "model_type": model_type,
            "target": target,
            "test_mae": mae,
        }
        for target, mae in mae_by_target.items()
    ]


def build_importance_rows(
    dataset_name: str,
    feature_config: str,
    model_type: str,
    feature_names: list[str],
    importance: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    group_importance = importance.get("group_importance", {})
    for group, value in sorted(group_importance.items()):
        rows.append(
            {
                "dataset": dataset_name,
                "feature_config": feature_config,
                "model_type": model_type,
                "importance_type": importance.get("importance_type", "unavailable"),
                "feature_group": group,
                "feature_count": sum(1 for name in feature_names if feature_drift_group(name) == group),
                "importance": value,
            }
        )
    return rows


def linear_group_importance(feature_names: list[str], coefficients: np.ndarray) -> dict[str, Any]:
    abs_coefficients = np.abs(coefficients)
    feature_importance = abs_coefficients.mean(axis=1)
    return group_importance_payload(feature_names, feature_importance, "mean_abs_coefficient")


def sklearn_group_importance(model_type: str, feature_names: list[str], estimator: Any) -> dict[str, Any]:
    values: np.ndarray | None = None
    if model_type in {"elasticnet", "lasso"}:
        coefficients = np.asarray([model.coef_ for model in estimator.estimators_], dtype=np.float64).T
        values = np.abs(coefficients).mean(axis=1)
        importance_type = "mean_abs_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
        importance_type = "feature_importance"
    elif hasattr(estimator, "estimators_"):
        importances = [
            getattr(model, "feature_importances_", None)
            for model in estimator.estimators_
        ]
        importances = [value for value in importances if value is not None]
        if importances:
            values = np.asarray(importances, dtype=np.float64).mean(axis=0)
            importance_type = "feature_importance"
        else:
            importance_type = "unavailable"
    else:
        importance_type = "unavailable"
    if values is None:
        return {"importance_type": importance_type, "group_importance": {}}
    return group_importance_payload(feature_names, values, importance_type)


def group_importance_payload(feature_names: list[str], feature_importance: np.ndarray, importance_type: str) -> dict[str, Any]:
    groups = sorted({feature_drift_group(name) for name in feature_names})
    total = float(np.abs(feature_importance).sum())
    if total <= 0:
        total = 1.0
    group_values = {}
    for group in groups:
        indices = [index for index, name in enumerate(feature_names) if feature_drift_group(name) == group]
        group_values[group] = float(np.abs(feature_importance[indices]).sum() / total)
    return {
        "importance_type": importance_type,
        "group_importance": group_values,
    }


def build_summary(
    run_rows: list[dict[str, Any]],
    per_target_rows: list[dict[str, Any]],
    selected_features: dict[str, list[str]],
    skipped_runs: list[dict[str, Any]],
    drift_csv: str | Path | None,
    drift_threshold: float,
) -> dict[str, Any]:
    if not run_rows:
        raise ValueError("No hybrid feature selection runs completed.")
    best_by_dataset = {}
    for dataset in sorted({row["dataset"] for row in run_rows}):
        rows = [row for row in run_rows if row["dataset"] == dataset]
        best_by_dataset[dataset] = min(rows, key=lambda row: row["test_mae"])
    best_per_target = {}
    for dataset in sorted({row["dataset"] for row in per_target_rows}):
        best_per_target[dataset] = {}
        for target in TARGET_COLUMNS:
            rows = [row for row in per_target_rows if row["dataset"] == dataset and row["target"] == target]
            if rows:
                best_per_target[dataset][target] = min(rows, key=lambda row: row["test_mae"])
    return {
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "target_columns": TARGET_COLUMNS,
        "drift_csv": str(drift_csv) if drift_csv else None,
        "drift_threshold": drift_threshold,
        "feature_configs": {
            name: {
                "feature_count": len(features),
                "features": features,
            }
            for name, features in selected_features.items()
        },
        "run_count": len(run_rows),
        "skipped_runs": skipped_runs,
        "results": run_rows,
        "best_by_dataset": best_by_dataset,
        "best_per_target": best_per_target,
    }


def build_feature_selection_payload(
    selected_features: dict[str, list[str]],
    drift_scores: dict[str, float],
    drift_threshold: float,
) -> dict[str, Any]:
    return {
        "drift_threshold": drift_threshold,
        "feature_configs": {
            name: {
                "feature_count": len(features),
                "features": features,
                "max_drift_score": max((drift_scores.get(feature, 0.0) for feature in features), default=0.0),
            }
            for name, features in selected_features.items()
        },
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Feature Selection Benchmark",
        "",
        f"Feature extractor: `{summary['feature_extractor_version']}`",
        f"Runs completed: {summary['run_count']}",
        f"Drift threshold: {summary['drift_threshold']}",
        "",
        "## Best By Dataset",
        "",
        markdown_table(
            ["Dataset", "Feature Config", "Model", "Features", "Train MAE", "Val MAE", "Test MAE"],
            [
                [
                    dataset,
                    row["feature_config"],
                    row["model_type"],
                    str(row["feature_count"]),
                    format_float(row["train_mae"]),
                    format_float(row["val_mae"]),
                    format_float(row["test_mae"]),
                ]
                for dataset, row in summary["best_by_dataset"].items()
            ],
        ),
        "",
        "## Feature Configs",
        "",
        markdown_table(
            ["Feature Config", "Feature Count"],
            [
                [name, str(config["feature_count"])]
                for name, config in summary["feature_configs"].items()
            ],
        ),
        "",
        "## Notes",
        "",
        "- `selected_low_drift_features` excludes crop offsets and features above the configured drift threshold.",
        "- Tree-based models are included only when scikit-learn is available.",
        "- Keep Phase 3L clean ridge as current best unless a same-body run beats 6.5780 test MAE.",
        "",
    ]
    if summary["skipped_runs"]:
        lines.extend(["## Skipped Runs", "", markdown_table(["Dataset", "Feature Config", "Model", "Reason"], [
            [row["dataset"], row["feature_config"], row["model_type"], row["reason"]]
            for row in summary["skipped_runs"]
        ]), ""])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def worst_target(mae_by_target: dict[str, float]) -> str:
    return max(mae_by_target, key=mae_by_target.get)


def sklearn_available() -> bool:
    try:
        require_sklearn()
    except ImportError:
        return False
    return True


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError as error:
        raise ImportError("scikit-learn is required for this model type.") from error
    return {
        "ElasticNet": ElasticNet,
        "MultiOutputRegressor": MultiOutputRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    }


def validate_feature_config(feature_config: str) -> None:
    if feature_config not in DEFAULT_FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature config '{feature_config}'. Expected one of: {', '.join(DEFAULT_FEATURE_CONFIGS)}.")


def validate_model_type(model_type: str) -> None:
    valid = ["ridge", "elasticnet", "lasso", "random_forest", "gradient_boosting", "hist_gradient_boosting"]
    if model_type not in valid:
        raise ValueError(f"Unknown model type '{model_type}'. Expected one of: {', '.join(valid)}.")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _header in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and regularize hybrid silhouette features.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset roots to benchmark.")
    parser.add_argument("--output", required=True, help="Output analysis directory.")
    parser.add_argument("--feature-configs", nargs="+", default=DEFAULT_FEATURE_CONFIGS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_TYPES)
    parser.add_argument("--drift-csv", help="Feature drift CSV for selected_low_drift_features.")
    parser.add_argument("--drift-threshold", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=30.0)
    parser.add_argument("--elasticnet-alpha", type=float, default=0.05)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = run_hybrid_feature_selection_benchmark(
        args.datasets,
        args.output,
        feature_configs=args.feature_configs,
        model_types=args.models,
        drift_csv=args.drift_csv,
        drift_threshold=args.drift_threshold,
        ridge_alpha=args.ridge_alpha,
        elasticnet_alpha=args.elasticnet_alpha,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
        random_state=args.seed,
    )
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    for dataset, row in result["summary"]["best_by_dataset"].items():
        print(f"{dataset}: {row['feature_config']} + {row['model_type']} test MAE {row['test_mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
