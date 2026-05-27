from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.audit_label_geometry_alignment import (
    extract_geometry_proxy_matrix,
    find_geometry_label_ambiguity_pairs,
    standardize_matrix,
)
from training.experiments.optimize_silhouette_targets import promotion_gate
from training.experiments.select_regularized_hybrid_features import (
    SKLEARN_MODEL_TYPES,
    predict_selected_model,
    select_feature_names,
    sklearn_available,
    train_selected_model,
    validate_model_type,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, get_feature_names
from training.features.measurement_band_features import MEASUREMENT_BAND_TARGETS
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import _target_matrix, extract_sample_feature_matrix

AMBIGUITY_SCORES_CSV = "ambiguity_scores.csv"
FILTERED_MANIFEST_CLEAN_TRAIN_CSV = "filtered_manifest_clean_train.csv"
FILTERED_MANIFEST_CLEAN_ALL_CSV = "filtered_manifest_clean_all.csv"
AMBIGUOUS_PAIRS_CSV = "ambiguous_pairs.csv"
BENCHMARK_RESULTS_JSON = "clean_subset_benchmark_results.json"
BENCHMARK_RESULTS_CSV = "clean_subset_benchmark_results.csv"
PER_TARGET_RESULTS_CSV = "per_target_clean_subset_results.csv"
SUMMARY_MD = "ambiguity_summary.md"

TARGETS = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm"]
DEFAULT_MODEL_TYPES = ["gradient_boosting", "ridge", "elasticnet", "random_forest"]
DEFAULT_AMBIGUITY_PERCENTILE = 85.0
RAW_SCALE_FEATURE_CONFIG = "raw_scale_camera"


def filter_label_geometry_ambiguity(
    dataset: str | Path,
    output_dir: str | Path,
    phase3y_artifacts: str | Path | None = None,
    ambiguity_percentile: float = DEFAULT_AMBIGUITY_PERCENTILE,
    model_types: list[str] | None = None,
    ambiguity_pairs_per_target: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_model_types = model_types or DEFAULT_MODEL_TYPES
    for model_type in selected_model_types:
        validate_model_type(model_type)

    warnings = validate_optional_phase3y_artifacts(phase3y_artifacts)
    all_samples = list(SyntheticBodyDataset(dataset_path, split="all"))
    if not all_samples:
        raise ValueError(f"No samples available for ambiguity filtering: {dataset_path}")

    sample_ids, proxy_names, proxy_matrix, target_matrix = extract_geometry_proxy_matrix(all_samples)
    split_by_sample_id = {sample["sample_id"]: sample["dataset_split"] for sample in all_samples}
    ambiguity = calculate_ambiguity_scores(
        sample_ids,
        proxy_names,
        proxy_matrix,
        target_matrix,
        split_by_sample_id,
        ambiguity_percentile=ambiguity_percentile,
    )
    ambiguous_pairs = find_geometry_label_ambiguity_pairs(
        sample_ids,
        proxy_matrix,
        target_matrix,
        proxy_names,
        ambiguity_pairs_per_target=ambiguity_pairs_per_target,
    )
    best_proxy_ranges = summarize_clean_ambiguous_ranges(proxy_names, proxy_matrix, target_matrix, ambiguity["rows"])

    manifest_rows, manifest_fieldnames = read_manifest(dataset_path / "manifest.csv")
    clean_train_manifest = filter_manifest_rows(manifest_rows, ambiguity["ambiguous_sample_ids"], mode="clean_train_only")
    clean_all_manifest = filter_manifest_rows(manifest_rows, ambiguity["ambiguous_sample_ids"], mode="clean_train_clean_test")

    raw_feature_names = select_feature_names(get_feature_names(), RAW_SCALE_FEATURE_CONFIG)
    split_samples = {
        split: [sample for sample in all_samples if sample["dataset_split"] == split]
        for split in ("train", "val", "test")
    }
    benchmark = benchmark_clean_subsets(
        split_samples,
        ambiguity["ambiguous_sample_ids"],
        raw_feature_names,
        selected_model_types,
        random_state=random_state,
    )

    summary = {
        "dataset": str(dataset_path),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "feature_config": RAW_SCALE_FEATURE_CONFIG,
        "targets": TARGETS,
        "ambiguity_percentile": ambiguity_percentile,
        "sample_count": len(all_samples),
        "ambiguous_sample_count": len(ambiguity["ambiguous_sample_ids"]),
        "ambiguous_sample_percent": percentage(len(ambiguity["ambiguous_sample_ids"]), len(all_samples)),
        "score_distributions": ambiguity["score_distributions"],
        "clean_vs_ambiguous_ranges": best_proxy_ranges,
        "best_run": benchmark["best_run"],
        "below_5cm_targets": benchmark["below_5cm_targets"],
        "group_below_5cm": bool(float(benchmark["best_run"]["test_group_mae"]) <= 5.0),
        "warnings": [*warnings, *benchmark["warnings"]],
        "interpretation": interpretation(benchmark["best_run"], len(ambiguity["ambiguous_sample_ids"]), len(all_samples)),
        "benchmark_results": benchmark["run_rows"],
    }

    paths = {
        "ambiguity_scores_csv": output_path / AMBIGUITY_SCORES_CSV,
        "filtered_manifest_clean_train_csv": output_path / FILTERED_MANIFEST_CLEAN_TRAIN_CSV,
        "filtered_manifest_clean_all_csv": output_path / FILTERED_MANIFEST_CLEAN_ALL_CSV,
        "ambiguous_pairs_csv": output_path / AMBIGUOUS_PAIRS_CSV,
        "benchmark_results_json": output_path / BENCHMARK_RESULTS_JSON,
        "benchmark_results_csv": output_path / BENCHMARK_RESULTS_CSV,
        "per_target_results_csv": output_path / PER_TARGET_RESULTS_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_csv(paths["ambiguity_scores_csv"], ambiguity["rows"], ambiguity_score_fieldnames())
    write_csv(paths["filtered_manifest_clean_train_csv"], clean_train_manifest, manifest_fieldnames)
    write_csv(paths["filtered_manifest_clean_all_csv"], clean_all_manifest, manifest_fieldnames)
    write_csv(paths["ambiguous_pairs_csv"], ambiguous_pairs, ambiguous_pair_fieldnames())
    write_json(paths["benchmark_results_json"], summary)
    write_csv(paths["benchmark_results_csv"], benchmark["run_rows"], benchmark_fieldnames())
    write_csv(paths["per_target_results_csv"], benchmark["per_target_rows"], per_target_fieldnames())
    paths["summary_md"].write_text(format_summary(summary), encoding="utf-8")

    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def calculate_ambiguity_scores(
    sample_ids: list[str],
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    target_matrix: np.ndarray,
    split_by_sample_id: dict[str, str],
    ambiguity_percentile: float = DEFAULT_AMBIGUITY_PERCENTILE,
    nearest_neighbors: int = 8,
) -> dict[str, Any]:
    if not sample_ids:
        raise ValueError("Cannot calculate ambiguity scores with no samples.")
    target_scores: dict[str, np.ndarray] = {}
    target_neighbor_ids: dict[str, list[str]] = {}
    target_label_diffs: dict[str, np.ndarray] = {}
    target_thresholds: dict[str, float] = {}
    rows_by_sample: dict[str, dict[str, Any]] = {
        sample_id: {"sample_id": sample_id, "dataset_split": split_by_sample_id.get(sample_id, "")}
        for sample_id in sample_ids
    }

    for target_index, target in enumerate(TARGETS):
        feature_indices = target_proxy_indices(target, proxy_names)
        if not feature_indices:
            scores = np.zeros(len(sample_ids), dtype=np.float64)
            neighbor_ids = [""] * len(sample_ids)
            label_diffs = np.zeros(len(sample_ids), dtype=np.float64)
        else:
            scores, neighbor_ids, label_diffs = target_ambiguity_scores(
                sample_ids,
                proxy_matrix[:, feature_indices],
                target_matrix[:, target_index],
                nearest_neighbors=nearest_neighbors,
            )
        threshold = float(np.percentile(scores, ambiguity_percentile)) if len(scores) else 0.0
        target_scores[target] = scores
        target_neighbor_ids[target] = neighbor_ids
        target_label_diffs[target] = label_diffs
        target_thresholds[target] = threshold

    group_scores = np.vstack([target_scores[target] for target in TARGETS]).max(axis=0)
    group_threshold = float(np.percentile(group_scores, ambiguity_percentile))
    ambiguous_sample_ids: set[str] = set()
    for sample_index, sample_id in enumerate(sample_ids):
        row = rows_by_sample[sample_id]
        target_flags = []
        for target in TARGETS:
            score = float(target_scores[target][sample_index])
            is_ambiguous = bool(score >= target_thresholds[target] and score > 0.0)
            target_flags.append(is_ambiguous)
            row[f"{target}_ambiguity_score"] = score
            row[f"{target}_nearest_ambiguous_neighbor"] = target_neighbor_ids[target][sample_index]
            row[f"{target}_nearest_label_diff"] = float(target_label_diffs[target][sample_index])
            row[f"{target}_is_ambiguous"] = is_ambiguous
        group_ambiguous = bool(group_scores[sample_index] >= group_threshold and group_scores[sample_index] > 0.0)
        row["group_ambiguity_score"] = float(group_scores[sample_index])
        row["group_is_ambiguous"] = group_ambiguous or any(target_flags)
        if row["group_is_ambiguous"]:
            ambiguous_sample_ids.add(sample_id)

    rows = [rows_by_sample[sample_id] for sample_id in sample_ids]
    return {
        "rows": rows,
        "ambiguous_sample_ids": ambiguous_sample_ids,
        "target_thresholds": target_thresholds,
        "group_threshold": group_threshold,
        "score_distributions": build_score_distributions(rows),
    }


def target_ambiguity_scores(
    sample_ids: list[str],
    target_proxy_matrix: np.ndarray,
    label_values: np.ndarray,
    nearest_neighbors: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    geometry = standardize_matrix(target_proxy_matrix)
    labels = standardize_matrix(label_values.reshape(-1, 1)).reshape(-1)
    distances = pairwise_euclidean(geometry)
    np.fill_diagonal(distances, np.inf)
    neighbor_count = max(1, min(nearest_neighbors, len(sample_ids) - 1))
    nearest = np.argsort(distances, axis=1)[:, :neighbor_count]
    scores = np.zeros(len(sample_ids), dtype=np.float64)
    neighbor_ids: list[str] = [""] * len(sample_ids)
    label_diffs = np.zeros(len(sample_ids), dtype=np.float64)
    for left_index, neighbors in enumerate(nearest):
        best_score = -1.0
        best_neighbor = ""
        best_label_diff = 0.0
        for right_index in neighbors:
            right = int(right_index)
            label_diff = abs(float(label_values[left_index] - label_values[right]))
            standardized_label_diff = abs(float(labels[left_index] - labels[right]))
            geometry_distance = float(distances[left_index, right])
            score = standardized_label_diff / max(geometry_distance, 1e-9)
            if score > best_score:
                best_score = score
                best_neighbor = sample_ids[right]
                best_label_diff = label_diff
        scores[left_index] = max(best_score, 0.0)
        neighbor_ids[left_index] = best_neighbor
        label_diffs[left_index] = best_label_diff
    return scores, neighbor_ids, label_diffs


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    differences = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def target_proxy_indices(target: str, proxy_names: list[str]) -> list[int]:
    target_prefix = target.removesuffix("_cm")
    return [index for index, name in enumerate(proxy_names) if name.startswith(f"{target_prefix}_band_")]


def benchmark_clean_subsets(
    split_samples: dict[str, list[dict[str, Any]]],
    ambiguous_sample_ids: set[str],
    feature_names: list[str],
    model_types: list[str],
    random_state: int,
) -> dict[str, Any]:
    variants = build_split_variants(split_samples, ambiguous_sample_ids)
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for variant_name, variant_samples in variants.items():
        if not variant_samples["train"] or not variant_samples["test"]:
            warnings.append(f"Skipped {variant_name}: train or test split is empty after filtering.")
            continue
        features_by_split = {
            split: extract_sample_feature_matrix(samples, feature_names)
            for split, samples in variant_samples.items()
        }
        targets_by_split = {
            split: _target_matrix(samples, TARGETS)
            for split, samples in variant_samples.items()
        }
        for model_type in model_types:
            if model_type in SKLEARN_MODEL_TYPES and not sklearn_available():
                warnings.append(f"Skipped {variant_name} {model_type}: scikit-learn is not available.")
                continue
            try:
                predictions_by_split = train_and_predict_target_specific(
                    model_type,
                    features_by_split,
                    targets_by_split,
                    feature_names,
                    random_state=random_state,
                )
            except Exception as error:  # pragma: no cover - for optional sklearn runtime errors
                warnings.append(f"Skipped {variant_name} {model_type}: {type(error).__name__}: {error}")
                continue
            metrics = evaluate_predictions(targets_by_split, predictions_by_split)
            run_name = f"{variant_name}__raw_scale_camera__target_specific__{model_type}"
            run_rows.append(
                {
                    "run_name": run_name,
                    "variant": variant_name,
                    "feature_config": RAW_SCALE_FEATURE_CONFIG,
                    "model_type": model_type,
                    "mode": "target_specific",
                    "feature_count": len(feature_names),
                    "train_count": len(variant_samples["train"]),
                    "val_count": len(variant_samples["val"]),
                    "test_count": len(variant_samples["test"]),
                    "filtered_train_count": len(split_samples["train"]) - len(variant_samples["train"]),
                    "filtered_val_count": len(split_samples["val"]) - len(variant_samples["val"]),
                    "filtered_test_count": len(split_samples["test"]) - len(variant_samples["test"]),
                    "train_group_mae": metrics["train"]["overall_mae"],
                    "val_group_mae": metrics["val"]["overall_mae"] if variant_samples["val"] else "",
                    "test_group_mae": metrics["test"]["overall_mae"],
                    "promotion_gate": promotion_gate(metrics["test"]["overall_mae"])["gate"],
                    "worst_target": max(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
                    "best_target": min(metrics["test"]["mae_by_target"], key=metrics["test"]["mae_by_target"].get),
                }
            )
            for target in TARGETS:
                per_target_rows.append(
                    {
                        "run_name": run_name,
                        "variant": variant_name,
                        "feature_config": RAW_SCALE_FEATURE_CONFIG,
                        "model_type": model_type,
                        "target": target,
                        "test_mae": metrics["test"]["mae_by_target"][target],
                        "promotion_gate": promotion_gate(metrics["test"]["mae_by_target"][target])["gate"],
                    }
                )
    if not run_rows:
        raise ValueError("No ambiguity filtering benchmark runs completed.")
    best_run = min(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"]))
    below_5cm_targets = [
        row
        for row in per_target_rows
        if row["run_name"] == best_run["run_name"] and float(row["test_mae"]) <= 5.0
    ]
    return {
        "run_rows": sorted(run_rows, key=lambda row: (float(row["test_group_mae"]), row["run_name"])),
        "per_target_rows": per_target_rows,
        "best_run": best_run,
        "below_5cm_targets": below_5cm_targets,
        "warnings": warnings,
    }


def train_and_predict_target_specific(
    model_type: str,
    features_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    feature_names: list[str],
    random_state: int,
) -> dict[str, np.ndarray]:
    predictions_by_split = {
        split: np.zeros((targets.shape[0], len(TARGETS)), dtype=np.float64)
        for split, targets in targets_by_split.items()
    }
    for target_index, _target in enumerate(TARGETS):
        train_targets = targets_by_split["train"][:, [target_index]]
        if model_type == "random_forest":
            train_targets = train_targets.reshape(-1)
        trained = train_selected_model(
            model_type,
            features_by_split["train"],
            train_targets,
            feature_names,
            ridge_alpha=30.0,
            elasticnet_alpha=0.05,
            elasticnet_l1_ratio=0.35,
            random_state=random_state,
        )
        for split, matrix in features_by_split.items():
            predictions = predict_selected_model(trained, matrix)
            predictions_by_split[split][:, target_index] = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return predictions_by_split


def evaluate_predictions(targets_by_split: dict[str, np.ndarray], predictions_by_split: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split, targets in targets_by_split.items():
        if targets.shape[0] == 0:
            metrics[split] = {"overall_mae": "", "mae_by_target": {target: "" for target in TARGETS}}
            continue
        errors = np.abs(predictions_by_split[split] - targets)
        mae_by_target = {
            target: float(errors[:, index].mean())
            for index, target in enumerate(TARGETS)
        }
        metrics[split] = {"overall_mae": _mean(list(mae_by_target.values())), "mae_by_target": mae_by_target}
    return metrics


def build_split_variants(
    split_samples: dict[str, list[dict[str, Any]]],
    ambiguous_sample_ids: set[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        "full_baseline": {split: list(samples) for split, samples in split_samples.items()},
        "clean_train_only": {
            "train": clean_samples(split_samples["train"], ambiguous_sample_ids),
            "val": list(split_samples["val"]),
            "test": list(split_samples["test"]),
        },
        "clean_train_clean_test": {
            split: clean_samples(samples, ambiguous_sample_ids)
            for split, samples in split_samples.items()
        },
        "ambiguous_only_eval": {
            "train": list(split_samples["train"]),
            "val": list(split_samples["val"]),
            "test": ambiguous_samples(split_samples["test"], ambiguous_sample_ids),
        },
    }


def clean_samples(samples: list[dict[str, Any]], ambiguous_sample_ids: set[str]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["sample_id"] not in ambiguous_sample_ids]


def ambiguous_samples(samples: list[dict[str, Any]], ambiguous_sample_ids: set[str]) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["sample_id"] in ambiguous_sample_ids]


def filter_manifest_rows(
    manifest_rows: list[dict[str, str]],
    ambiguous_sample_ids: set[str],
    mode: str,
) -> list[dict[str, str]]:
    if mode == "clean_train_only":
        return [
            row
            for row in manifest_rows
            if not (row.get("dataset_split") == "train" and row.get("sample_id") in ambiguous_sample_ids)
        ]
    if mode == "clean_train_clean_test":
        return [row for row in manifest_rows if row.get("sample_id") not in ambiguous_sample_ids]
    raise ValueError(f"Unknown manifest filter mode: {mode}")


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest.csv does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def summarize_clean_ambiguous_ranges(
    proxy_names: list[str],
    proxy_matrix: np.ndarray,
    target_matrix: np.ndarray,
    ambiguity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    group_flags = np.asarray([bool(row["group_is_ambiguous"]) for row in ambiguity_rows], dtype=bool)
    summary: dict[str, Any] = {}
    for target_index, target in enumerate(TARGETS):
        indices = target_proxy_indices(target, proxy_names)
        if not indices:
            continue
        label_values = target_matrix[:, target_index]
        correlations = []
        for proxy_index in indices:
            corr = pearson_correlation(proxy_matrix[:, proxy_index], label_values)
            correlations.append((abs(corr), corr, proxy_index))
        _abs_corr, corr, best_index = max(correlations, key=lambda item: item[0])
        values = proxy_matrix[:, best_index]
        summary[target] = {
            "best_proxy": proxy_names[best_index],
            "best_proxy_correlation": corr,
            "clean": {
                "label": range_stats(label_values[~group_flags]),
                "proxy": range_stats(values[~group_flags]),
            },
            "ambiguous": {
                "label": range_stats(label_values[group_flags]),
                "proxy": range_stats(values[group_flags]),
            },
        }
    return summary


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def range_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def build_score_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distributions = {"group": score_distribution([row["group_ambiguity_score"] for row in rows])}
    for target in TARGETS:
        distributions[target] = score_distribution([row[f"{target}_ambiguity_score"] for row in rows])
    return distributions


def score_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p85": float(np.percentile(array, 85.0)),
        "max": float(array.max()),
    }


def validate_optional_phase3y_artifacts(phase3y_artifacts: str | Path | None) -> list[str]:
    if phase3y_artifacts is None:
        return []
    path = Path(phase3y_artifacts)
    if not path.exists():
        return [f"Optional Phase 3Y artifact directory was not found; recomputed geometry proxies from dataset: {path}"]
    missing = []
    for filename in ("label_geometry_correlations.csv", "ambiguity_pairs.csv"):
        if not (path / filename).exists():
            missing.append(filename)
    if missing:
        return [f"Optional Phase 3Y artifact files missing ({', '.join(missing)}); recomputed needed values from dataset."]
    return []


def interpretation(best_run: dict[str, Any], ambiguous_count: int, sample_count: int) -> str:
    if float(best_run["test_group_mae"]) <= 5.0:
        return "Clean subset filtering moved the focused target group into assisted-measurement range."
    if ambiguous_count == 0:
        return "No ambiguous samples were identified, so ambiguity filtering is unlikely to explain the remaining MAE ceiling."
    return "Filtering label-geometry collisions did not move the focused target group below 5 cm; generator/label fidelity likely remains a bottleneck."


def percentage(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator * 100.0)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ambiguity_score_fieldnames() -> list[str]:
    fields = ["sample_id", "dataset_split"]
    for target in TARGETS:
        fields.extend(
            [
                f"{target}_ambiguity_score",
                f"{target}_nearest_ambiguous_neighbor",
                f"{target}_nearest_label_diff",
                f"{target}_is_ambiguous",
            ]
        )
    fields.extend(["group_ambiguity_score", "group_is_ambiguous"])
    return fields


def ambiguous_pair_fieldnames() -> list[str]:
    return [
        "target",
        "sample_id_a",
        "sample_id_b",
        "geometry_distance",
        "label_diff",
        "standardized_label_diff",
        "ambiguity_score",
    ]


def benchmark_fieldnames() -> list[str]:
    return [
        "run_name",
        "variant",
        "feature_config",
        "model_type",
        "mode",
        "feature_count",
        "train_count",
        "val_count",
        "test_count",
        "filtered_train_count",
        "filtered_val_count",
        "filtered_test_count",
        "train_group_mae",
        "val_group_mae",
        "test_group_mae",
        "promotion_gate",
        "worst_target",
        "best_target",
    ]


def per_target_fieldnames() -> list[str]:
    return ["run_name", "variant", "feature_config", "model_type", "target", "test_mae", "promotion_gate"]


def format_summary(summary: dict[str, Any]) -> str:
    best = summary["best_run"]
    lines = [
        "# Phase 3Z Label-Geometry Ambiguity Filtering",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Feature extractor: `{summary['feature_extractor_version']}`",
        f"Ambiguity threshold percentile: {summary['ambiguity_percentile']:.1f}",
        f"Ambiguous samples: {summary['ambiguous_sample_count']} / {summary['sample_count']} ({summary['ambiguous_sample_percent']:.1f}%)",
        "",
        "## Best Clean-Subset Benchmark",
        "",
        f"Best run: `{best['run_name']}`",
        f"Test group MAE: {float(best['test_group_mae']):.4f}",
        f"Promotion gate: `{best['promotion_gate']}`",
        f"Group below 5 cm: `{summary['group_below_5cm']}`",
        "",
        "## Benchmark Results",
        "",
        "| Variant | Model | Train Count | Test Count | Test Group MAE | Worst Target |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["benchmark_results"]:
        lines.append(
            f"| {row['variant']} | {row['model_type']} | {row['train_count']} | {row['test_count']} | {float(row['test_group_mae']):.4f} | {row['worst_target']} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter label-geometry ambiguity and benchmark clean subset variants.")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root, such as data/synthetic/phase_3t.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument("--phase3y-artifacts", help="Optional Phase 3Y artifact directory for provenance checks.")
    parser.add_argument("--ambiguity-percentile", type=float, default=DEFAULT_AMBIGUITY_PERCENTILE)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODEL_TYPES, help="Model types to benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = filter_label_geometry_ambiguity(
        args.dataset,
        args.output,
        phase3y_artifacts=args.phase3y_artifacts,
        ambiguity_percentile=args.ambiguity_percentile,
        model_types=args.models,
        random_state=args.seed,
    )
    best = result["summary"]["best_run"]
    print(f"Ambiguous samples: {result['summary']['ambiguous_sample_count']} / {result['summary']['sample_count']}")
    print(f"Best run: {best['run_name']} test group MAE {best['test_group_mae']:.4f}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
