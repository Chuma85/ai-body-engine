from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.validate_synthetic_dataset import validate_dataset
from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.features.back_view_features import (
    FEATURE_EXTRACTOR_VERSION as BACK_FEATURE_EXTRACTOR_VERSION,
    extract_back_view_features,
    extract_front_side_back_features,
    get_back_view_feature_names,
    get_front_side_back_feature_names,
)
from training.features.image_silhouette_features import FEATURE_EXTRACTOR_VERSION, extract_front_side_features, get_feature_names
from training.train_baseline_measurements import _mean
from training.train_image_feature_baseline import predict_feature_regressor, train_ridge_regressor

DEFAULT_DATASET = "artifacts/phase_5l_back_view_shoulder_benchmark/dataset"
DEFAULT_OUTPUT = "artifacts/phase_5l_back_view_shoulder_benchmark"
DEFAULT_SAMPLE_COUNT = 360
DEFAULT_SEED = 50512
MODEL_TYPES = ["ridge", "elasticnet", "random_forest", "gradient_boosting"]
SHOULDER_TARGET_CANDIDATES = ["shoulder_cm", "across_back_cm", "upper_back_cm"]
REFERENCE_TARGET_CANDIDATES = ["chest_cm", "waist_cm", "hip_cm", "thigh_cm"]


def run_back_view_shoulder_benchmark(
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    model_types: list[str] | None = None,
    regenerate_dataset: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if regenerate_dataset or not (dataset_path / "manifest.csv").exists():
        generate_controlled_back_view_dataset(dataset_path, sample_count=sample_count, seed=seed)

    validation = validate_dataset(dataset_path, require_back=True)
    if not validation["valid"]:
        raise ValueError("Phase 5L dataset is invalid: " + "; ".join(validation["errors"]))
    manifest_result = build_dataset_manifest(dataset_path, require_back=True)
    if not manifest_result["valid"]:
        raise ValueError("Phase 5L manifest is invalid: " + "; ".join(manifest_result["errors"]))

    samples_by_split = {
        split: list(SyntheticBodyDataset(dataset_path, split=split))
        for split in ("train", "val", "test")
    }
    all_samples = [sample for split in ("train", "val", "test") for sample in samples_by_split[split]]
    targets = available_targets(all_samples)
    shoulder_targets = [target for target in SHOULDER_TARGET_CANDIDATES if target in targets]
    reference_targets = [target for target in REFERENCE_TARGET_CANDIDATES if target in targets]
    selected_models = model_types or MODEL_TYPES

    feature_sets = build_feature_sets(samples_by_split)
    run_rows: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []
    for feature_set_name, payload in feature_sets.items():
        for model_type in selected_models:
            try:
                predictions_by_split = train_predict_model(
                    model_type,
                    payload["features_by_split"],
                    target_matrices(samples_by_split, targets),
                    targets,
                    payload["feature_names"],
                    seed,
                )
            except Exception as error:  # pragma: no cover - optional sklearn runtime failures
                skipped_runs.append({"run_name": f"{feature_set_name}__{model_type}", "reason": f"{type(error).__name__}: {error}"})
                continue
            metrics = evaluate_predictions(predictions_by_split, target_matrices(samples_by_split, targets), targets)
            run_name = f"{feature_set_name}__{model_type}"
            run_rows.append(run_row(run_name, feature_set_name, model_type, len(payload["feature_names"]), metrics, shoulder_targets, reference_targets))
            per_target_rows.extend(per_target_result_rows(run_name, feature_set_name, model_type, metrics, targets))

    residual_rows, residual_per_target_rows, residual_skips = run_geometry_residual_compatible_benchmark(
        feature_sets["front_side_back_combined"],
        samples_by_split,
        targets,
        shoulder_targets,
        selected_models,
        seed,
    )
    run_rows.extend(residual_rows)
    per_target_rows.extend(residual_per_target_rows)
    skipped_runs.extend(residual_skips)

    summary = build_summary(
        dataset_path,
        validation,
        manifest_result,
        run_rows,
        per_target_rows,
        skipped_runs,
        targets,
        shoulder_targets,
        reference_targets,
        sample_count=sum(len(rows) for rows in samples_by_split.values()),
    )
    write_artifacts(output_path, validation, summary, run_rows, per_target_rows)
    return {"output_dir": str(output_path), "summary": summary}


def generate_controlled_back_view_dataset(dataset_root: str | Path, sample_count: int, seed: int) -> Path:
    dataset_path = Path(dataset_root)
    front_dir = dataset_path / "images" / "front"
    side_dir = dataset_path / "images" / "side"
    back_dir = dataset_path / "images" / "back"
    labels_dir = dataset_path / "labels"
    for directory in (front_dir, side_dir, back_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    fieldnames = [*LABEL_COLUMNS, "across_back_cm", "upper_back_cm", "shoulder_asymmetry_cm"]
    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, sample_count + 1):
            row = synthetic_measurement_row(index, rng)
            sample_id = row["sample_id"]
            front_path = front_dir / f"{sample_id}_front.png"
            side_path = side_dir / f"{sample_id}_side.png"
            back_path = back_dir / f"{sample_id}_back.png"
            render_controlled_view(row, "front", front_path)
            render_controlled_view(row, "side", side_path)
            render_controlled_view(row, "back", back_path)
            row.update(
                {
                    "front_image_path": front_path.as_posix(),
                    "side_image_path": side_path.as_posix(),
                    "back_image_path": back_path.as_posix(),
                    "has_front": "true",
                    "has_side": "true",
                    "has_back": "true",
                    "capture_views": "front,side,back",
                    "minimum_scan_views": "front,side",
                    "enhanced_scan_views": "front,side,back",
                    "generator_version": "phase_5l_back_view_silhouette_benchmark_v1",
                }
            )
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    build_dataset_manifest(dataset_path, require_back=True)
    return labels_dir / "labels.csv"


def synthetic_measurement_row(index: int, rng: random.Random) -> dict[str, Any]:
    body_shape = rng.choice(["slim", "average", "athletic", "curvy", "broad"])
    shape_bias = {"slim": -0.10, "average": 0.0, "athletic": 0.08, "curvy": 0.07, "broad": 0.14}[body_shape]
    height_cm = round(rng.uniform(152, 202), 1)
    shoulder_cm = round(_bounded(rng.gauss(45.0 + shape_bias * 18.0, 4.8), 34.0, 62.0), 1)
    chest_cm = round(_bounded(rng.gauss(96.0 + shape_bias * 24.0, 12.0) + (shoulder_cm - 45.0) * 0.7, 74.0, 138.0), 1)
    waist_cm = round(_bounded(rng.gauss(80.0 + shape_bias * 22.0, 13.0), 54.0, 126.0), 1)
    hip_cm = round(_bounded(rng.gauss(98.0 + shape_bias * 18.0, 12.0), 76.0, 138.0), 1)
    thigh_cm = round(_bounded(rng.gauss(55.0 + shape_bias * 15.0, 7.5), 38.0, 82.0), 1)
    calf_cm = round(_bounded(rng.gauss(37.0 + shape_bias * 8.0, 4.5), 28.0, 56.0), 1)
    neck_cm = round(_bounded(rng.gauss(38.0 + shape_bias * 8.0, 3.0), 30.0, 52.0), 1)
    asymmetry = round(rng.uniform(-2.2, 2.2), 2)
    across_back_cm = round(_bounded(shoulder_cm * 0.82 + chest_cm * 0.08 + rng.gauss(0.0, 0.75), 34.0, 62.0), 1)
    upper_back_cm = round(_bounded(shoulder_cm * 0.58 + chest_cm * 0.32 + rng.gauss(0.0, 0.9), 42.0, 88.0), 1)
    return {
        "sample_id": f"sample_{index:06d}",
        "height_cm": height_cm,
        "weight_kg": round(_bounded((chest_cm + waist_cm + hip_cm) * 0.36 + rng.gauss(0.0, 5.0), 45.0, 135.0), 1),
        "chest_cm": chest_cm,
        "waist_cm": waist_cm,
        "hip_cm": hip_cm,
        "shoulder_cm": shoulder_cm,
        "inseam_cm": round(_bounded(height_cm * rng.uniform(0.43, 0.48), 65.0, 96.0), 1),
        "sleeve_cm": round(_bounded(height_cm * rng.uniform(0.32, 0.37), 50.0, 76.0), 1),
        "neck_cm": neck_cm,
        "thigh_cm": thigh_cm,
        "calf_cm": calf_cm,
        "body_shape": body_shape,
        "across_back_cm": across_back_cm,
        "upper_back_cm": upper_back_cm,
        "shoulder_asymmetry_cm": asymmetry,
    }


def render_controlled_view(row: dict[str, Any], view: str, output_path: Path, width: int = 192, height: int = 256) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    cx = width // 2
    top = int(height * 0.07)
    bottom = int(height * 0.94)
    body_height = bottom - top
    shoulder_y = top + int(body_height * 0.19)
    upper_y = top + int(body_height * 0.28)
    chest_y = top + int(body_height * 0.34)
    waist_y = top + int(body_height * 0.48)
    hip_y = top + int(body_height * 0.60)
    knee_y = top + int(body_height * 0.79)
    ankle_y = bottom - int(body_height * 0.03)

    if view == "side":
        torso_widths = {
            "shoulder": _scale(row["shoulder_cm"], 34, 62, width * 0.09, width * 0.16),
            "chest": _scale(row["chest_cm"], 74, 138, width * 0.11, width * 0.23),
            "waist": _scale(row["waist_cm"], 54, 126, width * 0.10, width * 0.21),
            "hip": _scale(row["hip_cm"], 76, 138, width * 0.12, width * 0.25),
        }
        torso = [
            (cx - torso_widths["shoulder"] * 0.35, shoulder_y),
            (cx + torso_widths["shoulder"] * 0.75, shoulder_y),
            (cx + torso_widths["chest"], chest_y),
            (cx + torso_widths["waist"], waist_y),
            (cx + torso_widths["hip"], hip_y),
            (cx - torso_widths["hip"] * 0.30, hip_y),
            (cx - torso_widths["waist"] * 0.30, waist_y),
            (cx - torso_widths["chest"] * 0.30, chest_y),
        ]
    else:
        shoulder_source = row["shoulder_cm"]
        if view == "front":
            shoulder_source = row["shoulder_cm"] * 0.62 + row["chest_cm"] * 0.16
            upper_source = row["chest_cm"] * 0.78 + row["shoulder_cm"] * 0.18
        else:
            shoulder_source = row["shoulder_cm"] + row["shoulder_asymmetry_cm"] * 0.35
            upper_source = row["upper_back_cm"]
        shoulder_w = _scale(shoulder_source, 34, 92, width * 0.22, width * 0.44)
        upper_w = _scale(upper_source, 42, 138, width * 0.20, width * 0.38)
        waist_w = _scale(row["waist_cm"], 54, 126, width * 0.13, width * 0.31)
        hip_w = _scale(row["hip_cm"], 76, 138, width * 0.19, width * 0.38)
        torso = [
            (cx - shoulder_w / 2, shoulder_y),
            (cx - upper_w / 2, upper_y),
            (cx - waist_w / 2, waist_y),
            (cx - hip_w / 2, hip_y),
            (cx + hip_w / 2, hip_y),
            (cx + waist_w / 2, waist_y),
            (cx + upper_w / 2, upper_y),
            (cx + shoulder_w / 2, shoulder_y),
        ]
    head_r = int(body_height * 0.055)
    draw.ellipse((cx - head_r, top, cx + head_r, top + head_r * 2), fill="black")
    draw.rounded_rectangle((cx - width * 0.03, top + int(body_height * 0.13), cx + width * 0.03, shoulder_y + 8), radius=5, fill="black")
    draw.polygon(torso, fill="black")

    if view != "side":
        shoulder_w = max(abs(torso[-1][0] - torso[0][0]), width * 0.24)
        hip_w = max(abs(torso[4][0] - torso[3][0]), width * 0.20)
        arm_w = max(7, int(width * 0.035))
        for side in (-1, 1):
            shoulder_x = cx + side * (shoulder_w / 2)
            wrist_x = cx + side * (hip_w / 2 + width * (0.08 if view == "back" else 0.10))
            wrist_y = hip_y
            draw.line((shoulder_x, shoulder_y + 8, wrist_x, wrist_y), fill="black", width=arm_w)

    leg_w = _scale(row["thigh_cm"], 38, 82, width * 0.07, width * 0.15)
    calf_w = _scale(row["calf_cm"], 28, 56, width * 0.045, width * 0.095)
    gap = int(width * 0.025)
    for side in (-1, 1):
        leg = [
            (cx + side * gap, hip_y),
            (cx + side * leg_w, hip_y),
            (cx + side * leg_w * 0.8, knee_y),
            (cx + side * calf_w, ankle_y),
            (cx + side * gap * 0.7, ankle_y),
            (cx + side * gap, knee_y),
        ]
        draw.polygon(leg, fill="black")
    image.save(output_path)


def build_feature_sets(samples_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    front_side_names = get_feature_names()
    back_names = get_back_view_feature_names()
    combined_names = get_front_side_back_feature_names(front_side_names)
    return {
        "front_side_baseline": {
            "feature_names": front_side_names,
            "features_by_split": {
                split: feature_matrix(samples, front_side_names, "front_side")
                for split, samples in samples_by_split.items()
            },
        },
        "back_only": {
            "feature_names": back_names,
            "features_by_split": {
                split: feature_matrix(samples, back_names, "back_only")
                for split, samples in samples_by_split.items()
            },
        },
        "front_side_back_combined": {
            "feature_names": combined_names,
            "features_by_split": {
                split: feature_matrix(samples, combined_names, "combined")
                for split, samples in samples_by_split.items()
            },
        },
    }


def feature_matrix(samples: list[dict[str, Any]], feature_names: list[str], feature_set: str) -> np.ndarray:
    rows = []
    for sample in samples:
        if feature_set == "front_side":
            features = extract_front_side_features(sample["front_image_path"], sample["side_image_path"])
        elif feature_set == "back_only":
            features = extract_back_view_features(sample.get("back_image_path"))
        elif feature_set == "combined":
            features = extract_front_side_back_features(sample["front_image_path"], sample["side_image_path"], sample.get("back_image_path"))
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        rows.append([float(features[name]) for name in feature_names])
    return np.asarray(rows, dtype=np.float64)


def available_targets(samples: list[dict[str, Any]]) -> list[str]:
    candidates = [*SHOULDER_TARGET_CANDIDATES, *REFERENCE_TARGET_CANDIDATES]
    return [
        target
        for target in candidates
        if all(sample["labels"].get(target) not in ("", None) for sample in samples)
    ]


def target_matrices(samples_by_split: dict[str, list[dict[str, Any]]], targets: list[str]) -> dict[str, np.ndarray]:
    return {
        split: np.asarray(
            [[float(sample["labels"][target]) for target in targets] for sample in samples],
            dtype=np.float64,
        )
        for split, samples in samples_by_split.items()
    }


def train_predict_model(
    model_type: str,
    features_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    targets: list[str],
    feature_names: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    train_features = features_by_split["train"]
    train_targets = targets_by_split["train"]
    if model_type == "ridge":
        model = train_ridge_regressor(train_features, train_targets, feature_names, targets, ridge_alpha=30.0)
        return {split: predict_feature_regressor(model, matrix) for split, matrix in features_by_split.items()}

    sklearn = require_sklearn()
    if model_type == "elasticnet":
        estimator = sklearn["MultiOutputRegressor"](
            sklearn["ElasticNet"](alpha=0.03, l1_ratio=0.25, max_iter=10000, random_state=seed)
        )
        means = train_features.mean(axis=0)
        stds = np.where(train_features.std(axis=0) < 1e-8, 1.0, train_features.std(axis=0))
        estimator.fit((train_features - means) / stds, train_targets)
        return {split: np.asarray(estimator.predict((matrix - means) / stds), dtype=np.float64) for split, matrix in features_by_split.items()}
    if model_type == "random_forest":
        estimator = sklearn["RandomForestRegressor"](n_estimators=70, max_depth=9, min_samples_leaf=2, random_state=seed, n_jobs=1)
    elif model_type == "gradient_boosting":
        estimator = sklearn["MultiOutputRegressor"](
            sklearn["GradientBoostingRegressor"](n_estimators=70, max_depth=3, random_state=seed)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    estimator.fit(train_features, train_targets)
    return {split: np.asarray(estimator.predict(matrix), dtype=np.float64) for split, matrix in features_by_split.items()}


def run_geometry_residual_compatible_benchmark(
    combined_feature_set: dict[str, Any],
    samples_by_split: dict[str, list[dict[str, Any]]],
    targets: list[str],
    shoulder_targets: list[str],
    model_types: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    if not shoulder_targets:
        return [], [], []
    feature_names = combined_feature_set["feature_names"]
    proxy_names = [name for name in geometry_proxy_feature_names() if name in feature_names]
    proxy_indices = [feature_names.index(name) for name in proxy_names]
    proxy_features_by_split = {
        split: matrix[:, proxy_indices]
        for split, matrix in combined_feature_set["features_by_split"].items()
    }
    shoulder_indices = [targets.index(target) for target in shoulder_targets]
    targets_by_split = target_matrices(samples_by_split, targets)
    shoulder_targets_by_split = {
        split: matrix[:, shoulder_indices]
        for split, matrix in targets_by_split.items()
    }
    direct_predictions = train_predict_model(
        "ridge",
        proxy_features_by_split,
        shoulder_targets_by_split,
        shoulder_targets,
        proxy_names,
        seed,
    )
    residual_targets_by_split = {
        split: shoulder_targets_by_split[split] - direct_predictions[split]
        for split in ("train", "val", "test")
    }
    rows: list[dict[str, Any]] = []
    per_target: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    for model_type in model_types:
        try:
            residual_predictions = train_predict_model(
                model_type,
                combined_feature_set["features_by_split"],
                residual_targets_by_split,
                shoulder_targets,
                feature_names,
                seed,
            )
        except Exception as error:  # pragma: no cover
            skips.append({"run_name": f"geometry_plus_residual__{model_type}", "reason": f"{type(error).__name__}: {error}"})
            continue
        final_predictions = {
            split: direct_predictions[split] + residual_predictions[split]
            for split in ("train", "val", "test")
        }
        metrics = evaluate_predictions(final_predictions, shoulder_targets_by_split, shoulder_targets)
        run_name = f"geometry_plus_residual_front_side_back__{model_type}"
        rows.append(run_row(run_name, "geometry_plus_residual", model_type, len(feature_names) + len(proxy_names), metrics, shoulder_targets, []))
        per_target.extend(per_target_result_rows(run_name, "geometry_plus_residual", model_type, metrics, shoulder_targets))
    return rows, per_target, skips


def geometry_proxy_feature_names() -> list[str]:
    return [
        "back_shoulder_width_proxy",
        "back_across_back_width_proxy",
        "back_upper_back_width_proxy",
        "back_upper_back_area_proxy",
        "front_shoulder_peak_width_ratio",
        "front_upper_chest_width_ratio",
        "side_shoulder_peak_width_ratio",
        "front_side_back_shoulder_volume_proxy",
        "front_side_back_upper_torso_volume_proxy",
    ]


def evaluate_predictions(
    predictions_by_split: dict[str, np.ndarray],
    targets_by_split: dict[str, np.ndarray],
    targets: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        errors = np.abs(predictions_by_split[split] - targets_by_split[split])
        mae_by_target = {target: float(errors[:, index].mean()) for index, target in enumerate(targets)}
        metrics[split] = {"overall_mae": _mean(list(mae_by_target.values())), "mae_by_target": mae_by_target}
    return metrics


def run_row(
    run_name: str,
    feature_set: str,
    model_type: str,
    feature_count: int,
    metrics: dict[str, Any],
    shoulder_targets: list[str],
    reference_targets: list[str],
) -> dict[str, Any]:
    test_mae = metrics["test"]["mae_by_target"]
    shoulder_values = [test_mae[target] for target in shoulder_targets if target in test_mae]
    reference_values = [test_mae[target] for target in reference_targets if target in test_mae]
    return {
        "run_name": run_name,
        "feature_set": feature_set,
        "model_type": model_type,
        "feature_count": feature_count,
        "train_overall_mae": metrics["train"]["overall_mae"],
        "val_overall_mae": metrics["val"]["overall_mae"],
        "test_overall_mae": metrics["test"]["overall_mae"],
        "test_shoulder_group_mae": _mean(shoulder_values) if shoulder_values else "",
        "test_reference_group_mae": _mean(reference_values) if reference_values else "",
        "test_shoulder_cm_mae": test_mae.get("shoulder_cm", ""),
        "test_across_back_cm_mae": test_mae.get("across_back_cm", ""),
        "test_upper_back_cm_mae": test_mae.get("upper_back_cm", ""),
        "test_chest_cm_mae": test_mae.get("chest_cm", ""),
        "test_waist_cm_mae": test_mae.get("waist_cm", ""),
        "test_hip_cm_mae": test_mae.get("hip_cm", ""),
        "test_thigh_cm_mae": test_mae.get("thigh_cm", ""),
    }


def per_target_result_rows(
    run_name: str,
    feature_set: str,
    model_type: str,
    metrics: dict[str, Any],
    targets: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "run_name": run_name,
            "feature_set": feature_set,
            "model_type": model_type,
            "target": target,
            "train_mae": metrics["train"]["mae_by_target"][target],
            "val_mae": metrics["val"]["mae_by_target"][target],
            "test_mae": metrics["test"]["mae_by_target"][target],
        }
        for target in targets
    ]


def build_summary(
    dataset_path: Path,
    validation: dict[str, Any],
    manifest_result: dict[str, Any],
    run_rows: list[dict[str, Any]],
    per_target_rows: list[dict[str, Any]],
    skipped_runs: list[dict[str, str]],
    targets: list[str],
    shoulder_targets: list[str],
    reference_targets: list[str],
    sample_count: int,
) -> dict[str, Any]:
    if not run_rows:
        raise ValueError("No Phase 5L benchmark runs completed.")
    baseline_rows = [row for row in run_rows if row["feature_set"] == "front_side_baseline" and row["test_shoulder_group_mae"] != ""]
    combined_rows = [row for row in run_rows if row["feature_set"] == "front_side_back_combined" and row["test_shoulder_group_mae"] != ""]
    back_rows = [row for row in run_rows if row["feature_set"] == "back_only" and row["test_shoulder_group_mae"] != ""]
    best_baseline = min(baseline_rows, key=lambda row: float(row["test_shoulder_group_mae"]))
    best_combined = min(combined_rows, key=lambda row: float(row["test_shoulder_group_mae"]))
    best_back = min(back_rows, key=lambda row: float(row["test_shoulder_group_mae"])) if back_rows else {}
    improvement = float(best_baseline["test_shoulder_group_mae"]) - float(best_combined["test_shoulder_group_mae"])
    improvement_pct = improvement / max(float(best_baseline["test_shoulder_group_mae"]), 1e-9) * 100.0
    reference_delta = reference_group_delta(best_baseline, best_combined)
    recommendation = recommendation_text(improvement_pct, reference_delta)
    return {
        "dataset": str(dataset_path),
        "feature_extractor_versions": {
            "front_side": FEATURE_EXTRACTOR_VERSION,
            "back_view": BACK_FEATURE_EXTRACTOR_VERSION,
        },
        "sample_count": sample_count,
        "split_counts": manifest_result["split_counts"],
        "targets": targets,
        "shoulder_targets": shoulder_targets,
        "reference_targets": reference_targets,
        "validation": validation,
        "best_front_side_baseline": best_baseline,
        "best_back_only": best_back,
        "best_front_side_back": best_combined,
        "shoulder_group_mae_improvement_cm": improvement,
        "shoulder_group_mae_improvement_pct": improvement_pct,
        "reference_group_mae_delta_cm": reference_delta,
        "back_view_improves_shoulder": improvement > 0.0,
        "back_view_worsens_reference_targets": reference_delta > 0.25,
        "recommendation": recommendation,
        "benchmark_results": sorted(run_rows, key=lambda row: (str(row["feature_set"]), float(row["test_overall_mae"]))),
        "per_target_results": per_target_rows,
        "skipped_runs": skipped_runs,
    }


def reference_group_delta(best_baseline: dict[str, Any], best_combined: dict[str, Any]) -> float:
    if best_baseline.get("test_reference_group_mae") == "" or best_combined.get("test_reference_group_mae") == "":
        return 0.0
    return float(best_combined["test_reference_group_mae"]) - float(best_baseline["test_reference_group_mae"])


def recommendation_text(improvement_pct: float, reference_delta: float) -> str:
    if improvement_pct >= 10.0 and reference_delta <= 0.25:
        return "Back capture should remain optional in product, but should be recommended for structured garments and shoulder-sensitive fits."
    if improvement_pct > 0.0:
        return "Back capture shows directional synthetic benefit, but the improvement is not strong enough to make it mandatory."
    return "Back capture should remain optional; this synthetic benchmark did not show enough shoulder improvement."


def write_artifacts(
    output_path: Path,
    validation: dict[str, Any],
    summary: dict[str, Any],
    run_rows: list[dict[str, Any]],
    per_target_rows: list[dict[str, Any]],
) -> None:
    write_json(output_path / "dataset_validation.json", validation)
    write_csv(output_path / "dataset_validation.csv", [flatten_validation(validation)])
    write_json(output_path / "benchmark_results.json", summary)
    write_csv(output_path / "benchmark_results.csv", run_rows)
    write_csv(output_path / "per_target_results.csv", per_target_rows)
    (output_path / "back_view_feature_summary.md").write_text(format_feature_summary(summary), encoding="utf-8")
    (output_path / "recommendation_summary.md").write_text(format_recommendation_summary(summary), encoding="utf-8")


def flatten_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": validation["valid"],
        "dataset": validation["dataset"],
        "sample_count": validation["sample_count"],
        "front_image_count": validation["front_image_count"],
        "side_image_count": validation["side_image_count"],
        "back_image_count": validation["back_image_count"],
        "label_row_count": validation["label_row_count"],
        "errors": ";".join(validation["errors"]),
        "warnings": ";".join(validation["warnings"]),
    }


def format_feature_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 5L Back-View Feature Summary",
            "",
            f"Dataset: `{summary['dataset']}`",
            f"Samples: {summary['sample_count']}",
            f"Targets: {', '.join(summary['targets'])}",
            "",
            "Back-view features added:",
            "",
            "- Back shoulder width proxy from shoulder peak width.",
            "- Across-back proxy from upper-body maximum width.",
            "- Upper-back width and area proxies.",
            "- Shoulder slope proxy.",
            "- Back torso width bands at shoulder, upper-chest, chest, mid-torso, waist, and hip levels.",
            "- Front/back shoulder comparison and combined front/side/back volume proxies.",
            "",
            f"Back feature extractor: `{summary['feature_extractor_versions']['back_view']}`",
            "",
        ]
    )


def format_recommendation_summary(summary: dict[str, Any]) -> str:
    baseline = summary["best_front_side_baseline"]
    combined = summary["best_front_side_back"]
    back_only = summary.get("best_back_only") or {}
    lines = [
        "# Phase 5L Recommendation Summary",
        "",
        f"Best front+side shoulder group MAE: {float(baseline['test_shoulder_group_mae']):.4f} cm (`{baseline['run_name']}`)",
        f"Best front+side+back shoulder group MAE: {float(combined['test_shoulder_group_mae']):.4f} cm (`{combined['run_name']}`)",
    ]
    if back_only:
        lines.append(f"Best back-only shoulder group MAE: {float(back_only['test_shoulder_group_mae']):.4f} cm (`{back_only['run_name']}`)")
    lines.extend(
        [
            f"Shoulder group improvement: {summary['shoulder_group_mae_improvement_cm']:.4f} cm ({summary['shoulder_group_mae_improvement_pct']:.2f}%)",
            f"Reference target group MAE delta: {summary['reference_group_mae_delta_cm']:.4f} cm",
            f"Back view improves shoulder group: `{summary['back_view_improves_shoulder']}`",
            f"Back view worsens reference targets: `{summary['back_view_worsens_reference_targets']}`",
            "",
            "Recommendation:",
            "",
            summary["recommendation"],
            "",
            "This remains a synthetic benchmark and is not a real-world production-readiness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError as error:
        raise ImportError("scikit-learn is required for this model type.") from error
    return {
        "ElasticNet": ElasticNet,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "MultiOutputRegressor": MultiOutputRegressor,
        "RandomForestRegressor": RandomForestRegressor,
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    ratio = (float(value) - source_min) / (source_max - source_min)
    ratio = max(0.0, min(1.0, ratio))
    return target_min + ratio * (target_max - target_min)


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark optional back-view shoulder features.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--models", nargs="*", default=MODEL_TYPES)
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args(argv)
    result = run_back_view_shoulder_benchmark(
        dataset_root=args.dataset,
        output_dir=args.output,
        sample_count=args.sample_count,
        seed=args.seed,
        model_types=args.models,
        regenerate_dataset=not args.reuse_dataset,
    )
    summary = result["summary"]
    print(f"Shoulder improvement: {summary['shoulder_group_mae_improvement_cm']:.4f} cm ({summary['shoulder_group_mae_improvement_pct']:.2f}%)")
    print(f"Recommendation: {summary['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
