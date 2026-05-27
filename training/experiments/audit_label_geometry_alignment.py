from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from training.datasets.synthetic_body_dataset import SyntheticBodyDataset
from training.experiments.audit_measurement_bands import (
    pearson_correlation,
    thumbnail,
    write_contact_sheet,
)
from training.features.measurement_band_features import (
    FEATURE_EXTRACTOR_VERSION as BAND_FEATURE_VERSION,
    MEASUREMENT_BAND_TARGETS,
    candidate_band_definitions,
    extract_front_side_band_features,
)

CORRELATIONS_JSON = "label_geometry_correlations.json"
CORRELATIONS_CSV = "label_geometry_correlations.csv"
CORRELATIONS_MD = "label_geometry_correlations.md"
MONOTONICITY_CSV = "monotonicity_checks.csv"
AMBIGUITY_CSV = "ambiguity_pairs.csv"
SUMMARY_MD = "deformation_realism_summary.md"
CONTACT_DIR = "visual_contact_sheets"


def audit_label_geometry_alignment(
    dataset: str | Path,
    output_dir: str | Path,
    ambiguity_pairs_per_target: int = 12,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples = list(SyntheticBodyDataset(dataset, split="all"))
    if not samples:
        raise ValueError(f"No samples available for label-geometry audit: {dataset}")

    sample_ids, proxy_names, proxy_matrix, target_matrix = extract_geometry_proxy_matrix(samples)
    correlation_rows = build_label_geometry_correlations(proxy_matrix, target_matrix, proxy_names)
    monotonicity_rows = build_monotonicity_rows(proxy_matrix, target_matrix, proxy_names)
    ambiguity_rows = find_geometry_label_ambiguity_pairs(
        sample_ids,
        proxy_matrix,
        target_matrix,
        proxy_names,
        ambiguity_pairs_per_target=ambiguity_pairs_per_target,
    )
    contact_warnings = write_alignment_contact_sheets(samples, ambiguity_rows, output_path / CONTACT_DIR)
    summary = build_deformation_summary(correlation_rows, monotonicity_rows, samples, dataset, contact_warnings)

    paths = write_outputs(output_path, summary, correlation_rows, monotonicity_rows, ambiguity_rows)
    return {key: str(value) for key, value in paths.items()} | {"summary": summary}


def extract_geometry_proxy_matrix(
    samples: list[dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    sample_ids: list[str] = []
    proxy_names: list[str] | None = None
    proxy_rows: list[list[float]] = []
    target_rows: list[list[float]] = []
    for sample in samples:
        proxies = extract_label_geometry_proxies(sample["front_image_path"], sample["side_image_path"])
        if proxy_names is None:
            proxy_names = list(proxies)
        sample_ids.append(sample["sample_id"])
        proxy_rows.append([float(proxies[name]) for name in proxy_names])
        target_rows.append([float(sample["measurements"][target]) for target in MEASUREMENT_BAND_TARGETS])
    return sample_ids, proxy_names or [], np.asarray(proxy_rows, dtype=np.float64), np.asarray(target_rows, dtype=np.float64)


def extract_label_geometry_proxies(front_image_path: str | Path, side_image_path: str | Path) -> dict[str, float]:
    band_features = extract_front_side_band_features(front_image_path, side_image_path)
    proxies: dict[str, float] = {}
    for definition in candidate_band_definitions():
        prefix = definition["band_name"]
        front_norm = band_features[f"{prefix}_front_norm_width_ratio"]
        side_norm = band_features[f"{prefix}_side_norm_width_ratio"]
        front_raw = band_features[f"{prefix}_front_raw_width_ratio"]
        side_raw = band_features[f"{prefix}_side_raw_width_ratio"]
        proxies[f"{prefix}_front_norm_width"] = front_norm
        proxies[f"{prefix}_side_norm_depth"] = side_norm
        proxies[f"{prefix}_front_side_norm_ratio"] = band_features[f"{prefix}_norm_front_side_width_ratio"]
        proxies[f"{prefix}_front_norm_local_area"] = band_features[f"{prefix}_front_norm_local_area_ratio"]
        proxies[f"{prefix}_side_norm_local_area"] = band_features[f"{prefix}_side_norm_local_area_ratio"]
        proxies[f"{prefix}_front_norm_contour_slope"] = band_features[f"{prefix}_front_norm_contour_slope"]
        proxies[f"{prefix}_side_norm_contour_slope"] = band_features[f"{prefix}_side_norm_contour_slope"]
        proxies[f"{prefix}_norm_width_depth_product"] = band_features[f"{prefix}_norm_width_depth_product"]
        proxies[f"{prefix}_norm_ellipse_circumference_proxy"] = ellipse_circumference_proxy(front_norm, side_norm)
        proxies[f"{prefix}_front_raw_width"] = front_raw
        proxies[f"{prefix}_side_raw_depth"] = side_raw
        proxies[f"{prefix}_front_side_raw_ratio"] = band_features[f"{prefix}_raw_front_side_width_ratio"]
        proxies[f"{prefix}_front_raw_local_area"] = band_features[f"{prefix}_front_raw_local_area_ratio"]
        proxies[f"{prefix}_side_raw_local_area"] = band_features[f"{prefix}_side_raw_local_area_ratio"]
        proxies[f"{prefix}_front_raw_contour_slope"] = band_features[f"{prefix}_front_raw_contour_slope"]
        proxies[f"{prefix}_side_raw_contour_slope"] = band_features[f"{prefix}_side_raw_contour_slope"]
        proxies[f"{prefix}_raw_width_depth_product"] = band_features[f"{prefix}_raw_width_depth_product"]
        proxies[f"{prefix}_raw_ellipse_circumference_proxy"] = ellipse_circumference_proxy(front_raw, side_raw)
        proxies[f"{prefix}_band_center_y_ratio"] = band_features[f"{prefix}_band_center_y_ratio"]
    return proxies


def ellipse_circumference_proxy(width: float, depth: float) -> float:
    a = max(float(width), 0.0) / 2.0
    b = max(float(depth), 0.0) / 2.0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float(math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def build_label_geometry_correlations(
    proxy_matrix: np.ndarray,
    target_matrix: np.ndarray,
    proxy_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for target_index, target in enumerate(MEASUREMENT_BAND_TARGETS):
        target_prefix = target.removesuffix("_cm")
        label_values = target_matrix[:, target_index]
        for proxy_index, proxy_name in enumerate(proxy_names):
            if not proxy_name.startswith(f"{target_prefix}_band_"):
                continue
            corr = pearson_correlation(proxy_matrix[:, proxy_index], label_values)
            rows.append(
                {
                    "target": target,
                    "band_name": proxy_band_name(proxy_name),
                    "proxy": proxy_name,
                    "proxy_role": proxy_role(proxy_name),
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                }
            )
    return rows


def build_monotonicity_rows(
    proxy_matrix: np.ndarray,
    target_matrix: np.ndarray,
    proxy_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for target_index, target in enumerate(MEASUREMENT_BAND_TARGETS):
        target_prefix = target.removesuffix("_cm")
        label_values = target_matrix[:, target_index]
        low_threshold, high_threshold = np.quantile(label_values, [1 / 3, 2 / 3])
        low_mask = label_values <= low_threshold
        mid_mask = (label_values > low_threshold) & (label_values <= high_threshold)
        high_mask = label_values > high_threshold
        for proxy_index, proxy_name in enumerate(proxy_names):
            if not proxy_name.startswith(f"{target_prefix}_band_"):
                continue
            values = proxy_matrix[:, proxy_index]
            low_mean = float(values[low_mask].mean())
            mid_mean = float(values[mid_mask].mean())
            high_mean = float(values[high_mask].mean())
            step_1 = mid_mean >= low_mean
            step_2 = high_mean >= mid_mean
            rows.append(
                {
                    "target": target,
                    "band_name": proxy_band_name(proxy_name),
                    "proxy": proxy_name,
                    "proxy_role": proxy_role(proxy_name),
                    "low_label_proxy_mean": low_mean,
                    "mid_label_proxy_mean": mid_mean,
                    "high_label_proxy_mean": high_mean,
                    "monotonic_increasing": step_1 and step_2,
                    "monotonic_score": (int(step_1) + int(step_2)) / 2.0,
                    "low_label_max": float(low_threshold),
                    "high_label_min": float(high_threshold),
                }
            )
    return rows


def find_geometry_label_ambiguity_pairs(
    sample_ids: list[str],
    proxy_matrix: np.ndarray,
    target_matrix: np.ndarray,
    proxy_names: list[str],
    ambiguity_pairs_per_target: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(MEASUREMENT_BAND_TARGETS):
        target_prefix = target.removesuffix("_cm")
        feature_indices = [index for index, name in enumerate(proxy_names) if name.startswith(f"{target_prefix}_band_")]
        if not feature_indices:
            continue
        target_geometry = standardize_matrix(proxy_matrix[:, feature_indices])
        distances = pairwise_euclidean(target_geometry)
        np.fill_diagonal(distances, np.inf)
        nearest = np.argsort(distances, axis=1)[:, : min(8, len(sample_ids) - 1)]
        seen: set[tuple[str, str]] = set()
        candidates = []
        label_values = target_matrix[:, target_index]
        standardized_labels = standardize_matrix(label_values.reshape(-1, 1)).reshape(-1)
        for left_index, neighbors in enumerate(nearest):
            for right_index in neighbors:
                pair = tuple(sorted((sample_ids[left_index], sample_ids[int(right_index)])))
                if pair in seen:
                    continue
                seen.add(pair)
                label_diff = abs(float(label_values[left_index] - label_values[int(right_index)]))
                standardized_label_diff = abs(float(standardized_labels[left_index] - standardized_labels[int(right_index)]))
                geometry_distance = float(distances[left_index, int(right_index)])
                candidates.append(
                    {
                        "target": target,
                        "sample_id_a": pair[0],
                        "sample_id_b": pair[1],
                        "geometry_distance": geometry_distance,
                        "label_diff": label_diff,
                        "standardized_label_diff": standardized_label_diff,
                        "ambiguity_score": standardized_label_diff / max(geometry_distance, 1e-9),
                    }
                )
        rows.extend(sorted(candidates, key=lambda row: (row["ambiguity_score"], row["label_diff"]), reverse=True)[:ambiguity_pairs_per_target])
    return rows


def build_deformation_summary(
    correlation_rows: list[dict[str, Any]],
    monotonicity_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    dataset: str | Path,
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "band_feature_version": BAND_FEATURE_VERSION,
        "sample_count": len(samples),
        "targets": MEASUREMENT_BAND_TARGETS,
        "target_summaries": {},
        "warnings": warnings,
    }
    for target in MEASUREMENT_BAND_TARGETS:
        target_rows = [row for row in correlation_rows if row["target"] == target]
        monotonic_rows = [row for row in monotonicity_rows if row["target"] == target]
        best = max(target_rows, key=lambda row: row["abs_correlation"])
        best_monotonic = next((row for row in monotonic_rows if row["proxy"] == best["proxy"]), None)
        role_best = {}
        for role in ("front", "side", "combined", "ellipse", "local_area", "contour"):
            rows = [row for row in target_rows if row["proxy_role"] == role]
            if rows:
                role_best[role] = max(rows, key=lambda row: row["abs_correlation"])
        target_values = np.asarray([sample["measurements"][target] for sample in samples], dtype=np.float64)
        summary["target_summaries"][target] = {
            "best_proxy": best,
            "best_proxy_monotonic": best_monotonic,
            "best_by_role": role_best,
            "label_mean": float(target_values.mean()),
            "label_std": float(target_values.std()),
            "label_range": float(target_values.max() - target_values.min()),
            "alignment_grade": alignment_grade(best, best_monotonic),
            "dominant_geometry_channel": dominant_geometry_channel(role_best),
        }
    return summary


def alignment_grade(best: dict[str, Any], monotonic: dict[str, Any] | None) -> str:
    corr = float(best["abs_correlation"])
    monotonic_ok = bool(monotonic and monotonic["monotonic_increasing"])
    if corr >= 0.75 and monotonic_ok:
        return "good_alignment"
    if corr >= 0.55:
        return "partial_alignment"
    return "weak_alignment"


def dominant_geometry_channel(role_best: dict[str, dict[str, Any]]) -> str:
    front = float(role_best.get("front", {}).get("abs_correlation", 0.0))
    side = float(role_best.get("side", {}).get("abs_correlation", 0.0))
    combined = float(role_best.get("combined", {}).get("abs_correlation", 0.0))
    if combined >= max(front, side):
        return "front_and_side_combined"
    if side > front:
        return "side_depth_dominant"
    return "front_width_dominant"


def write_alignment_contact_sheets(samples: list[dict[str, Any]], ambiguity_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    for target in MEASUREMENT_BAND_TARGETS:
        ordered = sorted(samples, key=lambda sample: sample["measurements"][target])
        count = 5
        midpoint = len(ordered) // 2
        selected = [*ordered[:count], *ordered[midpoint - count // 2 : midpoint - count // 2 + count], *ordered[-count:]]
        warnings.extend(write_contact_sheet(output_dir / f"{target}_low_mid_high.png", selected, f"{target}: low / mid / high"))

        target_ambiguities = [row for row in ambiguity_rows if row["target"] == target][:3]
        ambiguous_samples = []
        for row in target_ambiguities:
            for sample_id in (row["sample_id_a"], row["sample_id_b"]):
                if sample_id in samples_by_id:
                    ambiguous_samples.append(samples_by_id[sample_id])
        if ambiguous_samples:
            warnings.extend(write_contact_sheet(output_dir / f"{target}_ambiguous_pairs.png", ambiguous_samples, f"{target}: similar geometry / different labels"))
        same_height_samples = same_height_different_target_samples(samples, target)
        if same_height_samples:
            warnings.extend(write_contact_sheet(output_dir / f"{target}_same_height_different_label.png", same_height_samples, f"{target}: same height / different labels"))
    return warnings


def same_height_different_target_samples(
    samples: list[dict[str, Any]],
    target: str,
    height_tolerance_cm: float = 2.0,
    max_pairs: int = 4,
) -> list[dict[str, Any]]:
    candidates = []
    for left_index, left in enumerate(samples):
        for right in samples[left_index + 1 :]:
            height_diff = abs(left["measurements"].get("height_cm", 0.0) - right["measurements"].get("height_cm", 0.0))
            if height_diff > height_tolerance_cm:
                continue
            target_diff = abs(left["measurements"][target] - right["measurements"][target])
            candidates.append((target_diff, left["sample_id"], right["sample_id"], left, right))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _target_diff, _left_id, _right_id, left, right in sorted(candidates, reverse=True)[:max_pairs]:
        for sample in (left, right):
            sample_id = sample["sample_id"]
            if sample_id not in seen:
                seen.add(sample_id)
                selected.append(sample)
    return selected


def proxy_band_name(proxy_name: str) -> str:
    return "_".join(proxy_name.split("_")[:4])


def proxy_role(proxy_name: str) -> str:
    if "ellipse_circumference" in proxy_name:
        return "ellipse"
    if "width_depth_product" in proxy_name or "front_side" in proxy_name:
        return "combined"
    if "local_area" in proxy_name:
        return "local_area"
    if "contour_slope" in proxy_name:
        return "contour"
    if "_front_" in proxy_name:
        return "front"
    if "_side_" in proxy_name:
        return "side"
    return "metadata"


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    return (matrix - matrix.mean(axis=0)) / stds


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(matrix * matrix, axis=1, keepdims=True)
    squared = squared_norms + squared_norms.T - 2.0 * (matrix @ matrix.T)
    return np.sqrt(np.maximum(squared, 0.0))


def write_outputs(
    output_path: Path,
    summary: dict[str, Any],
    correlation_rows: list[dict[str, Any]],
    monotonicity_rows: list[dict[str, Any]],
    ambiguity_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "correlations_json": output_path / CORRELATIONS_JSON,
        "correlations_csv": output_path / CORRELATIONS_CSV,
        "correlations_md": output_path / CORRELATIONS_MD,
        "monotonicity_csv": output_path / MONOTONICITY_CSV,
        "ambiguity_csv": output_path / AMBIGUITY_CSV,
        "summary_md": output_path / SUMMARY_MD,
    }
    write_json(paths["correlations_json"], summary)
    write_csv(paths["correlations_csv"], correlation_rows, ["target", "band_name", "proxy", "proxy_role", "correlation", "abs_correlation"])
    write_csv(paths["monotonicity_csv"], monotonicity_rows, ["target", "band_name", "proxy", "proxy_role", "low_label_proxy_mean", "mid_label_proxy_mean", "high_label_proxy_mean", "monotonic_increasing", "monotonic_score", "low_label_max", "high_label_min"])
    write_csv(paths["ambiguity_csv"], ambiguity_rows, ["target", "sample_id_a", "sample_id_b", "geometry_distance", "label_diff", "standardized_label_diff", "ambiguity_score"])
    paths["correlations_md"].write_text(format_correlations_markdown(summary), encoding="utf-8")
    paths["summary_md"].write_text(format_summary_markdown(summary), encoding="utf-8")
    return paths


def format_correlations_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Label Geometry Correlations",
        "",
        "| Target | Best Proxy | Role | Abs Corr | Monotonic | Grade | Channel |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for target, target_summary in summary["target_summaries"].items():
        best = target_summary["best_proxy"]
        monotonic = target_summary.get("best_proxy_monotonic") or {}
        lines.append(
            f"| {target} | {best['proxy']} | {best['proxy_role']} | {float(best['abs_correlation']):.4f} | "
            f"{monotonic.get('monotonic_increasing', '')} | {target_summary['alignment_grade']} | {target_summary['dominant_geometry_channel']} |"
        )
    return "\n".join(lines) + "\n"


def format_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3Y Label Geometry Alignment",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Band feature version: `{summary['band_feature_version']}`",
        f"Samples: {summary['sample_count']}",
        "",
        "## Target Summary",
        "",
        "| Target | Grade | Channel | Best Abs Corr | Label Range | Recommendation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for target, target_summary in summary["target_summaries"].items():
        best = target_summary["best_proxy"]
        lines.append(
            f"| {target} | {target_summary['alignment_grade']} | {target_summary['dominant_geometry_channel']} | "
            f"{float(best['abs_correlation']):.4f} | {float(target_summary['label_range']):.2f} | {recommendation_for_target(target_summary)} |"
        )
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def recommendation_for_target(target_summary: dict[str, Any]) -> str:
    if target_summary["alignment_grade"] == "good_alignment":
        return "Label is visibly expressed; improve model/features or reduce noise."
    if target_summary["alignment_grade"] == "partial_alignment":
        return "Some geometry signal exists; inspect deformation locality and cross-target coupling."
    return "Weak geometry signal; consider generator/deformation changes or geometry-derived labels."


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether labels are visibly expressed in rendered geometry.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ambiguity-pairs-per-target", type=int, default=12)
    args = parser.parse_args(argv)

    result = audit_label_geometry_alignment(args.dataset, args.output, ambiguity_pairs_per_target=args.ambiguity_pairs_per_target)
    print(f"Correlations: {result['correlations_json']}")
    print(f"Monotonicity: {result['monotonicity_csv']}")
    print(f"Summary: {result['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
