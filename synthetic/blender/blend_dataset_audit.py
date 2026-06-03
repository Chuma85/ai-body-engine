from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageStat

from synthetic.blender.blend_dataset import CAMERA_VIEWS, LEGACY_BLEND_LABEL_COLUMNS


AUDIT_REPORT_JSON = "audit_report.json"
AUDIT_SUMMARY_MD = "audit_summary.md"
CONTACT_SHEET_PNG = "sample_contact_sheet.png"
LABEL_DISTRIBUTION_CSV = "label_distribution_summary.csv"
FLAGGED_SAMPLES_CSV = "flagged_samples.csv"
IMPORTANT_MEASUREMENT_COLUMNS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
]
SAFE_MEASUREMENT_RANGES = {
    "height_cm": (140.0, 220.0),
    "chest_cm": (60.0, 160.0),
    "waist_cm": (45.0, 160.0),
    "hip_cm": (60.0, 170.0),
    "shoulder_cm": (25.0, 80.0),
    "inseam_cm": (50.0, 110.0),
}
REQUIRED_METADATA_FIELDS = [
    "generator_version",
    "source_blend_file",
    "camera_set",
    "sample_count",
    "seed",
    "synthetic_labels",
    "real_world_validated",
    "variation_source",
    "shape_key_count",
]
MIN_VIEW_DIFFERENCE_SCORE = 0.02


def audit_blend_dataset(
    dataset: str | Path,
    out: str | Path,
    expected_samples: int | None = None,
    max_contact_sheet_samples: int = 12,
    strict: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset)
    output_dir = Path(out)
    labels_path = dataset_root / "labels.csv"
    metadata_path = dataset_root / "metadata.json"
    images_dir = dataset_root / "images"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    rows, label_columns = read_labels(labels_path)
    metadata = read_metadata(metadata_path)
    warnings: list[str] = []
    errors: list[str] = []
    strict_failures: list[str] = []
    flagged_samples: list[dict[str, str]] = []

    schema_result = validate_label_schema(label_columns)
    warnings.extend(schema_result["warnings"])
    errors.extend(schema_result["errors"])
    metadata_result = validate_metadata(metadata)
    warnings.extend(metadata_result["warnings"])
    errors.extend(metadata_result["errors"])

    if expected_samples is not None and len(rows) != expected_samples:
        errors.append(f"labels.csv row count {len(rows)} does not match expected sample count {expected_samples}.")
    if metadata.get("sample_count") is not None and int(metadata.get("sample_count", -1)) != len(rows):
        warnings.append(f"metadata sample_count {metadata.get('sample_count')} does not match labels row count {len(rows)}.")

    image_result = audit_images(dataset_root, rows)
    warnings.extend(image_result["warnings"])
    errors.extend(image_result["errors"])
    flagged_samples.extend(image_result["flagged_samples"])
    if strict:
        strict_failures.extend(image_result["strict_failures"])

    label_result = audit_labels(rows)
    warnings.extend(label_result["warnings"])
    errors.extend(label_result["errors"])
    if strict:
        strict_failures.extend(label_result["strict_failures"])

    shape_key_result = audit_shape_key_metadata(metadata)
    warnings.extend(shape_key_result["warnings"])
    strict_failures.extend(shape_key_result["strict_failures"] if strict else [])

    usable = not errors
    passed = usable and not strict_failures
    output_dir.mkdir(parents=True, exist_ok=True)
    write_distribution_summary(output_dir / LABEL_DISTRIBUTION_CSV, label_result["measurement_stats"])
    write_flagged_samples(output_dir / FLAGGED_SAMPLES_CSV, flagged_samples)
    contact_sheet_warning = write_contact_sheet(
        dataset_root=dataset_root,
        rows=rows,
        output_path=output_dir / CONTACT_SHEET_PNG,
        max_samples=max_contact_sheet_samples,
    )
    if contact_sheet_warning:
        warnings.append(contact_sheet_warning)

    report = {
        "dataset": str(dataset_root),
        "output_dir": str(output_dir),
        "strict": strict,
        "passed": passed,
        "usable": usable,
        "row_count": len(rows),
        "expected_samples": expected_samples,
        "metadata": {
            "variation_source": metadata.get("variation_source"),
            "shape_key_count": metadata.get("shape_key_count"),
            "synthetic_labels": metadata.get("synthetic_labels"),
            "real_world_validated": metadata.get("real_world_validated"),
        },
        "label_schema": schema_result,
        "metadata_schema": metadata_result,
        "image_audit": image_result["summary"],
        "view_sanity": image_result["view_sanity"],
        "label_audit": label_result,
        "shape_key_audit": shape_key_result,
        "flagged_sample_count": len(flagged_samples),
        "warnings": warnings,
        "errors": errors,
        "strict_failures": strict_failures,
        "outputs": {
            "audit_report_json": str(output_dir / AUDIT_REPORT_JSON),
            "audit_summary_md": str(output_dir / AUDIT_SUMMARY_MD),
            "sample_contact_sheet_png": str(output_dir / CONTACT_SHEET_PNG),
            "label_distribution_summary_csv": str(output_dir / LABEL_DISTRIBUTION_CSV),
            "flagged_samples_csv": str(output_dir / FLAGGED_SAMPLES_CSV),
        },
    }
    (output_dir / AUDIT_REPORT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / AUDIT_SUMMARY_MD).write_text(format_summary(report), encoding="utf-8")
    return report


def read_labels(labels_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def read_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def validate_label_schema(columns: list[str]) -> dict[str, Any]:
    missing_columns = [column for column in LEGACY_BLEND_LABEL_COLUMNS if column not in columns]
    errors = []
    warnings = []
    if missing_columns:
        errors.append(f"labels.csv missing required columns: {', '.join(missing_columns)}")
        warnings.append("labels.csv schema mismatch blocks model training until fixed.")
    return {
        "valid": not missing_columns,
        "missing_columns": missing_columns,
        "warnings": warnings,
        "errors": errors,
    }


def validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    errors = []
    warnings = []
    if missing_fields:
        errors.append(f"metadata.json missing required fields: {', '.join(missing_fields)}")
        warnings.append("metadata.json schema mismatch blocks model training until fixed.")
    return {
        "valid": not missing_fields,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "errors": errors,
    }


def audit_images(dataset_root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    strict_failures: list[str] = []
    flagged_samples: list[dict[str, str]] = []
    dimensions: set[tuple[int, int]] = set()
    image_summaries: list[dict[str, Any]] = []
    view_scores: list[dict[str, Any]] = []

    for row in rows:
        sample_id = row.get("sample_id", "")
        sample_images: dict[str, Image.Image] = {}
        for view in CAMERA_VIEWS:
            column = f"{view}_image"
            relative_path = row.get(column, "")
            if not relative_path:
                message = f"{sample_id}: missing {column} value"
                errors.append(message)
                strict_failures.append(message)
                flagged_samples.append(flag(sample_id, view, "missing_image_path", message))
                continue
            image_path = dataset_root / relative_path
            if not image_path.exists():
                message = f"{sample_id}: missing image file {relative_path}"
                errors.append(message)
                strict_failures.append(message)
                flagged_samples.append(flag(sample_id, view, "missing_image_file", message))
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except OSError as exc:
                message = f"{sample_id}: could not open {relative_path}: {exc}"
                errors.append(message)
                strict_failures.append(message)
                flagged_samples.append(flag(sample_id, view, "unreadable_png", message))
                continue
            sample_images[view] = image
            metrics = image_metrics(image)
            dimensions.add(metrics["dimensions"])
            image_summaries.append({"sample_id": sample_id, "view": view, "path": relative_path, **metrics})
            for issue in image_quality_issues(metrics):
                warnings.append(f"{sample_id} {view}: {issue}")
                flagged_samples.append(flag(sample_id, view, issue.split(":")[0], issue))
                if issue.startswith(("blank_render", "near_blank_render")):
                    strict_failures.append(f"{sample_id} {view}: {issue}")

        if set(sample_images) == set(CAMERA_VIEWS):
            for view_a, view_b in (("front", "side"), ("side", "back"), ("back", "front")):
                score = view_difference_score(sample_images[view_a], sample_images[view_b])
                passed = score >= MIN_VIEW_DIFFERENCE_SCORE
                view_scores.append(
                    {
                        "sample_id": sample_id,
                        "view_pair": f"{view_a}_{view_b}",
                        "difference_score": score,
                        "passed": passed,
                    }
                )
                if not passed:
                    message = (
                        f"{sample_id}: {view_a}/{view_b} views are near-identical "
                        f"(difference_score={score:.4f})"
                    )
                    warnings.append(message)
                    strict_failures.append(message)
                    flagged_samples.append(flag(sample_id, f"{view_a}_{view_b}", "near_identical_views", message))

    if len(dimensions) > 1:
        warnings.append(f"Image dimensions are inconsistent: {sorted(dimensions)}")
    view_sanity_passed = bool(view_scores) and all(score["passed"] for score in view_scores)
    return {
        "summary": {
            "image_count": len(image_summaries),
            "dimensions": [list(dimensions_item) for dimensions_item in sorted(dimensions)],
            "dimensions_consistent": len(dimensions) <= 1,
            "quality_metrics": image_summaries,
        },
        "view_sanity": {
            "passed": view_sanity_passed,
            "minimum_difference_score": MIN_VIEW_DIFFERENCE_SCORE,
            "scores": view_scores,
        },
        "warnings": warnings,
        "errors": errors,
        "strict_failures": strict_failures,
        "flagged_samples": flagged_samples,
    }


def image_metrics(image: Image.Image) -> dict[str, Any]:
    stat = ImageStat.Stat(image)
    grayscale = image.convert("L")
    gray_stat = ImageStat.Stat(grayscale)
    background = corner_background_value(grayscale)
    foreground_count = 0
    pixels = grayscale.load()
    width, height = grayscale.size
    for y in range(height):
        for x in range(width):
            if abs(pixels[x, y] - background) > 18:
                foreground_count += 1
    foreground_ratio = foreground_count / max(width * height, 1)
    return {
        "dimensions": image.size,
        "mean_luma": float(gray_stat.mean[0]),
        "std_luma": float(gray_stat.stddev[0]),
        "channel_extrema": stat.extrema,
        "foreground_ratio": foreground_ratio,
    }


def corner_background_value(image: Image.Image) -> int:
    width, height = image.size
    corner_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    pixels = image.load()
    return int(sum(pixels[x, y] for x, y in corner_points) / len(corner_points))


def image_quality_issues(metrics: dict[str, Any]) -> list[str]:
    issues = []
    if float(metrics["std_luma"]) < 1.0:
        issues.append("blank_render: luma standard deviation is below 1.0")
    elif float(metrics["std_luma"]) < 4.0:
        issues.append("near_blank_render: luma standard deviation is below 4.0")
    if float(metrics["mean_luma"]) < 20.0:
        issues.append("extremely_dark_image: mean luma is below 20")
    if float(metrics["mean_luma"]) > 245.0:
        issues.append("extremely_light_image: mean luma is above 245")
    if float(metrics["foreground_ratio"]) < 0.01:
        issues.append("missing_body_silhouette: foreground ratio is below 1%")
    return issues


def view_difference_score(image_a: Image.Image, image_b: Image.Image) -> float:
    size = (96, 96)
    a = image_a.convert("L").resize(size)
    b = image_b.convert("L").resize(size)
    diff = ImageChops.difference(a, b)
    stat = ImageStat.Stat(diff)
    return float(stat.mean[0]) / 255.0


def audit_labels(rows: list[dict[str, str]]) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    strict_failures: list[str] = []
    numeric_values = {column: [] for column in IMPORTANT_MEASUREMENT_COLUMNS}
    non_numeric: list[dict[str, str]] = []
    missing_numeric_columns = [column for column in IMPORTANT_MEASUREMENT_COLUMNS if rows and column not in rows[0]]
    if missing_numeric_columns:
        errors.append(f"Missing numeric measurement columns: {', '.join(missing_numeric_columns)}")

    for row in rows:
        sample_id = row.get("sample_id", "")
        for column in IMPORTANT_MEASUREMENT_COLUMNS:
            value = row.get(column, "")
            try:
                numeric_values[column].append(float(value))
            except (TypeError, ValueError):
                non_numeric.append({"sample_id": sample_id, "column": column, "value": str(value)})

    if non_numeric:
        errors.append(f"Non-numeric measurement values found: {len(non_numeric)}")

    measurement_stats = {
        column: numeric_summary(values)
        for column, values in numeric_values.items()
        if values
    }
    identical_columns = [
        column
        for column, stats in measurement_stats.items()
        if float(stats["std"]) == 0.0 or float(stats["min"]) == float(stats["max"])
    ]
    if identical_columns:
        message = "Measurement columns have no variation: " + ", ".join(identical_columns)
        warnings.append(message)
        strict_failures.append(message)

    unrealistic_values = []
    for column, values in numeric_values.items():
        lower, upper = SAFE_MEASUREMENT_RANGES[column]
        for value in values:
            if value < lower or value > upper:
                unrealistic_values.append({"column": column, "value": value, "safe_min": lower, "safe_max": upper})
    if unrealistic_values:
        warnings.append(f"Measurement values outside safe ranges: {len(unrealistic_values)}")

    return {
        "measurement_columns": IMPORTANT_MEASUREMENT_COLUMNS,
        "measurement_stats": measurement_stats,
        "identical_columns": identical_columns,
        "non_numeric_values": non_numeric,
        "unrealistic_values": unrealistic_values,
        "variation_exists": bool(measurement_stats) and not identical_columns,
        "warnings": warnings,
        "errors": errors,
        "strict_failures": strict_failures,
    }


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    value_min = min(values)
    value_max = max(values)
    value_mean = sum(values) / count
    variance = sum((value - value_mean) ** 2 for value in values) / count
    return {
        "count": count,
        "min": value_min,
        "max": value_max,
        "mean": value_mean,
        "std": math.sqrt(variance),
    }


def audit_shape_key_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    strict_failures = []
    variation_source = metadata.get("variation_source")
    shape_key_count = metadata.get("shape_key_count")
    variation_active = variation_source == "shape_keys_safe_range" and int(shape_key_count or 0) > 0
    if variation_source == "static_blend_mesh":
        warnings.append("variation_source=static_blend_mesh: dataset is not suitable for serious training variation yet.")
    elif variation_source == "shape_keys_safe_range" and variation_active:
        pass
    else:
        warnings.append(f"Unexpected variation_source={variation_source!r}; inspect blend metadata before training.")
    if not variation_active and variation_source != "static_blend_mesh":
        strict_failures.append("Shape-key variation metadata is missing or inconsistent.")
    return {
        "variation_source": variation_source,
        "shape_key_count": shape_key_count,
        "variation_active": variation_active,
        "warnings": warnings,
        "strict_failures": strict_failures,
    }


def write_distribution_summary(path: Path, measurement_stats: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["column", "count", "min", "max", "mean", "std"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for column, stats in measurement_stats.items():
            writer.writerow({"column": column, **stats})


def write_flagged_samples(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["sample_id", "view", "issue", "detail"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(dataset_root: Path, rows: list[dict[str, str]], output_path: Path, max_samples: int) -> str | None:
    selected = rows[: max(0, max_samples)]
    if not selected:
        return "No samples available for contact sheet."
    thumb_width = 160
    thumb_height = 224
    label_height = 24
    row_height = thumb_height + label_height
    sheet = Image.new("RGB", (thumb_width * 3, row_height * len(selected)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(selected):
        sample_id = row.get("sample_id", "")
        y_offset = row_index * row_height
        draw.text((4, y_offset + 4), sample_id, fill=(0, 0, 0))
        for view_index, view in enumerate(CAMERA_VIEWS):
            image_path = dataset_root / row.get(f"{view}_image", "")
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_width, thumb_height - label_height))
            x = view_index * thumb_width + (thumb_width - image.width) // 2
            y = y_offset + label_height + (thumb_height - label_height - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((view_index * thumb_width + 4, y_offset + label_height - 14), view, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return None


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3H Blend Dataset Audit Summary",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Passed: `{report['passed']}`",
        f"- Usable: `{report['usable']}`",
        f"- Strict mode: `{report['strict']}`",
        f"- Label rows: `{report['row_count']}`",
        f"- Variation source: `{report['metadata'].get('variation_source')}`",
        f"- Shape key count: `{report['metadata'].get('shape_key_count')}`",
        f"- View sanity passed: `{report['view_sanity']['passed']}`",
        f"- Label variation exists: `{report['label_audit']['variation_exists']}`",
        f"- Flagged samples: `{report['flagged_sample_count']}`",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"]) if report["warnings"] else lines.append("- None")
    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in report["errors"]) if report["errors"] else lines.append("- None")
    lines.extend(["", "## Strict Failures"])
    lines.extend(f"- {failure}" for failure in report["strict_failures"]) if report["strict_failures"] else lines.append("- None")
    return "\n".join(lines) + "\n"


def flag(sample_id: str, view: str, issue: str, detail: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "view": view,
        "issue": issue,
        "detail": detail,
    }
