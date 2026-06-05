from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image, ImageChops, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_phase_3h_d_blend_dataset_scale import DEFAULT_BLEND_FILE, discover_blender_executable
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    BODY_FACTOR_COLUMNS,
    CAMERA_VIEWS,
    LABEL_GENERATION_MODE,
    PHASE_3H_J_LABEL_FORMULA_VERSION,
    validate_shape_key_coupled_rows,
)
from training.features.image_silhouette_features import extract_image_features

DEFAULT_DATASET = "data/synthetic/phase_3h_j_mobile_realism_1000"
DEFAULT_AUDIT_OUT = "artifacts/phase_3h_j_mobile_realism_1000_audit"
DEFAULT_CORRELATION_OUT = "artifacts/phase_3h_j_mobile_realism_label_visual_correlation"
DEFAULT_TRAINING_OUT = "artifacts/phase_3h_j_mobile_realism_blend_baseline"
DEFAULT_BATCH_OUT = "artifacts/phase_3h_j_mobile_realism_generation_batches"
DEFAULT_SMOKE_DATASET = "data/synthetic/phase_3h_j_mobile_realism_smoke"
DEFAULT_SMOKE_BATCH_OUT = "artifacts/phase_3h_j_mobile_realism_smoke_generation_batches"
PHASE_3H_I_METRICS = "artifacts/phase_3h_i_blend_baseline/metrics.json"
DEFAULT_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 250
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SHAPE_KEY_RANGE = 0.24
DEFAULT_LABEL_MEASUREMENT_SCALE = 2.0
DEFAULT_SAFE_FRAMING_SCALE = 1.34
DEFAULT_MOBILE_REALISM = True
DEFAULT_DISTANCE_JITTER = 0.015
DEFAULT_CAMERA_HEIGHT_JITTER = 0.015
DEFAULT_BODY_ROTATION_JITTER = 2.0
DEFAULT_LIGHTING_JITTER = 0.08
DEFAULT_BACKGROUND_JITTER = 0.04
DEFAULT_PHONE_FRAMING_JITTER = 0.006
DEFAULT_MIN_ABS_CORRELATION = 0.25
MIN_VIEW_DIFFERENCE_SCORE = 0.018
TARGET_COLUMNS = [
    "height_cm",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
]
TRACEABILITY_COLUMNS = [
    "label_generation_mode",
    "synthetic_labels",
    "real_world_validated",
    "shape_key_values_json",
    *BODY_FACTOR_COLUMNS,
]
ARCHIVED_DATASET_MARKERS = ("_archived_old_mannequin", "archived", "old_mannequin")


def verify_coupled_1000(
    *,
    dataset: str = DEFAULT_DATASET,
    audit_out: str = DEFAULT_AUDIT_OUT,
    correlation_out: str = DEFAULT_CORRELATION_OUT,
    training_out: str = DEFAULT_TRAINING_OUT,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    blend_file: str = DEFAULT_BLEND_FILE,
    blender_executable: str | None = None,
    resume: bool = False,
    force: bool = False,
    start_index: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_out: str = DEFAULT_BATCH_OUT,
    run_benchmark: bool = True,
    smoke: bool = False,
    no_render: bool = False,
    mobile_realism: bool = DEFAULT_MOBILE_REALISM,
    distance_jitter: float = DEFAULT_DISTANCE_JITTER,
    camera_height_jitter: float = DEFAULT_CAMERA_HEIGHT_JITTER,
    body_rotation_jitter: float = DEFAULT_BODY_ROTATION_JITTER,
    lighting_jitter: float = DEFAULT_LIGHTING_JITTER,
    background_jitter: float = DEFAULT_BACKGROUND_JITTER,
    phone_framing_jitter: float = DEFAULT_PHONE_FRAMING_JITTER,
    phase_3h_i_metrics: str = PHASE_3H_I_METRICS,
) -> dict[str, Any]:
    ensure_not_archived_dataset(dataset)
    if no_render:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(
                "--no-render/--reuse-existing requires the final merged Phase 3H-J dataset to already exist: "
                f"{dataset_path}. Run a clean full or resumable generation first."
            )
        generation_summary = {
            "batch_size": batch_size,
            "batch_count": 0,
            "batch_out": batch_out,
            "commands": ["no-render: reused existing dataset; Blender generation skipped"],
        }
    else:
        blender = discover_blender_executable(blender_executable)
        if blender is None:
            raise RuntimeError(
                "Blender executable was not found. Install Blender, add blender to PATH, or pass "
                "--blender-executable with the full blender.exe path."
            )

        generation_summary = generate_batched_dataset(
            blender_executable=blender,
            dataset=dataset,
            samples=samples,
            seed=seed,
            blend_file=blend_file,
            resume=resume,
            force=force,
            start_index=start_index,
            batch_size=batch_size,
            batch_out=batch_out,
            mobile_realism=mobile_realism,
            distance_jitter=distance_jitter,
            camera_height_jitter=camera_height_jitter,
            body_rotation_jitter=body_rotation_jitter,
            lighting_jitter=lighting_jitter,
            background_jitter=background_jitter,
            phone_framing_jitter=phone_framing_jitter,
        )
    dataset_summary = validate_phase_3h_j_dataset(dataset, expected_samples=samples)
    if smoke or not run_benchmark:
        return {
            "dataset": dataset,
            "sample_count": dataset_summary["sample_count"],
            "image_count": dataset_summary["image_count"],
            "label_generation_mode": dataset_summary["label_generation_mode"],
            "label_formula_version": dataset_summary["label_formula_version"],
            "synthetic_labels": dataset_summary["synthetic_labels"],
            "real_world_validated": dataset_summary["real_world_validated"],
            "variation_source": dataset_summary["variation_source"],
            "shape_key_count": dataset_summary["shape_key_count"],
            "mobile_realism": dataset_summary["mobile_realism"],
            "mobile_realism_settings": dataset_summary["mobile_realism_settings"],
            "clipping": dataset_summary["clipping"],
            "view_sanity": dataset_summary["view_sanity"],
            "label_variation": dataset_summary["label_variation"],
            "audit": None,
            "correlation_by_target": {},
            "weak_targets_below_threshold": [],
            "label_variation_warnings": dataset_summary["label_variation"]["low_label_variation_warnings"],
            "best_model": None,
            "overall_mean_mae": None,
            "mae_by_target": {},
            "model_ranking": [],
            "train_sample_count": None,
            "test_sample_count": None,
            "comparison_to_phase_3h_i": {"available": False, "reason": "Benchmark skipped."},
            "commands": {
                "generation": generation_summary["commands"],
                "audit": "skipped",
                "correlation": "skipped",
                "training": "skipped",
            },
            "smoke": smoke,
        }

    audit_command = build_audit_command(dataset=dataset, audit_out=audit_out, samples=samples)
    subprocess.run(audit_command, cwd=REPO_ROOT, check=True)
    audit_summary = read_audit_summary(Path(audit_out) / "audit_report.json")

    correlation_command = build_correlation_command(dataset=dataset, out=correlation_out)
    subprocess.run(correlation_command, cwd=REPO_ROOT, check=True)
    correlation_report = json.loads((Path(correlation_out) / "correlation_report.json").read_text(encoding="utf-8"))

    training_command = build_training_command(
        dataset=dataset,
        out=training_out,
        seed=seed,
        test_size=test_size,
        audit_report=str(Path(audit_out) / "audit_report.json"),
    )
    subprocess.run(training_command, cwd=REPO_ROOT, check=True)
    training_metrics = json.loads((Path(training_out) / "metrics.json").read_text(encoding="utf-8"))

    comparison = compare_to_phase_3h_i(training_metrics, phase_3h_i_metrics)
    label_variation_warnings = [
        row for row in correlation_report.get("flagged_targets", []) if row.get("category") == "low_label_variation"
    ]

    return {
        "dataset": dataset,
        "audit_out": audit_out,
        "correlation_out": correlation_out,
        "training_out": training_out,
        "sample_count": dataset_summary["sample_count"],
        "image_count": dataset_summary["image_count"],
        "label_generation_mode": dataset_summary["label_generation_mode"],
        "label_formula_version": dataset_summary["label_formula_version"],
        "synthetic_labels": dataset_summary["synthetic_labels"],
        "real_world_validated": dataset_summary["real_world_validated"],
        "variation_source": dataset_summary["variation_source"],
        "shape_key_count": dataset_summary["shape_key_count"],
        "mobile_realism": dataset_summary["mobile_realism"],
        "mobile_realism_settings": dataset_summary["mobile_realism_settings"],
        "clipping": dataset_summary["clipping"],
        "view_sanity": dataset_summary["view_sanity"],
        "label_variation": dataset_summary["label_variation"],
        "audit": audit_summary,
        "correlation_by_target": correlation_report["strongest_visual_correlation_by_target"],
        "weak_targets_below_threshold": correlation_report["weakly_learnable_targets"],
        "label_variation_warnings": label_variation_warnings,
        "best_model": training_metrics["best_model"],
        "overall_mean_mae": training_metrics["overall_mean_mae"],
        "mae_by_target": training_metrics["mae_by_target"],
        "model_ranking": training_metrics.get("model_ranking", []),
        "train_sample_count": training_metrics.get("train_sample_count"),
        "test_sample_count": training_metrics.get("test_sample_count"),
        "comparison_to_phase_3h_i": comparison,
        "commands": {
            "generation": generation_summary["commands"],
            "audit": " ".join(audit_command),
            "correlation": " ".join(correlation_command),
            "training": " ".join(training_command),
        },
    }


def generate_batched_dataset(
    *,
    blender_executable: str,
    dataset: str,
    samples: int,
    seed: int,
    blend_file: str,
    resume: bool,
    force: bool,
    start_index: int,
    batch_size: int,
    batch_out: str,
    mobile_realism: bool,
    distance_jitter: float,
    camera_height_jitter: float,
    body_rotation_jitter: float,
    lighting_jitter: float,
    background_jitter: float,
    phone_framing_jitter: float,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    dataset_path = Path(dataset)
    batch_root = Path(batch_out)
    ensure_not_archived_dataset(dataset_path)
    if start_index <= 0:
        raise ValueError("start_index must be greater than 0")
    if dataset_path.exists() and any(dataset_path.iterdir()):
        if not force and not resume:
            raise FileExistsError(f"Output directory already exists and is not empty: {dataset_path}")
        shutil.rmtree(dataset_path)
    if batch_root.exists():
        if not resume and not force:
            raise FileExistsError(f"Batch directory already exists: {batch_root}. Pass --resume to reuse completed chunks or --force to rebuild.")
        if force and not resume:
            shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    chunk_paths: list[Path] = []
    min_start_index = start_index
    for chunk_start in range(1, samples + 1, batch_size):
        chunk_count = min(batch_size, samples - chunk_start + 1)
        end_index = chunk_start + chunk_count - 1
        chunk_path = batch_root / f"chunk_{chunk_start:06d}_{end_index:06d}"
        status = inspect_chunk(chunk_path, expected_start=chunk_start, expected_count=chunk_count)
        if status["complete"]:
            if force and chunk_start >= min_start_index:
                print(f"Phase 3H-J batch {chunk_start}-{end_index}: force enabled; replacing complete chunk.", flush=True)
                shutil.rmtree(chunk_path)
            else:
                print(f"Phase 3H-J batch {chunk_start}-{end_index}: already complete; skipping.", flush=True)
                chunk_paths.append(chunk_path)
                continue
        status = inspect_chunk(chunk_path, expected_start=chunk_start, expected_count=chunk_count)
        if status["complete"]:
            print(f"Phase 3H-J batch {chunk_start}-{end_index}: already complete; skipping.", flush=True)
            chunk_paths.append(chunk_path)
            continue
        if chunk_path.exists() and not status["complete"]:
            if not force:
                raise ValueError(
                    f"Phase 3H-J batch {chunk_start}-{end_index} is incomplete "
                    f"(rows={status['rows']}, pngs={status['pngs']}). Pass --resume --force to replace only incomplete chunks."
                )
            print(f"Phase 3H-J batch {chunk_start}-{end_index}: incomplete; replacing partial chunk.", flush=True)
            shutil.rmtree(chunk_path)
        if chunk_start < min_start_index:
            raise ValueError(f"Phase 3H-J batch {chunk_start}-{end_index} is before --start-index and is not complete.")
        command = build_generation_command(
            blender_executable=blender_executable,
            dataset=str(chunk_path),
            samples=chunk_count,
            seed=seed,
            blend_file=blend_file,
            overwrite=True,
            start_index=chunk_start,
            mobile_realism=mobile_realism,
            distance_jitter=distance_jitter,
            camera_height_jitter=camera_height_jitter,
            body_rotation_jitter=body_rotation_jitter,
            lighting_jitter=lighting_jitter,
            background_jitter=background_jitter,
            phone_framing_jitter=phone_framing_jitter,
        )
        commands.append(" ".join(command))
        print(f"Phase 3H-J batch {chunk_start}-{end_index}: rendering {chunk_count} samples.", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        status = inspect_chunk(chunk_path, expected_start=chunk_start, expected_count=chunk_count)
        if not status["complete"]:
            raise ValueError(f"Phase 3H-J batch {chunk_start}-{end_index} did not complete: {status}")
        print(f"Phase 3H-J batch {chunk_start}-{end_index}: complete ({status['rows']} labels, {status['pngs']} PNGs).", flush=True)
        chunk_paths.append(chunk_path)

    merge_chunk_datasets(chunk_paths=chunk_paths, dataset_path=dataset_path, expected_samples=samples, batch_size=batch_size)
    return {
        "batch_size": batch_size,
        "batch_count": len(chunk_paths),
        "batch_out": str(batch_root),
        "commands": commands,
    }


def build_generation_command(
    *,
    blender_executable: str,
    dataset: str,
    samples: int,
    seed: int,
    blend_file: str,
    overwrite: bool,
    start_index: int = 1,
    mobile_realism: bool = DEFAULT_MOBILE_REALISM,
    distance_jitter: float = DEFAULT_DISTANCE_JITTER,
    camera_height_jitter: float = DEFAULT_CAMERA_HEIGHT_JITTER,
    body_rotation_jitter: float = DEFAULT_BODY_ROTATION_JITTER,
    lighting_jitter: float = DEFAULT_LIGHTING_JITTER,
    background_jitter: float = DEFAULT_BACKGROUND_JITTER,
    phone_framing_jitter: float = DEFAULT_PHONE_FRAMING_JITTER,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/generate_blend_dataset.py",
        "--source",
        "blend",
        "--blend-file",
        blend_file,
        "--out",
        dataset,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--start-index",
        str(start_index),
        "--blender-executable",
        blender_executable,
        "--shape-key-range",
        str(DEFAULT_SHAPE_KEY_RANGE),
        "--label-formula-version",
        PHASE_3H_J_LABEL_FORMULA_VERSION,
        "--label-measurement-scale",
        str(DEFAULT_LABEL_MEASUREMENT_SCALE),
        "--safe-framing-scale",
        str(DEFAULT_SAFE_FRAMING_SCALE),
        "--distance-jitter",
        str(distance_jitter),
        "--camera-height-jitter",
        str(camera_height_jitter),
        "--body-rotation-jitter",
        str(body_rotation_jitter),
        "--lighting-jitter",
        str(lighting_jitter),
        "--background-jitter",
        str(background_jitter),
        "--phone-framing-jitter",
        str(phone_framing_jitter),
        "--view-subdirs",
    ]
    if mobile_realism:
        command.append("--mobile-realism")
    if overwrite:
        command.append("--overwrite")
    return command


def inspect_chunk(chunk_path: Path, *, expected_start: int, expected_count: int) -> dict[str, Any]:
    labels_path = chunk_path / "labels.csv"
    metadata_path = chunk_path / "metadata.json"
    images_dir = chunk_path / "images"
    rows: list[dict[str, str]] = []
    if labels_path.exists():
        rows = read_labels(labels_path)
    png_count = sum(1 for _ in images_dir.rglob("*.png")) if images_dir.exists() else 0
    first_expected = f"sample_{expected_start:06d}"
    last_expected = f"sample_{expected_start + expected_count - 1:06d}"
    first_sample = rows[0]["sample_id"] if rows else ""
    last_sample = rows[-1]["sample_id"] if rows else ""
    view_dirs_exist = all((images_dir / view).exists() for view in CAMERA_VIEWS)
    complete = (
        labels_path.exists()
        and metadata_path.exists()
        and len(rows) == expected_count
        and png_count == expected_count * len(CAMERA_VIEWS)
        and first_sample == first_expected
        and last_sample == last_expected
        and view_dirs_exist
    )
    return {
        "chunk": str(chunk_path),
        "complete": complete,
        "labels_exists": labels_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "rows": len(rows),
        "pngs": png_count,
        "first_sample": first_sample,
        "last_sample": last_sample,
        "expected_first_sample": first_expected,
        "expected_last_sample": last_expected,
        "view_dirs_exist": view_dirs_exist,
    }


def merge_chunk_datasets(
    *,
    chunk_paths: list[Path],
    dataset_path: Path,
    expected_samples: int,
    batch_size: int,
) -> None:
    rows: list[dict[str, str]] = []
    samples_metadata: list[dict[str, Any]] = []
    metadata_template: dict[str, Any] | None = None
    for chunk_path in chunk_paths:
        chunk_labels = read_labels(chunk_path / "labels.csv")
        chunk_metadata = json.loads((chunk_path / "metadata.json").read_text(encoding="utf-8"))
        if metadata_template is None:
            metadata_template = chunk_metadata
        rows.extend(chunk_labels)
        samples_metadata.extend(chunk_metadata.get("samples", []))
        for row in chunk_labels:
            for view in CAMERA_VIEWS:
                relative_path = row[f"{view}_image"]
                source_image = chunk_path / relative_path
                target_image = dataset_path / relative_path
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, target_image)

    if len(rows) != expected_samples:
        raise ValueError(f"Merged labels count {len(rows)} does not match expected {expected_samples}")
    if metadata_template is None:
        raise ValueError("No batch metadata was generated")

    dataset_path.mkdir(parents=True, exist_ok=True)
    with (dataset_path / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = dict(metadata_template)
    metadata["sample_count"] = expected_samples
    metadata["samples"] = samples_metadata
    metadata["batch_generation"] = {
        "enabled": True,
        "batch_size": batch_size,
        "batch_count": len(chunk_paths),
        "chunk_paths": [str(path) for path in chunk_paths],
    }
    (dataset_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_audit_command(*, dataset: str, audit_out: str, samples: int) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_blend_dataset.py",
        "--dataset",
        dataset,
        "--out",
        audit_out,
        "--expected-samples",
        str(samples),
        "--strict",
    ]


def build_correlation_command(*, dataset: str, out: str) -> list[str]:
    return [
        sys.executable,
        "scripts/audit_blend_label_visual_correlation.py",
        "--dataset",
        dataset,
        "--out",
        out,
        "--target-columns",
        *TARGET_COLUMNS,
        "--min-abs-correlation",
        str(DEFAULT_MIN_ABS_CORRELATION),
    ]


def build_training_command(
    *,
    dataset: str,
    out: str,
    seed: int,
    test_size: float,
    audit_report: str,
) -> list[str]:
    return [
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
        *TARGET_COLUMNS,
        "--strict-audit-required",
        "--audit-report",
        audit_report,
    ]


def validate_phase_3h_j_dataset(dataset: str | Path, expected_samples: int = DEFAULT_SAMPLES) -> dict[str, Any]:
    dataset_path = Path(dataset)
    ensure_not_archived_dataset(dataset_path)
    labels_path = dataset_path / "labels.csv"
    metadata_path = dataset_path / "metadata.json"
    images_dir = dataset_path / "images"
    view_dirs = {view: images_dir / view for view in CAMERA_VIEWS}
    for path in (dataset_path, labels_path, metadata_path, images_dir, *view_dirs.values()):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Phase 3H-J dataset path: {path}")

    rows = read_labels(labels_path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} labels, found {len(rows)}")
    image_count = sum(1 for _ in images_dir.rglob("*.png"))
    expected_images = expected_samples * len(CAMERA_VIEWS)
    if image_count != expected_images:
        raise ValueError(f"Expected {expected_images} PNG images, found {image_count}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_columns = [column for column in [*TARGET_COLUMNS, *TRACEABILITY_COLUMNS] if column not in rows[0]]
    if missing_columns:
        raise ValueError("labels.csv missing required columns: " + ", ".join(missing_columns))
    missing_images = missing_images_for_rows(dataset_path, rows)
    if missing_images:
        raise ValueError("Missing images: " + "; ".join(missing_images[:10]))
    row_errors = validate_shape_key_coupled_rows(rows)
    if row_errors:
        raise ValueError("Shape-key coupled label validation failed: " + "; ".join(row_errors[:10]))

    expected_metadata = {
        "label_generation_mode": LABEL_GENERATION_MODE,
        "label_formula_version": PHASE_3H_J_LABEL_FORMULA_VERSION,
        "synthetic_labels": True,
        "real_world_validated": False,
        "variation_source": "shape_keys_safe_range_plus_mobile_realism",
        "mobile_realism": True,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"metadata {key} must be {expected_value!r}, got {metadata.get(key)!r}")
    if int(metadata.get("shape_key_count", 0)) <= 0:
        raise ValueError("metadata shape_key_count must be positive")
    mobile_realism_summary = validate_mobile_realism_metadata(metadata)

    clipping = audit_clipped_views(dataset_path, rows)
    if clipping["clipped_view_count"] > 0:
        raise ValueError(
            "Phase 3H-J clipping check failed; do not run correlation or training. "
            f"clipped_views={clipping['clipped_view_count']}, "
            f"by_view={clipping['clipped_views_by_view']}, "
            f"first_failures={clipping['first_failures']}"
        )
    view_sanity = compute_view_sanity(dataset_path, rows)
    if not view_sanity["passed"]:
        raise ValueError("front/side/back view sanity failed")
    label_variation = summarize_label_variation(rows)
    if not label_variation["variation_exists"]:
        raise ValueError("Target labels do not vary")

    return {
        "dataset": str(dataset_path),
        "sample_count": len(rows),
        "image_count": image_count,
        "label_generation_mode": metadata["label_generation_mode"],
        "label_formula_version": metadata["label_formula_version"],
        "synthetic_labels": metadata["synthetic_labels"],
        "real_world_validated": metadata["real_world_validated"],
        "variation_source": metadata["variation_source"],
        "shape_key_count": int(metadata["shape_key_count"]),
        "mobile_realism": bool(metadata["mobile_realism"]),
        "mobile_realism_settings": mobile_realism_summary,
        "clipping": clipping,
        "view_folders": {view: str(path) for view, path in view_dirs.items()},
        "view_sanity": view_sanity,
        "label_variation": label_variation,
    }


def ensure_not_archived_dataset(dataset: str | Path) -> None:
    normalized = str(dataset).replace("\\", "/").lower()
    if any(marker in normalized for marker in ARCHIVED_DATASET_MARKERS):
        raise ValueError(f"Phase 3H-J must not use archived old datasets: {dataset}")


def validate_mobile_realism_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    settings = metadata.get("mobile_realism_settings")
    if not isinstance(settings, dict):
        raise ValueError("metadata mobile_realism_settings must be present")
    expected_enabled = settings.get("mobile_realism")
    if expected_enabled is not True:
        raise ValueError("metadata mobile_realism_settings.mobile_realism must be true")
    allowed_ranges = {
        "distance_jitter": (0.0, 0.08),
        "camera_height_jitter": (0.0, 0.08),
        "body_rotation_jitter": (0.0, 6.0),
        "lighting_jitter": (0.0, 0.18),
        "background_jitter": (0.0, 0.12),
        "phone_framing_jitter": (0.0, 0.04),
    }
    normalized: dict[str, Any] = {"mobile_realism": True}
    for key, (lower, upper) in allowed_ranges.items():
        if key not in settings:
            raise ValueError(f"metadata mobile_realism_settings.{key} is required")
        value = float(settings[key])
        if not lower <= value <= upper:
            raise ValueError(f"metadata mobile_realism_settings.{key}={value} outside conservative range {lower}-{upper}")
        normalized[key] = value
    return normalized


def missing_images_for_rows(dataset_path: Path, rows: list[dict[str, str]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        sample_id = row.get("sample_id", "<missing sample_id>")
        for view in CAMERA_VIEWS:
            relative_path = row.get(f"{view}_image", "")
            expected_prefix = f"images/{view}/"
            if not relative_path.replace("\\", "/").startswith(expected_prefix):
                missing.append(f"{sample_id}:{view}:expected {expected_prefix} path, got {relative_path}")
                continue
            image_path = dataset_path / relative_path
            if not image_path.exists():
                missing.append(f"{sample_id}:{view}:{relative_path}")
    return missing


def audit_clipped_views(dataset_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    clipped_by_view = {view: 0 for view in CAMERA_VIEWS}
    first_failures: list[dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        for view in CAMERA_VIEWS:
            relative_path = row.get(f"{view}_image", "")
            try:
                extract_image_features(dataset_path / relative_path, view)
            except ValueError as exc:
                message = str(exc)
                if "truncated at the image boundary" not in message:
                    raise
                clipped_by_view[view] += 1
                if len(first_failures) < 10:
                    first_failures.append(
                        {
                            "sample_id": sample_id,
                            "view": view,
                            "path": relative_path,
                            "error": message,
                        }
                    )
    clipped_count = sum(clipped_by_view.values())
    return {
        "clipped_view_count": clipped_count,
        "clipped_views_by_view": clipped_by_view,
        "first_failures": first_failures,
        "fatal_for_correlation_training": clipped_count > 0,
        "instruction": "Do not run correlation or benchmark/training when clipped_view_count is greater than 0.",
    }


def compute_view_sanity(dataset_path: Path, rows: list[dict[str, str]], max_samples: int = 25) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for row in rows[:max_samples]:
        images = {
            view: Image.open(dataset_path / row[f"{view}_image"]).convert("L").resize((96, 96))
            for view in CAMERA_VIEWS
        }
        for left, right in (("front", "side"), ("side", "back"), ("back", "front")):
            diff = ImageChops.difference(images[left], images[right])
            score = float(ImageStat.Stat(diff).mean[0]) / 255.0
            scores.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "view_pair": f"{left}_{right}",
                    "difference_score": score,
                    "passed": score >= MIN_VIEW_DIFFERENCE_SCORE,
                }
            )
    return {
        "passed": bool(scores) and all(score["passed"] for score in scores),
        "minimum_difference_score": MIN_VIEW_DIFFERENCE_SCORE,
        "checked_sample_count": min(len(rows), max_samples),
        "scores": scores,
    }


def summarize_label_variation(rows: list[dict[str, str]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    stats: dict[str, dict[str, float | int | bool]] = {}
    for target in TARGET_COLUMNS:
        values = [float(row[target]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = variance ** 0.5
        cv = std / max(abs(mean), 1e-6)
        unique_count = len({round(value, 6) for value in values})
        low = std < 1.0 or cv < 0.02 or unique_count < 5
        stats[target] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "std": std,
            "coefficient_of_variation": cv,
            "unique_count": unique_count,
            "low_variation": low,
        }
        if low:
            warnings.append(
                {
                    "target": target,
                    "category": "low_label_variation",
                    "severity": "warning",
                    "value": std,
                }
            )
    return {
        "variation_exists": all(not bool(row["low_variation"]) or float(row["std"]) > 0 for row in stats.values()),
        "stats": stats,
        "low_label_variation_warnings": warnings,
    }


def read_audit_summary(audit_report_path: Path) -> dict[str, Any]:
    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if report.get("strict") is not True:
        raise ValueError(f"Audit report was not strict: {audit_report_path}")
    if report.get("passed") is not True:
        raise ValueError(f"Strict audit did not pass: {audit_report_path}")
    return {
        "passed": bool(report.get("passed")),
        "warnings_count": len(report.get("warnings", [])),
        "errors_count": len(report.get("errors", [])),
        "strict_failures_count": len(report.get("strict_failures", [])),
        "flagged_sample_count": int(report.get("flagged_sample_count", 0)),
        "view_sanity_passed": bool(report.get("view_sanity", {}).get("passed")),
        "label_variation_exists": bool(report.get("label_audit", {}).get("variation_exists")),
    }


def compare_to_phase_3h_i(metrics: dict[str, Any], phase_3h_i_metrics: str | Path = PHASE_3H_I_METRICS) -> dict[str, Any]:
    baseline_path = Path(phase_3h_i_metrics)
    if not baseline_path.exists():
        return {
            "available": False,
            "baseline_metrics": str(baseline_path),
            "reason": "Phase 3H-I metrics report was not found.",
        }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    target_comparisons = {}
    for target in TARGET_COLUMNS:
        new_value = float(metrics["mae_by_target"][target])
        old_value = float(baseline["mae_by_target"][target])
        delta = new_value - old_value
        target_comparisons[target] = {
            "phase_3h_i_mae": old_value,
            "phase_3h_j_mae": new_value,
            "delta": delta,
            "status": "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged",
        }
    overall_delta = float(metrics["overall_mean_mae"]) - float(baseline["overall_mean_mae"])
    return {
        "available": True,
        "phase_3h_i_best_model": baseline.get("best_model"),
        "phase_3h_i_overall_mean_mae": float(baseline["overall_mean_mae"]),
        "phase_3h_j_overall_mean_mae": float(metrics["overall_mean_mae"]),
        "overall_delta": overall_delta,
        "overall_status": "improved" if overall_delta < 0 else "worsened" if overall_delta > 0 else "unchanged",
        "mae_by_target": target_comparisons,
    }


def read_labels(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))


def format_summary(summary: dict[str, Any]) -> str:
    audit = summary.get("audit") or {"passed": "skipped", "warnings_count": 0, "errors_count": 0, "flagged_sample_count": 0}
    lines = [
        "Phase 3H-J coupled verification passed.",
        f"Dataset: {summary['dataset']}",
        f"Samples: {summary['sample_count']}",
        f"Images: {summary['image_count']}",
        f"Train/Test: {summary.get('train_sample_count')}/{summary.get('test_sample_count')}",
        "Model candidates: " + ", ".join(row["model"] for row in summary.get("model_ranking", [])),
        f"Strict audit passed: {audit['passed']}",
        f"Audit warnings/errors/flagged: {audit['warnings_count']}/{audit['errors_count']}/{audit['flagged_sample_count']}",
        f"Label generation mode: {summary['label_generation_mode']}",
        f"Label formula version: {summary['label_formula_version']}",
        f"Mobile realism: {summary.get('mobile_realism')}",
        "Realism settings: " + json.dumps(summary.get("mobile_realism_settings", {}), sort_keys=True),
        f"Clipped views: {summary.get('clipping', {}).get('clipped_view_count', 'unknown')}",
        f"Best model: {summary['best_model'] or 'skipped'}",
    ]
    if summary.get("overall_mean_mae") is not None:
        lines.append(f"Overall mean MAE: {float(summary['overall_mean_mae']):.4f}")
    else:
        lines.append("Overall mean MAE: skipped")
    comparison = summary["comparison_to_phase_3h_i"]
    if comparison.get("available"):
        lines.append(
            "Comparison vs Phase 3H-I: "
            f"{comparison['overall_status']} ({comparison['overall_delta']:.4f} cm)"
        )
    else:
        lines.append(f"Comparison vs Phase 3H-I: unavailable ({comparison['reason']})")
    if summary.get("mae_by_target"):
        lines.append("MAE by target:")
        for target, value in summary["mae_by_target"].items():
            if comparison.get("available"):
                delta = comparison["mae_by_target"][target]["delta"]
                status = comparison["mae_by_target"][target]["status"]
                lines.append(f"  {target}: {float(value):.4f} ({status}, delta={delta:.4f})")
            else:
                lines.append(f"  {target}: {float(value):.4f}")
    else:
        lines.append("MAE by target: skipped")
    if summary.get("correlation_by_target"):
        lines.append("Strongest visual correlation per target:")
        for target, row in summary["correlation_by_target"].items():
            lines.append(f"  {target}: {float(row['abs_max_correlation']):.4f} via {row['feature']}")
    else:
        lines.append("Strongest visual correlation per target: skipped")
    lines.append("Weak targets below 0.25: " + (", ".join(summary["weak_targets_below_threshold"]) or "none"))
    lines.append(
        "Low-label-variation warnings: "
        + (", ".join(row["target_or_feature"] for row in summary["label_variation_warnings"]) or "none")
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and benchmark the Phase 3H-J mobile-realism 1000-sample Blender dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--correlation-out", default=DEFAULT_CORRELATION_OUT)
    parser.add_argument("--training-out", default=DEFAULT_TRAINING_OUT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--blend-file", default=DEFAULT_BLEND_FILE)
    parser.add_argument("--blender-executable", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-out", default=DEFAULT_BATCH_OUT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="Reuse the existing dataset and never invoke Blender generation.")
    parser.add_argument("--reuse-existing", action="store_true", help="Alias for --no-render.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--mobile-realism", action=argparse.BooleanOptionalAction, default=DEFAULT_MOBILE_REALISM)
    parser.add_argument("--distance-jitter", type=float, default=DEFAULT_DISTANCE_JITTER)
    parser.add_argument("--camera-height-jitter", type=float, default=DEFAULT_CAMERA_HEIGHT_JITTER)
    parser.add_argument("--body-rotation-jitter", type=float, default=DEFAULT_BODY_ROTATION_JITTER)
    parser.add_argument("--lighting-jitter", type=float, default=DEFAULT_LIGHTING_JITTER)
    parser.add_argument("--background-jitter", type=float, default=DEFAULT_BACKGROUND_JITTER)
    parser.add_argument("--phone-framing-jitter", type=float, default=DEFAULT_PHONE_FRAMING_JITTER)
    parser.add_argument("--phase-3h-i-metrics", default=PHASE_3H_I_METRICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = args.dataset
    batch_out = args.batch_out
    audit_out = args.audit_out
    correlation_out = args.correlation_out
    training_out = args.training_out
    resume = args.resume
    if args.smoke:
        if dataset == DEFAULT_DATASET:
            dataset = DEFAULT_SMOKE_DATASET
        if batch_out == DEFAULT_BATCH_OUT:
            batch_out = DEFAULT_SMOKE_BATCH_OUT
        if audit_out == DEFAULT_AUDIT_OUT:
            audit_out = "artifacts/phase_3h_j_mobile_realism_smoke_audit"
        if correlation_out == DEFAULT_CORRELATION_OUT:
            correlation_out = "artifacts/phase_3h_j_mobile_realism_smoke_label_visual_correlation"
        if training_out == DEFAULT_TRAINING_OUT:
            training_out = "artifacts/phase_3h_j_mobile_realism_smoke_blend_baseline"
        resume = True
    try:
        summary = verify_coupled_1000(
            dataset=dataset,
            audit_out=audit_out,
            correlation_out=correlation_out,
            training_out=training_out,
            samples=args.samples,
            seed=args.seed,
            test_size=args.test_size,
            blend_file=args.blend_file,
            blender_executable=args.blender_executable,
            resume=resume,
            force=args.force,
            start_index=args.start_index,
            batch_size=args.batch_size,
            batch_out=batch_out,
            run_benchmark=(args.run_benchmark or not args.smoke),
            smoke=args.smoke,
            no_render=args.no_render or args.reuse_existing,
            mobile_realism=args.mobile_realism,
            distance_jitter=args.distance_jitter,
            camera_height_jitter=args.camera_height_jitter,
            body_rotation_jitter=args.body_rotation_jitter,
            lighting_jitter=args.lighting_jitter,
            background_jitter=args.background_jitter,
            phone_framing_jitter=args.phone_framing_jitter,
            phase_3h_i_metrics=args.phase_3h_i_metrics,
        )
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Phase 3H-J coupled 1000 verification failed: {exc}")
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

