from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.train_blend_dataset_baseline import DEFAULT_TARGET_COLUMNS, train_blend_dataset_baseline
from scripts.verify_phase_3h_h_coupled_250_retrain import (
    PHASE_3H_E_BASELINE,
    compare_metrics_to_phase_3h_e,
    validate_coupled_dataset_outputs,
)
from synthetic.blender.blend_dataset import (
    BLEND_LABEL_COLUMNS,
    LABEL_FORMULA_VERSION,
    LABEL_GENERATION_MODE,
    SHAPE_KEY_COUPLED_LABEL_SOURCE,
    camera_set_name,
    generate_shape_key_coupled_measurements,
    shape_key_label_metadata,
    shape_key_traceability_json,
)


def test_train_script_accepts_explicit_audit_report(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=12)
    audit_report = _write_audit_report(tmp_path / "custom_audit" / "audit_report.json", passed=True, strict=True)

    result = train_blend_dataset_baseline(
        dataset=dataset,
        out=tmp_path / "training",
        seed=42,
        test_size=0.25,
        target_columns=DEFAULT_TARGET_COLUMNS,
        strict_audit_required=True,
        audit_report=audit_report,
    )

    assert Path(result["metrics_path"]).exists()
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["audit_report"] == str(audit_report)


def test_train_strict_audit_fails_when_explicit_report_missing(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=8)

    with pytest.raises(FileNotFoundError, match="Missing strict audit report"):
        train_blend_dataset_baseline(
            dataset=dataset,
            out=tmp_path / "training",
            target_columns=DEFAULT_TARGET_COLUMNS,
            strict_audit_required=True,
            audit_report=tmp_path / "missing" / "audit_report.json",
        )


def test_train_strict_audit_fails_when_report_not_passed(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=8)
    audit_report = _write_audit_report(tmp_path / "audit" / "audit_report.json", passed=False, strict=True)

    with pytest.raises(ValueError, match="Strict audit did not pass"):
        train_blend_dataset_baseline(
            dataset=dataset,
            out=tmp_path / "training",
            target_columns=DEFAULT_TARGET_COLUMNS,
            strict_audit_required=True,
            audit_report=audit_report,
        )


def test_coupled_dataset_validation_checks_schema_and_counts(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=5)

    summary = validate_coupled_dataset_outputs(dataset, expected_samples=5)

    assert summary["sample_count"] == 5
    assert summary["image_count"] == 15
    assert summary["label_generation_mode"] == LABEL_GENERATION_MODE
    assert summary["label_formula_version"] == LABEL_FORMULA_VERSION
    assert summary["synthetic_labels"] is True
    assert summary["real_world_validated"] is False


def test_coupled_dataset_validation_fails_on_missing_traceability_column(tmp_path: Path) -> None:
    dataset = _write_coupled_dataset(tmp_path, sample_count=5)
    rows = _read_rows(dataset / "labels.csv")
    for row in rows:
        row.pop("shape_key_values_json")
    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        fieldnames = [column for column in BLEND_LABEL_COLUMNS if column != "shape_key_values_json"]
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="missing traceability columns"):
        validate_coupled_dataset_outputs(dataset, expected_samples=5)


def test_verification_summary_comparison_logic() -> None:
    metrics = {
        "overall_mean_mae": 10.0,
        "mae_by_target": {
            target: old_value - 1.0
            for target, old_value in PHASE_3H_E_BASELINE["mae_by_target"].items()
        },
    }

    comparison = compare_metrics_to_phase_3h_e(metrics)

    assert comparison["overall_status"] == "improved"
    assert comparison["overall_delta"] == pytest.approx(10.0 - PHASE_3H_E_BASELINE["overall_mean_mae"])
    assert all(row["status"] == "improved" for row in comparison["mae_by_target"].values())


def _write_audit_report(path: Path, *, passed: bool, strict: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "strict": strict,
                "warnings": [],
                "errors": [] if passed else ["failed"],
                "strict_failures": [] if passed else ["failed"],
                "flagged_sample_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_coupled_dataset(tmp_path: Path, sample_count: int) -> Path:
    dataset = tmp_path / "phase_3h_h_coupled_250"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    for index in range(1, sample_count + 1):
        sample_id = f"sample_{index:06d}"
        value = 0.01 + index * 0.01
        shape_keys = {
            "$md-$af-$fe-$yn": value,
            "$md-$af-$ma-$yn": value,
            "$md-$ca-$ma-$yn": value,
            "$md-universal-$ma-$yn-$av$mu-$av$wg": value,
        }
        payload = generate_shape_key_coupled_measurements(
            sample_id=sample_id,
            seed=42,
            shape_key_values=shape_keys,
            label_noise_cm=0.0,
        )
        image_names = {
            "front": f"images/{sample_id}_front.png",
            "side": f"images/{sample_id}_side.png",
            "back": f"images/{sample_id}_back.png",
        }
        _write_rect_image(dataset / image_names["front"], rect=(30, 10, 50 + index, 88))
        _write_rect_image(dataset / image_names["side"], rect=(38, 10, 48 + index // 2, 88))
        _write_rect_image(dataset / image_names["back"], rect=(28, 10, 52 + index, 88))
        rows.append(
            {
                "sample_id": sample_id,
                "front_image": image_names["front"],
                "side_image": image_names["side"],
                "back_image": image_names["back"],
                **payload["measurements"],
                "source_blend_file": "assets/body_meshes/base_body_scene.blend",
                "variation_source": "shape_keys_safe_range",
                "camera_set": camera_set_name(),
                "seed": "42",
                "label_source": SHAPE_KEY_COUPLED_LABEL_SOURCE,
                "synthetic_labels": "true",
                "real_world_validated": "false",
                "label_generation_mode": LABEL_GENERATION_MODE,
                **payload["factors"],
                "shape_key_values_json": shape_key_traceability_json(shape_keys),
                "body_shape_profile_id": payload["body_shape_profile_id"],
            }
        )
    with (dataset / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=BLEND_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "generator_version": "phase_3h_blend_dataset_v1",
        "source_blend_file": "assets/body_meshes/base_body_scene.blend",
        "camera_set": camera_set_name(),
        "sample_count": sample_count,
        "seed": 42,
        "synthetic_labels": True,
        "real_world_validated": False,
        "variation_source": "shape_keys_safe_range",
        "shape_key_count": 4,
        **shape_key_label_metadata(42),
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _write_rect_image(path: Path, rect: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (96, 112), (50, 50, 50))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 190, 165)
    image.save(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
