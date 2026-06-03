from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.train_blend_dataset_baseline import (
    DEFAULT_TARGET_COLUMNS,
    extract_blend_image_features,
    rank_models,
    train_blend_dataset_baseline,
    validate_blend_dataset,
)
from scripts.verify_phase_3h_e_blend_baseline import verify_expected_artifacts


def test_dataset_validation_fails_when_labels_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "metadata.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="labels.csv"):
        validate_blend_dataset(dataset, DEFAULT_TARGET_COLUMNS)


def test_dataset_validation_detects_missing_view_image(tmp_path: Path) -> None:
    dataset = _write_fake_blend_dataset(tmp_path, sample_count=6)
    missing_image = dataset / "images" / "sample_000001_front.png"
    missing_image.unlink()

    with pytest.raises(FileNotFoundError, match="missing front image"):
        validate_blend_dataset(dataset, DEFAULT_TARGET_COLUMNS)


def test_feature_extraction_on_small_fake_front_side_back_images(tmp_path: Path) -> None:
    dataset = _write_fake_blend_dataset(tmp_path, sample_count=6)
    rows = _read_labels(dataset / "labels.csv")

    features = extract_blend_image_features(rows[0], dataset)

    assert features["front_raw_image_width_px"] == 80.0
    assert features["side_raw_image_width_px"] == 80.0
    assert features["back_raw_image_width_px"] == 80.0
    assert "front_horizontal_projection_bin_00" in features
    assert "side_back_raw_bbox_width_ratio_ratio" in features
    assert "front_side_back_area_proxy" in features


def test_training_writes_metrics_schema_and_expected_artifacts(tmp_path: Path) -> None:
    dataset = _write_fake_blend_dataset(tmp_path, sample_count=12)
    out = tmp_path / "artifacts" / "phase_3h_e"

    result = train_blend_dataset_baseline(
        dataset=dataset,
        out=out,
        seed=42,
        test_size=0.25,
        target_columns=DEFAULT_TARGET_COLUMNS,
        strict_audit_required=True,
        audit_report=tmp_path / "artifacts" / "phase_3h_blend_250_audit" / "audit_report.json",
    )

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    artifact_summary = verify_expected_artifacts(out)
    assert Path(result["metrics_path"]).exists()
    assert metrics["sample_count"] == 12
    assert metrics["image_count"] == 36
    assert metrics["real_world_validated"] is False
    assert metrics["synthetic_only"] is True
    assert metrics["best_model"] in {"ridge", "random_forest", "knn"}
    assert set(metrics["mae_by_target"]) == set(DEFAULT_TARGET_COLUMNS)
    assert artifact_summary["prediction_rows"] == 3
    assert (out / "best_model.joblib").exists()


def test_model_ranking_sorts_by_overall_mean_mae() -> None:
    ranking = rank_models(
        {
            "ridge": {"test": {"overall_mean_mae": 3.0}},
            "random_forest": {"test": {"overall_mean_mae": 2.0}},
            "knn": {"test": {"overall_mean_mae": 4.0}},
        }
    )

    assert [row["model"] for row in ranking] == ["random_forest", "ridge", "knn"]
    assert [row["rank"] for row in ranking] == [1, 2, 3]


def _write_fake_blend_dataset(tmp_path: Path, sample_count: int) -> Path:
    dataset = tmp_path / "data" / "synthetic" / "phase_3h_blend_250"
    images = dataset / "images"
    images.mkdir(parents=True)
    labels_path = dataset / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as labels_file:
        fieldnames = [
            "sample_id",
            "front_image",
            "side_image",
            "back_image",
            *DEFAULT_TARGET_COLUMNS,
            "variation_source",
            "synthetic_labels",
            "real_world_validated",
        ]
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, sample_count + 1):
            sample_id = f"sample_{index:06d}"
            front_width = 16 + index
            side_width = 8 + (index % 5)
            back_width = 14 + (index % 6)
            _write_rect_image(images / f"{sample_id}_front.png", x_min=30 - front_width // 2, x_max=30 + front_width // 2)
            _write_rect_image(images / f"{sample_id}_side.png", x_min=36 - side_width // 2, x_max=36 + side_width // 2)
            _write_rect_image(images / f"{sample_id}_back.png", x_min=32 - back_width // 2, x_max=32 + back_width // 2)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "front_image": f"images/{sample_id}_front.png",
                    "side_image": f"images/{sample_id}_side.png",
                    "back_image": f"images/{sample_id}_back.png",
                    "height_cm": f"{160 + index * 1.5:.1f}",
                    "chest_cm": f"{80 + front_width * 1.2:.1f}",
                    "waist_cm": f"{70 + front_width:.1f}",
                    "hip_cm": f"{85 + back_width:.1f}",
                    "shoulder_cm": f"{38 + index * 0.4:.1f}",
                    "inseam_cm": f"{70 + index * 0.7:.1f}",
                    "variation_source": "shape_keys_safe_range",
                    "synthetic_labels": "true",
                    "real_world_validated": "false",
                }
            )
    (dataset / "metadata.json").write_text(
        json.dumps(
            {
                "sample_count": sample_count,
                "synthetic_labels": True,
                "real_world_validated": False,
                "variation_source": "shape_keys_safe_range",
                "shape_key_count": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_out = tmp_path / "artifacts" / "phase_3h_blend_250_audit"
    audit_out.mkdir(parents=True)
    (audit_out / "audit_report.json").write_text(
        json.dumps({"passed": True, "strict": True, "row_count": sample_count}) + "\n",
        encoding="utf-8",
    )
    return dataset


def _write_rect_image(path: Path, *, x_min: int, x_max: int) -> None:
    image = Image.new("RGB", (80, 100), (50, 50, 50))
    pixels = image.load()
    for y in range(10, 90):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (215, 190, 165)
    image.save(path)


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as labels_file:
        return list(csv.DictReader(labels_file))
