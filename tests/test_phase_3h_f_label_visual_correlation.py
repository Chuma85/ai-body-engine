from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import numpy as np
import pytest

from scripts.audit_blend_label_visual_correlation import (
    DEFAULT_MIN_ABS_CORRELATION,
    DEFAULT_TARGET_COLUMNS,
    audit_label_visual_correlation,
    compute_feature_label_correlations,
    flag_targets,
    pearson_correlation,
    rank_values,
    summarize_labels,
    top_features_by_target,
)
from scripts.verify_phase_3h_f_label_visual_correlation import verify_phase_3h_f_label_visual_correlation


def test_correlation_computation_on_small_synthetic_arrays() -> None:
    features = np.asarray(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ]
    )
    targets = np.asarray([[10.0], [20.0], [30.0], [40.0]])

    correlations = compute_feature_label_correlations(
        feature_matrix=features,
        feature_names=["increasing_feature", "decreasing_feature"],
        target_matrix=targets,
        target_columns=["height_cm"],
    )

    by_feature = {row["feature"]: row for row in correlations}
    assert by_feature["increasing_feature"]["pearson"] == pytest.approx(1.0)
    assert by_feature["increasing_feature"]["spearman"] == pytest.approx(1.0)
    assert by_feature["decreasing_feature"]["pearson"] == pytest.approx(-1.0)
    assert rank_values(np.asarray([10.0, 20.0, 20.0, 40.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert pearson_correlation(np.asarray([1.0, 1.0]), np.asarray([2.0, 3.0])) is None


def test_weak_correlation_flagging() -> None:
    label_summary = [
        {
            "target": "height_cm",
            "std": 10.0,
            "coefficient_of_variation": 0.06,
            "unique_count": 8,
            "safe_range_violation_count": 0,
            "low_variation": False,
        }
    ]
    feature_summary = [
        {
            "feature": "front_raw_bbox_width_ratio",
            "std": 0.1,
            "near_zero_variation": False,
            "key_feature": True,
        }
    ]
    top_features = [
        {
            "target": "height_cm",
            "rank": 1,
            "feature": "front_raw_bbox_width_ratio",
            "abs_max_correlation": 0.12,
        }
    ]
    target_corr = [{"target_a": "height_cm", "target_b": "height_cm", "pearson": 1.0, "spearman": 1.0}]

    flags = flag_targets(
        label_summary=label_summary,
        feature_summary=feature_summary,
        top_features=top_features,
        target_correlation_rows=target_corr,
        min_abs_correlation=0.25,
    )

    assert any(flag["category"] == "weak_visual_correlation" for flag in flags)


def test_label_summary_schema_and_low_variation_flag() -> None:
    labels = np.asarray(
        [
            [180.0, 90.0],
            [180.1, 92.0],
            [180.2, 94.0],
            [180.3, 96.0],
        ]
    )

    summary = summarize_labels(labels, ["height_cm", "chest_cm"])

    height = next(row for row in summary if row["target"] == "height_cm")
    assert {"target", "min", "max", "mean", "std", "coefficient_of_variation", "unique_count"}.issubset(height)
    assert height["low_variation"] is True
    assert height["safe_range_violation_count"] == 0


def test_report_output_schema(tmp_path: Path) -> None:
    dataset = _write_fake_blend_dataset(tmp_path, sample_count=8)
    out = tmp_path / "artifacts" / "phase_3h_f"

    report = audit_label_visual_correlation(
        dataset=dataset,
        out=out,
        target_columns=DEFAULT_TARGET_COLUMNS,
        min_abs_correlation=DEFAULT_MIN_ABS_CORRELATION,
    )

    assert report["sample_count"] == 8
    assert set(report["strongest_visual_correlation_by_target"]) == set(DEFAULT_TARGET_COLUMNS)
    assert (out / "correlation_report.json").exists()
    assert (out / "feature_label_correlation.csv").exists()
    assert (out / "top_features_by_target.csv").exists()
    stored = json.loads((out / "correlation_report.json").read_text(encoding="utf-8"))
    assert stored["phase"] == "3H-F"
    assert "recommended_next_action" in stored


def test_missing_dataset_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset folder"):
        verify_phase_3h_f_label_visual_correlation(dataset=str(tmp_path / "missing_dataset"))


def test_top_features_by_target_sorts_by_abs_max_correlation() -> None:
    top = top_features_by_target(
        [
            {"target": "height_cm", "feature": "weak", "pearson": 0.1, "spearman": 0.2, "abs_max_correlation": 0.2},
            {"target": "height_cm", "feature": "strong", "pearson": -0.7, "spearman": -0.6, "abs_max_correlation": 0.7},
        ],
        top_n=2,
    )

    assert [row["feature"] for row in top] == ["strong", "weak"]
    assert [row["rank"] for row in top] == [1, 2]


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
        ]
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, sample_count + 1):
            sample_id = f"sample_{index:06d}"
            front_width = 12 + index * 2
            side_width = 7 + index
            back_width = 10 + index
            _write_rect_image(images / f"{sample_id}_front.png", x_min=36 - front_width // 2, x_max=36 + front_width // 2)
            _write_rect_image(images / f"{sample_id}_side.png", x_min=38 - side_width // 2, x_max=38 + side_width // 2)
            _write_rect_image(images / f"{sample_id}_back.png", x_min=37 - back_width // 2, x_max=37 + back_width // 2)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "front_image": f"images/{sample_id}_front.png",
                    "side_image": f"images/{sample_id}_side.png",
                    "back_image": f"images/{sample_id}_back.png",
                    "height_cm": f"{160 + index * 2.0:.1f}",
                    "chest_cm": f"{80 + front_width * 1.2:.1f}",
                    "waist_cm": f"{70 + front_width:.1f}",
                    "hip_cm": f"{85 + back_width:.1f}",
                    "shoulder_cm": f"{38 + index * 0.5:.1f}",
                    "inseam_cm": f"{70 + index:.1f}",
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
    return dataset


def _write_rect_image(path: Path, *, x_min: int, x_max: int) -> None:
    image = Image.new("RGB", (90, 110), (50, 50, 50))
    pixels = image.load()
    for y in range(12, 98):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 190, 165)
    image.save(path)
