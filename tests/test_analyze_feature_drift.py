import numpy as np
import pytest

from training.experiments.analyze_feature_drift import (
    ablation_name,
    build_feature_drift_summary,
    feature_drift_group,
    feature_drift_rows,
    feature_group_drift_rows,
    format_feature_drift_report,
    matched_feature_matrices,
)


def test_feature_drift_rows_calculate_stats() -> None:
    clean = np.asarray([[1.0, 10.0], [3.0, 14.0]], dtype=np.float64)
    current = np.asarray([[2.0, 9.0], [5.0, 13.0]], dtype=np.float64)

    rows = feature_drift_rows("lighting_only", clean, current, ["feature_a", "feature_b"])

    assert rows[0]["clean_mean"] == 2.0
    assert rows[0]["feature_group"] == "normalized_shape"
    assert rows[0]["ablation_mean"] == 3.5
    assert rows[0]["mean_abs_drift"] == 1.5
    assert rows[1]["mean_signed_drift"] == -1.0


def test_matched_feature_matrices_align_by_sample_id() -> None:
    clean = {
        "sample_ids": ["sample_000001", "sample_000002"],
        "matrix": np.asarray([[1.0], [2.0]], dtype=np.float64),
    }
    current = {
        "sample_ids": ["sample_000002", "sample_000001"],
        "matrix": np.asarray([[20.0], [10.0]], dtype=np.float64),
    }

    clean_matrix, current_matrix = matched_feature_matrices(clean, current)

    assert clean_matrix.tolist() == [[1.0], [2.0]]
    assert current_matrix.tolist() == [[10.0], [20.0]]


def test_matched_feature_matrices_requires_shared_samples() -> None:
    clean = {"sample_ids": ["a"], "matrix": np.asarray([[1.0]], dtype=np.float64)}
    current = {"sample_ids": ["b"], "matrix": np.asarray([[2.0]], dtype=np.float64)}

    with pytest.raises(ValueError, match="No matching sample IDs"):
        matched_feature_matrices(clean, current)


def test_build_feature_drift_summary_identifies_top_drift() -> None:
    features = ["stable", "drifty"]
    dataset_features = {
        "clean_baseline": {
            "sample_ids": ["a", "b"],
            "feature_names": features,
            "matrix": np.asarray([[1.0, 10.0], [1.0, 10.0]], dtype=np.float64),
        },
        "background_only": {
            "sample_ids": ["a", "b"],
            "feature_names": features,
            "matrix": np.asarray([[1.0, 20.0], [1.0, 30.0]], dtype=np.float64),
        },
    }

    summary = build_feature_drift_summary(dataset_features, "clean_baseline", features)

    assert summary["feature_count"] == 2
    assert summary["top_drift_by_ablation"]["background_only"][0]["feature"] == "drifty"
    assert summary["top_drift_by_ablation"]["background_only"][0]["mean_abs_drift"] == 15.0
    assert summary["feature_group_drift_by_ablation"]["background_only"][0]["feature_group"] == "normalized_shape"


def test_feature_drift_group_detects_hybrid_feature_groups() -> None:
    assert feature_drift_group("front_bbox_width_ratio") == "normalized_shape"
    assert feature_drift_group("front_raw_bbox_height_px") == "raw_scale_camera"
    assert feature_drift_group("side_normalization_scale_factor") == "raw_scale_camera"
    assert feature_drift_group("front_to_side_bbox_height_ratio") == "raw_scale_camera"
    assert feature_drift_group("front_side_torso_volume_proxy") == "combined_hybrid"


def test_feature_group_drift_rows_summarize_groups() -> None:
    rows = [
        {"feature": "front_bbox_width_ratio", "feature_group": "normalized_shape", "mean_abs_drift": 0.5, "max_abs_drift": 1.0},
        {"feature": "front_raw_bbox_height_px", "feature_group": "raw_scale_camera", "mean_abs_drift": 10.0, "max_abs_drift": 15.0},
        {"feature": "front_raw_bbox_width_px", "feature_group": "raw_scale_camera", "mean_abs_drift": 5.0, "max_abs_drift": 7.0},
    ]

    group_rows = feature_group_drift_rows(rows)

    assert group_rows[0]["feature_group"] == "raw_scale_camera"
    assert group_rows[0]["feature_count"] == 2
    assert group_rows[0]["mean_abs_drift"] == 7.5
    assert group_rows[0]["top_feature"] == "front_raw_bbox_height_px"


def test_feature_drift_report_includes_raw_comparison_when_present() -> None:
    summary = {
        "feature_extractor_version": "test",
        "clean_dataset": "clean_baseline",
        "sample_count": 2,
        "feature_count": 1,
        "top_drift_by_ablation": {
            "background_only": [
                {
                    "feature": "normalized_feature",
                    "mean_abs_drift": 0.5,
                    "mean_signed_drift": 0.0,
                    "clean_mean": 1.0,
                    "ablation_mean": 1.5,
                    "clean_std": 0.0,
                    "ablation_std": 0.0,
                    "clean_min": 1.0,
                    "clean_max": 1.0,
                    "ablation_min": 1.5,
                    "ablation_max": 1.5,
                    "sample_count": 2,
                    "ablation": "background_only",
                }
            ]
        },
        "raw_top_drift_by_ablation": {
            "background_only": [
                {
                    "feature": "raw_feature",
                    "mean_abs_drift": 12.0,
                }
            ]
        },
        "rows": [],
        "recommendations": [],
    }

    report = format_feature_drift_report(summary)

    assert "Raw Framing Comparison" in report
    assert "raw_feature" in report
    assert "normalized_feature" in report


def test_ablation_name_strips_phase_prefix() -> None:
    assert ablation_name("data/synthetic/phase_3n_camera_jitter_only") == "camera_jitter_only"
