import numpy as np
import pytest

from training.experiments.analyze_feature_drift import (
    ablation_name,
    build_feature_drift_summary,
    feature_drift_rows,
    matched_feature_matrices,
)


def test_feature_drift_rows_calculate_stats() -> None:
    clean = np.asarray([[1.0, 10.0], [3.0, 14.0]], dtype=np.float64)
    current = np.asarray([[2.0, 9.0], [5.0, 13.0]], dtype=np.float64)

    rows = feature_drift_rows("lighting_only", clean, current, ["feature_a", "feature_b"])

    assert rows[0]["clean_mean"] == 2.0
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


def test_ablation_name_strips_phase_prefix() -> None:
    assert ablation_name("data/synthetic/phase_3n_camera_jitter_only") == "camera_jitter_only"
