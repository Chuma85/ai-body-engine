from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from training.datasets.pose_feature_dataset import build_pose_feature_dataset


def test_valid_pose_metadata_produces_features_and_accepts_null_segmentation(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "raw" / "capture_sessions"
    output_path = tmp_path / "data" / "processed" / "pose_features.csv"
    _write_capture(input_dir / "front.json", _pose_record("front", segmentation_available=False))

    result = build_pose_feature_dataset(input_dir, output_path)

    rows = _read_csv(output_path)
    rejected_rows = _read_csv(result.rejected_report_path)
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert rejected_rows == []
    assert rows[0]["participant_id"] == "participant-1"
    assert rows[0]["view"] == "front"
    assert float(rows[0]["shoulder_width_proxy"]) > 0
    assert float(rows[0]["hip_width_proxy"]) > 0
    assert float(rows[0]["landmark_completeness_score"]) == 1.0
    assert rows[0]["front_width_features_available"] == "True"
    assert rows[0]["silhouette_height_px"] == ""
    assert rows[0]["side_depth_proxy_px"] == ""


def test_missing_landmarks_are_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "captures"
    output_path = tmp_path / "processed" / "pose_features.csv"
    record = _pose_record("front")
    record["normalized_landmarks"] = record["normalized_landmarks"][:27]
    _write_capture(input_dir / "missing-landmarks.json", record)

    result = build_pose_feature_dataset(input_dir, output_path)

    assert result.accepted_count == 0
    assert result.rejected_count == 1
    rows = _read_csv(result.rejected_report_path)
    assert "required_landmarks_missing:left_ankle|right_ankle" in rows[0]["rejection_reasons"]


def test_low_quality_score_is_flagged_in_rejected_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "captures"
    output_path = tmp_path / "processed" / "pose_features.csv"
    record = _pose_record("front")
    record["pose_quality_score"] = 0.4
    _write_capture(input_dir / "low-quality.json", record)

    result = build_pose_feature_dataset(input_dir, output_path, quality_threshold=0.75)

    assert result.accepted_count == 0
    assert result.rejected_count == 1
    rows = _read_csv(result.rejected_report_path)
    assert rows[0]["rejection_reasons"] == "low_pose_quality_score"


def test_side_and_back_views_do_not_produce_front_only_features(tmp_path: Path) -> None:
    input_dir = tmp_path / "captures"
    output_path = tmp_path / "processed" / "pose_features.csv"
    _write_capture(input_dir / "session.json", {"captures": [_pose_record("side"), _pose_record("back")]})

    build_pose_feature_dataset(input_dir, output_path)

    rows = {row["view"]: row for row in _read_csv(output_path)}
    assert rows["side"]["front_width_features_available"] == "False"
    assert rows["side"]["width_features_available"] == "False"
    assert rows["side"]["shoulder_width_proxy"] == ""
    assert rows["side"]["side_profile_features_available"] == "True"

    assert rows["back"]["front_width_features_available"] == "False"
    assert rows["back"]["back_width_features_available"] == "True"
    assert rows["back"]["symmetry_features_available"] == "False"
    assert float(rows["back"]["shoulder_width_proxy"]) > 0


def test_manual_capture_without_training_approval_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "captures"
    output_path = tmp_path / "processed" / "pose_features.csv"
    record = _pose_record("front")
    record["manual_capture"] = True
    record["approved_for_training"] = False
    _write_capture(input_dir / "manual.json", record)

    result = build_pose_feature_dataset(input_dir, output_path)

    assert result.accepted_count == 0
    rows = _read_csv(result.rejected_report_path)
    assert rows[0]["rejection_reasons"] == "manual_capture_without_admin_approval"


def _pose_record(view: str, *, segmentation_available: bool = True) -> dict[str, Any]:
    return {
        "participant_id": "participant-1",
        "tester_id": "tester-1",
        "capture_session_id": f"session-{view}",
        "view": view,
        "image_path": f"captures/{view}.jpg",
        "timestamp": "2026-06-22T12:00:00Z",
        "pose_quality_score": 0.92,
        "quality_reasons": ["stable_pose"],
        "normalized_landmarks": _landmarks(),
        "world_landmarks": _landmarks(),
        "segmentation_available": segmentation_available,
        "manual_capture": False,
        "approved_for_training": True,
    }


def _landmarks() -> list[dict[str, float]]:
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99} for _index in range(33)]
    overrides = {
        0: (0.5, 0.10),
        11: (0.30, 0.25),
        12: (0.70, 0.25),
        13: (0.25, 0.45),
        14: (0.75, 0.45),
        15: (0.20, 0.65),
        16: (0.80, 0.65),
        23: (0.35, 0.55),
        24: (0.65, 0.55),
        25: (0.36, 0.78),
        26: (0.64, 0.78),
        27: (0.37, 0.98),
        28: (0.63, 0.98),
    }
    for index, (x, y) in overrides.items():
        landmarks[index]["x"] = x
        landmarks[index]["y"] = y
    return landmarks


def _write_capture(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
