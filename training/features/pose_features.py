from __future__ import annotations

from math import isfinite, sqrt
from typing import Any

from app.schemas.pose_metadata import CapturedPhotoPoseMetadata, LandmarkCollection, PoseLandmark, PoseView


MEDIAPIPE_POSE_LANDMARK_INDEXES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

REQUIRED_POSE_LANDMARKS = tuple(MEDIAPIPE_POSE_LANDMARK_INDEXES)

POSE_FEATURE_NAMES = [
    "shoulder_width_proxy",
    "hip_width_proxy",
    "torso_length_proxy",
    "leg_length_proxy",
    "arm_length_proxy",
    "head_to_ankle_height_proxy",
    "shoulder_to_hip_ratio",
    "hip_to_ankle_ratio",
    "shoulder_y_symmetry_delta",
    "hip_y_symmetry_delta",
    "ankle_y_symmetry_delta",
    "arm_length_symmetry_delta",
    "leg_length_symmetry_delta",
    "landmark_completeness_score",
    "width_features_available",
    "symmetry_features_available",
    "front_width_features_available",
    "back_width_features_available",
    "side_profile_features_available",
]

SEGMENTATION_FEATURE_NAMES = [
    "silhouette_height_px",
    "silhouette_width_px",
    "shoulder_width_px",
    "waist_width_px",
    "hip_width_px",
    "side_depth_proxy_px",
]

POSE_DATASET_FEATURE_NAMES = [*POSE_FEATURE_NAMES, *SEGMENTATION_FEATURE_NAMES]


def extract_pose_features(metadata: CapturedPhotoPoseMetadata) -> dict[str, float | bool | None]:
    landmarks = landmark_map(metadata.normalized_landmarks)
    view = metadata.view
    features: dict[str, float | bool | None] = {name: None for name in POSE_DATASET_FEATURE_NAMES}

    width_available = view in {PoseView.FRONT, PoseView.BACK} and all(
        landmarks.get(name) is not None for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    )
    side_profile_available = view == PoseView.SIDE
    symmetry_available = view == PoseView.FRONT and all(landmarks.get(name) is not None for name in REQUIRED_POSE_LANDMARKS)

    features.update(
        {
            "landmark_completeness_score": landmark_completeness_score(metadata.normalized_landmarks),
            "width_features_available": width_available,
            "symmetry_features_available": symmetry_available,
            "front_width_features_available": view == PoseView.FRONT and width_available,
            "back_width_features_available": view == PoseView.BACK and width_available,
            "side_profile_features_available": side_profile_available,
        }
    )

    shoulder_mid = _midpoint(landmarks.get("left_shoulder"), landmarks.get("right_shoulder"))
    hip_mid = _midpoint(landmarks.get("left_hip"), landmarks.get("right_hip"))
    ankle_mid = _midpoint(landmarks.get("left_ankle"), landmarks.get("right_ankle"))

    left_leg = _polyline_length(landmarks.get("left_hip"), landmarks.get("left_knee"), landmarks.get("left_ankle"))
    right_leg = _polyline_length(landmarks.get("right_hip"), landmarks.get("right_knee"), landmarks.get("right_ankle"))
    left_arm = _polyline_length(landmarks.get("left_shoulder"), landmarks.get("left_elbow"), landmarks.get("left_wrist"))
    right_arm = _polyline_length(landmarks.get("right_shoulder"), landmarks.get("right_elbow"), landmarks.get("right_wrist"))

    features["torso_length_proxy"] = _distance(shoulder_mid, hip_mid)
    features["leg_length_proxy"] = _average_present(left_leg, right_leg)
    features["arm_length_proxy"] = _average_present(left_arm, right_arm)
    features["head_to_ankle_height_proxy"] = _distance(landmarks.get("nose"), ankle_mid)
    features["hip_to_ankle_ratio"] = _safe_ratio(_distance(hip_mid, ankle_mid), features["head_to_ankle_height_proxy"])

    if width_available:
        shoulder_width = _distance(landmarks.get("left_shoulder"), landmarks.get("right_shoulder"))
        hip_width = _distance(landmarks.get("left_hip"), landmarks.get("right_hip"))
        features["shoulder_width_proxy"] = shoulder_width
        features["hip_width_proxy"] = hip_width
        features["shoulder_to_hip_ratio"] = _safe_ratio(shoulder_width, hip_width)

    if symmetry_available:
        features["shoulder_y_symmetry_delta"] = _axis_delta(landmarks["left_shoulder"], landmarks["right_shoulder"], "y")
        features["hip_y_symmetry_delta"] = _axis_delta(landmarks["left_hip"], landmarks["right_hip"], "y")
        features["ankle_y_symmetry_delta"] = _axis_delta(landmarks["left_ankle"], landmarks["right_ankle"], "y")
        features["arm_length_symmetry_delta"] = _absolute_delta(left_arm, right_arm)
        features["leg_length_symmetry_delta"] = _absolute_delta(left_leg, right_leg)

    features.update(segmentation_placeholder_features(metadata))
    return features


def missing_required_landmarks(metadata: CapturedPhotoPoseMetadata) -> list[str]:
    landmarks = landmark_map(metadata.normalized_landmarks)
    return [name for name in REQUIRED_POSE_LANDMARKS if landmarks.get(name) is None]


def landmark_completeness_score(collection: LandmarkCollection) -> float:
    landmarks = landmark_map(collection)
    present = sum(1 for name in REQUIRED_POSE_LANDMARKS if landmarks.get(name) is not None)
    return present / len(REQUIRED_POSE_LANDMARKS)


def landmark_map(collection: LandmarkCollection) -> dict[str, PoseLandmark]:
    if isinstance(collection, list):
        return {
            name: collection[index]
            for name, index in MEDIAPIPE_POSE_LANDMARK_INDEXES.items()
            if index < len(collection) and _usable_landmark(collection[index])
        }
    return {
        _normalize_landmark_name(name): landmark
        for name, landmark in collection.items()
        if _usable_landmark(landmark)
    }


def segmentation_placeholder_features(metadata: CapturedPhotoPoseMetadata) -> dict[str, float | None]:
    if metadata.segmentation_features is None:
        return {name: None for name in SEGMENTATION_FEATURE_NAMES}
    payload = metadata.segmentation_features.model_dump(by_alias=False)
    return {name: payload.get(name) for name in SEGMENTATION_FEATURE_NAMES}


def _usable_landmark(landmark: PoseLandmark | None) -> bool:
    if landmark is None:
        return False
    if not (isfinite(float(landmark.x)) and isfinite(float(landmark.y))):
        return False
    if landmark.visibility is not None and float(landmark.visibility) <= 0.0:
        return False
    if landmark.presence is not None and float(landmark.presence) <= 0.0:
        return False
    return True


def _normalize_landmark_name(name: str) -> str:
    normalized = name.replace("-", "_")
    result = ""
    for char in normalized:
        if char.isupper() and result:
            result += "_"
        result += char.lower()
    return result


def _distance(left: PoseLandmark | None, right: PoseLandmark | None) -> float | None:
    if left is None or right is None:
        return None
    dz = (left.z or 0.0) - (right.z or 0.0)
    return sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2 + dz**2)


def _midpoint(left: PoseLandmark | None, right: PoseLandmark | None) -> PoseLandmark | None:
    if left is None or right is None:
        return None
    return PoseLandmark(
        x=(left.x + right.x) / 2,
        y=(left.y + right.y) / 2,
        z=((left.z or 0.0) + (right.z or 0.0)) / 2,
    )


def _polyline_length(*points: PoseLandmark | None) -> float | None:
    total = 0.0
    for start, end in zip(points, points[1:]):
        segment = _distance(start, end)
        if segment is None:
            return None
        total += segment
    return total


def _average_present(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _axis_delta(left: PoseLandmark, right: PoseLandmark, axis: str) -> float:
    return abs(float(getattr(left, axis)) - float(getattr(right, axis)))


def _absolute_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))
