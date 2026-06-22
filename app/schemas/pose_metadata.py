from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PoseView(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"


class PoseLandmark(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None


class SegmentationFeaturePlaceholder(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    silhouette_height_px: float | None = Field(default=None, alias="silhouetteHeightPx")
    silhouette_width_px: float | None = Field(default=None, alias="silhouetteWidthPx")
    shoulder_width_px: float | None = Field(default=None, alias="shoulderWidthPx")
    waist_width_px: float | None = Field(default=None, alias="waistWidthPx")
    hip_width_px: float | None = Field(default=None, alias="hipWidthPx")
    side_depth_proxy_px: float | None = Field(default=None, alias="sideDepthProxyPx")


LandmarkCollection = list[PoseLandmark] | dict[str, PoseLandmark]


class CapturedPhotoPoseMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    participant_id: str = Field(..., alias="participantId", min_length=1)
    tester_id: str = Field(..., alias="testerId", min_length=1)
    capture_session_id: str = Field(..., alias="captureSessionId", min_length=1)
    view: PoseView
    image_path: str = Field(..., alias="imagePath", min_length=1)
    timestamp: datetime
    pose_quality_score: float = Field(..., alias="poseQualityScore", ge=0)
    quality_reasons: list[str] = Field(default_factory=list, alias="qualityReasons")
    normalized_landmarks: LandmarkCollection = Field(..., alias="normalizedLandmarks")
    world_landmarks: LandmarkCollection | None = Field(default=None, alias="worldLandmarks")
    segmentation_available: bool = Field(default=False, alias="segmentationAvailable")
    manual_capture: bool = Field(default=False, alias="manualCapture")
    approved_for_training: bool = Field(default=False, alias="approvedForTraining")
    segmentation_features: SegmentationFeaturePlaceholder | None = Field(default=None, alias="segmentationFeatures")

    def metadata_fields(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "tester_id": self.tester_id,
            "capture_session_id": self.capture_session_id,
            "view": self.view.value,
            "image_path": self.image_path,
            "timestamp": self.timestamp.isoformat(),
            "pose_quality_score": self.pose_quality_score,
            "quality_reasons": "|".join(self.quality_reasons),
            "segmentation_available": self.segmentation_available,
            "manual_capture": self.manual_capture,
            "approved_for_training": self.approved_for_training,
        }
