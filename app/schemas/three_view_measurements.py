from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(str, Enum):
    CAMERA = "camera"
    UPLOAD = "upload"


class RealWorldValidationStatus(str, Enum):
    PENDING = "pending"


class FlexibleMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_score: float | None = Field(default=None, alias="confidenceScore", ge=0.0, le=1.0)
    pose_confidence: float | None = Field(default=None, alias="poseConfidence", ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, alias="qualityScore", ge=0.0, le=1.0)
    validation_score: float | None = Field(default=None, alias="validationScore", ge=0.0, le=1.0)
    is_valid: bool | None = Field(default=None, alias="isValid")
    missing_body_regions: list[str] = Field(default_factory=list, alias="missingBodyRegions")
    visible_body_regions: list[str] = Field(default_factory=list, alias="visibleBodyRegions")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    retake_recommendations: list[str] = Field(default_factory=list, alias="retakeRecommendations")


class ThreeViewMeasurementRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    scan_session_id: str = Field(..., alias="scanSessionId", min_length=1)
    height_cm: float = Field(..., alias="heightCm", gt=0)
    weight_kg: float | None = Field(default=None, alias="weightKg", gt=0)
    request_payload_version: str = Field(default="phase_d_mobile_three_view_v1", alias="requestPayloadVersion")

    user_id: str | None = Field(default=None, alias="userId")
    customer_id: str | None = Field(default=None, alias="customerId")
    order_id: str | None = Field(default=None, alias="orderId")

    front_image: str | None = Field(default=None, alias="frontImage")
    front_image_storage_key: str | None = Field(default=None, alias="frontImageStorageKey")
    side_image: str | None = Field(default=None, alias="sideImage")
    side_image_storage_key: str | None = Field(default=None, alias="sideImageStorageKey")
    back_image: str | None = Field(default=None, alias="backImage")
    back_image_storage_key: str | None = Field(default=None, alias="backImageStorageKey")

    front_source_type: SourceType = Field(default=SourceType.CAMERA, alias="frontSourceType")
    side_source_type: SourceType = Field(default=SourceType.CAMERA, alias="sideSourceType")
    back_source_type: SourceType = Field(default=SourceType.CAMERA, alias="backSourceType")
    source_types: dict[str, SourceType] | None = Field(default=None, alias="sourceTypes")

    front_pose_metadata: FlexibleMetadata | None = Field(default=None, alias="frontPoseMetadata")
    side_pose_metadata: FlexibleMetadata | None = Field(default=None, alias="sidePoseMetadata")
    back_pose_metadata: FlexibleMetadata | None = Field(default=None, alias="backPoseMetadata")
    front_validation_metadata: FlexibleMetadata | None = Field(default=None, alias="frontValidationMetadata")
    side_validation_metadata: FlexibleMetadata | None = Field(default=None, alias="sideValidationMetadata")
    back_validation_metadata: FlexibleMetadata | None = Field(default=None, alias="backValidationMetadata")

    @model_validator(mode="after")
    def require_three_views(self) -> Self:
        if self.source_types:
            for view in ("front", "side", "back"):
                if view in self.source_types:
                    setattr(self, f"{view}_source_type", self.source_types[view])
        missing = []
        for view in ("front", "side", "back"):
            if not getattr(self, f"{view}_image") and not getattr(self, f"{view}_image_storage_key"):
                missing.append(view)
        if missing:
            raise ValueError(f"Three-view measurement requires image or storage key for: {', '.join(missing)}")
        return self


class ViewInputSummary(BaseModel):
    source_type: SourceType = Field(..., alias="sourceType")
    input_kind: str = Field(..., alias="inputKind")
    has_image: bool = Field(..., alias="hasImage")
    has_storage_key: bool = Field(..., alias="hasStorageKey")
    metadata_available: bool = Field(..., alias="metadataAvailable")


class ViewQualitySummary(BaseModel):
    score: float
    tier: str
    source_type: SourceType = Field(..., alias="sourceType")
    missing_body_regions: list[str] = Field(default_factory=list, alias="missingBodyRegions")
    retake_recommendations: list[str] = Field(default_factory=list, alias="retakeRecommendations")
    warnings: list[str] = Field(default_factory=list)


class ScanQualitySummary(BaseModel):
    front_quality: ViewQualitySummary = Field(..., alias="frontQuality")
    side_quality: ViewQualitySummary = Field(..., alias="sideQuality")
    back_quality: ViewQualitySummary = Field(..., alias="backQuality")
    overall_quality: ViewQualitySummary = Field(..., alias="overallQuality")
    retake_recommendations: list[str] = Field(default_factory=list, alias="retakeRecommendations")
    view_inputs: dict[str, ViewInputSummary] = Field(default_factory=dict, alias="viewInputs")


class PoseSummary(BaseModel):
    front_pose_confidence: float = Field(..., alias="frontPoseConfidence")
    side_pose_confidence: float = Field(..., alias="sidePoseConfidence")
    back_pose_confidence: float = Field(..., alias="backPoseConfidence")
    overall_pose_confidence: float = Field(..., alias="overallPoseConfidence")
    metadata_available: dict[str, bool] = Field(default_factory=dict, alias="metadataAvailable")
    missing_body_regions: dict[str, list[str]] = Field(default_factory=dict, alias="missingBodyRegions")
    warnings: list[str] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    front_validation_score: float = Field(..., alias="frontValidationScore")
    side_validation_score: float = Field(..., alias="sideValidationScore")
    back_validation_score: float = Field(..., alias="backValidationScore")
    overall_validation_score: float = Field(..., alias="overallValidationScore")
    metadata_available: dict[str, bool] = Field(default_factory=dict, alias="metadataAvailable")
    missing_body_regions: dict[str, list[str]] = Field(default_factory=dict, alias="missingBodyRegions")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ConfidenceDetail(BaseModel):
    score: float
    tier: str
    basis: list[str] = Field(default_factory=list)


class ThreeViewMeasurementResponse(BaseModel):
    scan_session_id: str = Field(..., alias="scanSessionId")
    measurement_result_id: str = Field(..., alias="measurementResultId")
    estimated_measurements: dict[str, float | None] = Field(..., alias="estimatedMeasurements")
    per_measurement_confidence: dict[str, ConfidenceDetail] = Field(..., alias="perMeasurementConfidence")
    overall_scan_confidence: ConfidenceDetail = Field(..., alias="overallScanConfidence")
    scan_quality_summary: ScanQualitySummary = Field(..., alias="scanQualitySummary")
    pose_summary: PoseSummary = Field(..., alias="poseSummary")
    validation_summary: ValidationSummary = Field(..., alias="validationSummary")
    engine_version: str = Field(..., alias="engineVersion")
    model_version: str = Field(..., alias="modelVersion")
    real_world_validation_status: RealWorldValidationStatus = Field(
        default=RealWorldValidationStatus.PENDING,
        alias="realWorldValidationStatus",
    )
    maker_review_required: bool = Field(default=True, alias="makerReviewRequired")
    compatibility_mode: bool = Field(default=True, alias="compatibilityMode")
    estimator_path: str = Field(..., alias="estimatorPath")
    normalized_inputs: dict[str, ViewInputSummary] = Field(default_factory=dict, alias="normalizedInputs")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    raw_measurement_result: dict[str, Any] = Field(default_factory=dict, alias="rawMeasurementResult")
