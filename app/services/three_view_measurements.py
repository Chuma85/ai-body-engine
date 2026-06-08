from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.schemas.three_view_measurements import (
    ConfidenceDetail,
    FlexibleMetadata,
    PoseSummary,
    ScanQualitySummary,
    SourceType,
    ThreeViewMeasurementRequest,
    ThreeViewMeasurementResponse,
    ValidationSummary,
    ViewInputSummary,
    ViewQualitySummary,
)
from training.measurements.body_ai_inference import BodyAIMeasurementService
from training.measurements.measurement_confidence import DEFAULT_PHASE4D_PREDICTIONS, DEFAULT_RUN_NAME


logger = logging.getLogger(__name__)

COMPATIBILITY_WARNING = (
    "Compatibility estimator accepts front, side, and back views, but the current packaged "
    "measurement model is still front/side-oriented and does not fully model-weight the back view yet."
)
POSE_WARNING = "Pose metadata is summarized and used for confidence scoring, but is not yet model-weighted by the estimator."
VALIDATION_WARNING = "Validation metadata is summarized and used for confidence scoring, but is not yet model-weighted by the estimator."
DEMO_ESTIMATOR_WARNING = (
    "Current estimator path uses packaged synthetic-calibrated predictions and does not claim production-grade accuracy."
)
FUTURE_MODEL_TRAINING_TODO = (
    "TODO: train and calibrate a real three-view model that consumes front, side, back, pose, and validation signals directly."
)


@dataclass(frozen=True)
class NormalizedView:
    name: str
    image: str | None
    storage_key: str | None
    source_type: SourceType
    pose_metadata: FlexibleMetadata | None
    validation_metadata: FlexibleMetadata | None

    @property
    def input_kind(self) -> str:
        if self.image and self.storage_key:
            return "image_and_storage_key"
        if self.image:
            return "image"
        return "storage_key"

    def input_summary(self) -> ViewInputSummary:
        return ViewInputSummary(
            sourceType=self.source_type,
            inputKind=self.input_kind,
            hasImage=self.image is not None,
            hasStorageKey=self.storage_key is not None,
            metadataAvailable=self.pose_metadata is not None or self.validation_metadata is not None,
        )


class ThreeViewMeasurementService:
    def __init__(
        self,
        predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
        run_name: str = DEFAULT_RUN_NAME,
        engine_version: str = "phase_d1_three_view_measurement_backend",
        pipeline_version: str = "phase_d1_three_view_compatibility_adapter",
    ) -> None:
        self.inference_service = BodyAIMeasurementService(
            predictions_csv=predictions_csv,
            run_name=run_name,
            pipeline_version=pipeline_version,
        )
        self.engine_version = engine_version
        self.normalized_scan_packages: dict[str, dict[str, Any]] = {}

    def generate(self, request: ThreeViewMeasurementRequest) -> ThreeViewMeasurementResponse:
        views = normalize_views(request)
        normalized_inputs = {name: view.input_summary() for name, view in views.items()}
        pose_summary = summarize_pose(views)
        validation_summary = summarize_validation(views)
        scan_quality_summary = summarize_scan_quality(views, pose_summary, validation_summary, normalized_inputs)
        overall_confidence = calculate_overall_confidence(request.height_cm, pose_summary, validation_summary, scan_quality_summary)

        self._persist_normalized_package(request, views, normalized_inputs, pose_summary, validation_summary, scan_quality_summary)
        self._log_received_package(request, views, pose_summary, validation_summary, overall_confidence)

        result = self.inference_service.predict_from_normalized_references(
            scan_id=request.scan_session_id,
            height_cm=request.height_cm,
            user_id=request.user_id or request.customer_id,
            order_id=request.order_id,
        )

        raw_payload = result.to_payload()
        estimated_measurements = {
            target["target"]: target["estimate_cm"]
            for target in raw_payload["targets"]
        }
        per_measurement_confidence = calculate_per_measurement_confidence(raw_payload["targets"], overall_confidence.score)
        warnings = build_warnings(pose_summary, validation_summary, scan_quality_summary)

        logger.info(
            "three_view_estimator_path scanSessionId=%s estimatorPath=%s modelVersion=%s compatibilityMode=%s",
            request.scan_session_id,
            "phase_4h_body_ai_inference_wrapper_via_three_view_adapter",
            raw_payload["metadata"]["model_version"],
            True,
        )

        return ThreeViewMeasurementResponse(
            scanSessionId=request.scan_session_id,
            measurementResultId=raw_payload["result_id"],
            estimatedMeasurements=estimated_measurements,
            perMeasurementConfidence=per_measurement_confidence,
            overallScanConfidence=overall_confidence,
            scanQualitySummary=scan_quality_summary,
            poseSummary=pose_summary,
            validationSummary=validation_summary,
            engineVersion=self.engine_version,
            modelVersion=raw_payload["metadata"]["model_version"],
            realWorldValidationStatus="pending",
            makerReviewRequired=True,
            compatibilityMode=True,
            estimatorPath="phase_4h_body_ai_inference_wrapper_via_three_view_adapter",
            normalizedInputs=normalized_inputs,
            warnings=warnings,
            errors=list(validation_summary.errors),
            rawMeasurementResult=raw_payload,
        )

    def _persist_normalized_package(
        self,
        request: ThreeViewMeasurementRequest,
        views: dict[str, NormalizedView],
        normalized_inputs: dict[str, ViewInputSummary],
        pose_summary: PoseSummary,
        validation_summary: ValidationSummary,
        scan_quality_summary: ScanQualitySummary,
    ) -> None:
        self.normalized_scan_packages[request.scan_session_id] = {
            "requestPayloadVersion": request.request_payload_version,
            "heightCm": request.height_cm,
            "weightKg": request.weight_kg,
            "userId": request.user_id,
            "customerId": request.customer_id,
            "orderId": request.order_id,
            "views": {name: normalized_view_payload(view) for name, view in views.items()},
            "normalizedInputs": {name: value.model_dump(by_alias=True) for name, value in normalized_inputs.items()},
            "poseSummary": pose_summary.model_dump(by_alias=True),
            "validationSummary": validation_summary.model_dump(by_alias=True),
            "scanQualitySummary": scan_quality_summary.model_dump(by_alias=True),
        }

    def _log_received_package(
        self,
        request: ThreeViewMeasurementRequest,
        views: dict[str, NormalizedView],
        pose_summary: PoseSummary,
        validation_summary: ValidationSummary,
        overall_confidence: ConfidenceDetail,
    ) -> None:
        logger.info(
            "three_view_package_received scanSessionId=%s views=%s sourceTypes=%s",
            request.scan_session_id,
            sorted(views),
            {name: view.source_type.value for name, view in views.items()},
        )
        logger.info(
            "three_view_metadata_availability scanSessionId=%s pose=%s validation=%s",
            request.scan_session_id,
            pose_summary.metadata_available,
            validation_summary.metadata_available,
        )
        logger.info(
            "three_view_validation_summary scanSessionId=%s overallValidationScore=%.4f errors=%s warnings=%s",
            request.scan_session_id,
            validation_summary.overall_validation_score,
            validation_summary.errors,
            validation_summary.warnings,
        )
        logger.info(
            "three_view_pose_summary scanSessionId=%s overallPoseConfidence=%.4f missingBodyRegions=%s",
            request.scan_session_id,
            pose_summary.overall_pose_confidence,
            pose_summary.missing_body_regions,
        )
        logger.info(
            "three_view_confidence_calculation scanSessionId=%s score=%.4f tier=%s basis=%s",
            request.scan_session_id,
            overall_confidence.score,
            overall_confidence.tier,
            overall_confidence.basis,
        )


def normalize_views(request: ThreeViewMeasurementRequest) -> dict[str, NormalizedView]:
    return {
        "front": NormalizedView(
            name="front",
            image=request.front_image,
            storage_key=request.front_image_storage_key,
            source_type=request.front_source_type,
            pose_metadata=request.front_pose_metadata,
            validation_metadata=request.front_validation_metadata,
        ),
        "side": NormalizedView(
            name="side",
            image=request.side_image,
            storage_key=request.side_image_storage_key,
            source_type=request.side_source_type,
            pose_metadata=request.side_pose_metadata,
            validation_metadata=request.side_validation_metadata,
        ),
        "back": NormalizedView(
            name="back",
            image=request.back_image,
            storage_key=request.back_image_storage_key,
            source_type=request.back_source_type,
            pose_metadata=request.back_pose_metadata,
            validation_metadata=request.back_validation_metadata,
        ),
    }


def normalized_view_payload(view: NormalizedView) -> dict[str, Any]:
    return {
        **view.input_summary().model_dump(by_alias=True),
        "storageKey": view.storage_key,
        "inlineImageAccepted": view.image is not None,
        "inlineImagePersisted": False,
        "poseMetadata": view.pose_metadata.model_dump(by_alias=True) if view.pose_metadata is not None else None,
        "validationMetadata": view.validation_metadata.model_dump(by_alias=True) if view.validation_metadata is not None else None,
    }


def summarize_pose(views: dict[str, NormalizedView]) -> PoseSummary:
    scores = {name: score_metadata(view.pose_metadata, default_score=0.65) for name, view in views.items()}
    warnings = [
        f"{name} pose metadata missing; confidence uses compatibility default."
        for name, view in views.items()
        if view.pose_metadata is None
    ]
    for name, view in views.items():
        if view.pose_metadata is not None:
            warnings.extend(prefix_messages(f"{name} pose", view.pose_metadata.warnings))
    return PoseSummary(
        frontPoseConfidence=round_score(scores["front"]),
        sidePoseConfidence=round_score(scores["side"]),
        backPoseConfidence=round_score(scores["back"]),
        overallPoseConfidence=round_score(mean_score(scores.values())),
        metadataAvailable={name: view.pose_metadata is not None for name, view in views.items()},
        missingBodyRegions={name: metadata_missing_regions(view.pose_metadata) for name, view in views.items()},
        warnings=warnings,
    )


def summarize_validation(views: dict[str, NormalizedView]) -> ValidationSummary:
    scores = {name: score_metadata(view.validation_metadata, default_score=0.65) for name, view in views.items()}
    warnings = [
        f"{name} validation metadata missing; confidence uses compatibility default."
        for name, view in views.items()
        if view.validation_metadata is None
    ]
    errors: list[str] = []
    for name, view in views.items():
        metadata = view.validation_metadata
        if metadata is None:
            continue
        warnings.extend(prefix_messages(f"{name} validation", metadata.warnings))
        errors.extend(prefix_messages(f"{name} validation", metadata.errors))
        if metadata.is_valid is False:
            warnings.append(f"{name} validation marked the scan view invalid; maker review and advisory retake remain required.")
    return ValidationSummary(
        frontValidationScore=round_score(scores["front"]),
        sideValidationScore=round_score(scores["side"]),
        backValidationScore=round_score(scores["back"]),
        overallValidationScore=round_score(mean_score(scores.values())),
        metadataAvailable={name: view.validation_metadata is not None for name, view in views.items()},
        missingBodyRegions={name: metadata_missing_regions(view.validation_metadata) for name, view in views.items()},
        warnings=warnings,
        errors=errors,
    )


def summarize_scan_quality(
    views: dict[str, NormalizedView],
    pose_summary: PoseSummary,
    validation_summary: ValidationSummary,
    normalized_inputs: dict[str, ViewInputSummary],
) -> ScanQualitySummary:
    view_qualities = {
        name: summarize_view_quality(name, view, pose_summary, validation_summary)
        for name, view in views.items()
    }
    overall_score = mean_score(quality.score for quality in view_qualities.values())
    retake_recommendations: list[str] = []
    warnings: list[str] = []
    missing_regions: list[str] = []
    for name, quality in view_qualities.items():
        retake_recommendations.extend(quality.retake_recommendations)
        warnings.extend(prefix_messages(name, quality.warnings))
        missing_regions.extend(f"{name}:{region}" for region in quality.missing_body_regions)
    overall_quality = ViewQualitySummary(
        score=round_score(overall_score),
        tier=tier_for_score(overall_score),
        sourceType=SourceType.CAMERA,
        missingBodyRegions=sorted(set(missing_regions)),
        retakeRecommendations=dedupe(retake_recommendations),
        warnings=dedupe(warnings),
    )
    return ScanQualitySummary(
        frontQuality=view_qualities["front"],
        sideQuality=view_qualities["side"],
        backQuality=view_qualities["back"],
        overallQuality=overall_quality,
        retakeRecommendations=dedupe(retake_recommendations),
        viewInputs=normalized_inputs,
    )


def summarize_view_quality(
    name: str,
    view: NormalizedView,
    pose_summary: PoseSummary,
    validation_summary: ValidationSummary,
) -> ViewQualitySummary:
    pose_score = getattr(pose_summary, f"{name}_pose_confidence")
    validation_score = getattr(validation_summary, f"{name}_validation_score")
    source_score = 1.0 if view.image or view.storage_key else 0.0
    if view.source_type == SourceType.UPLOAD:
        source_score -= 0.03
    score = clamp(pose_score * 0.35 + validation_score * 0.55 + source_score * 0.10)
    missing_regions = dedupe(
        [
            *pose_summary.missing_body_regions.get(name, []),
            *validation_summary.missing_body_regions.get(name, []),
        ]
    )
    retakes = retake_recommendations_for_view(name, score, missing_regions, view.validation_metadata)
    warnings = []
    if score < 0.70:
        warnings.append(f"{name} quality is below preferred threshold.")
    if view.source_type == SourceType.UPLOAD:
        warnings.append(f"{name} came from upload; camera capture is preferred when retake is feasible.")
    return ViewQualitySummary(
        score=round_score(score),
        tier=tier_for_score(score),
        sourceType=view.source_type,
        missingBodyRegions=missing_regions,
        retakeRecommendations=retakes,
        warnings=warnings,
    )


def calculate_overall_confidence(
    height_cm: float,
    pose_summary: PoseSummary,
    validation_summary: ValidationSummary,
    scan_quality_summary: ScanQualitySummary,
) -> ConfidenceDetail:
    height_score = 1.0 if height_cm > 0 else 0.0
    score = clamp(
        validation_summary.overall_validation_score * 0.45
        + pose_summary.overall_pose_confidence * 0.35
        + scan_quality_summary.overall_quality.score * 0.10
        + height_score * 0.10
    )
    basis = [
        f"validation={validation_summary.overall_validation_score:.4f}",
        f"pose={pose_summary.overall_pose_confidence:.4f}",
        f"scan_quality={scan_quality_summary.overall_quality.score:.4f}",
        "height=provided",
    ]
    return ConfidenceDetail(score=round_score(score), tier=tier_for_score(score), basis=basis)


def calculate_per_measurement_confidence(targets: list[dict[str, Any]], overall_scan_score: float) -> dict[str, ConfidenceDetail]:
    output: dict[str, ConfidenceDetail] = {}
    for target in targets:
        base = {
            "high_confidence": 0.84,
            "medium_confidence": 0.68,
            "low_confidence": 0.48,
            "not_applicable": 0.42,
        }.get(str(target["confidence_tier"]), 0.42)
        if target["estimate_cm"] is None:
            base = min(base, 0.40)
        score = clamp(base * 0.55 + overall_scan_score * 0.45)
        output[str(target["target"])] = ConfidenceDetail(
            score=round_score(score),
            tier=tier_for_score(score),
            basis=[
                f"target_tier={target['confidence_tier']}",
                f"overall_scan_confidence={overall_scan_score:.4f}",
                "maker_review_required=true",
            ],
        )
    return output


def build_warnings(
    pose_summary: PoseSummary,
    validation_summary: ValidationSummary,
    scan_quality_summary: ScanQualitySummary,
) -> list[str]:
    warnings = [
        COMPATIBILITY_WARNING,
        POSE_WARNING,
        VALIDATION_WARNING,
        DEMO_ESTIMATOR_WARNING,
        FUTURE_MODEL_TRAINING_TODO,
        "Back view is accepted, normalized, logged, and scored for scan quality, but it is not yet a learned model input.",
        "Maker review remains required before production use.",
        "Real-world validation status remains pending.",
    ]
    warnings.extend(pose_summary.warnings)
    warnings.extend(validation_summary.warnings)
    warnings.extend(scan_quality_summary.overall_quality.warnings)
    return dedupe(warnings)


def score_metadata(metadata: FlexibleMetadata | None, default_score: float) -> float:
    if metadata is None:
        return default_score
    score = first_score(metadata)
    if score is None:
        score = 0.85 if metadata.is_valid is True else 0.35 if metadata.is_valid is False else default_score
    score -= len(metadata_missing_regions(metadata)) * 0.08
    score -= len(metadata.errors) * 0.10
    score -= len(metadata.warnings) * 0.04
    if metadata.is_valid is False:
        score -= 0.20
    return clamp(score)


def first_score(metadata: FlexibleMetadata) -> float | None:
    for value in (
        metadata.confidence_score,
        metadata.pose_confidence,
        metadata.validation_score,
        metadata.quality_score,
        metadata.confidence,
    ):
        if value is not None:
            return float(value)
    return None


def metadata_missing_regions(metadata: FlexibleMetadata | None) -> list[str]:
    if metadata is None:
        return []
    return dedupe(str(region) for region in metadata.missing_body_regions)


def retake_recommendations_for_view(
    name: str,
    score: float,
    missing_regions: list[str],
    validation_metadata: FlexibleMetadata | None,
) -> list[str]:
    recommendations = []
    if validation_metadata is not None:
        recommendations.extend(validation_metadata.retake_recommendations)
    if score < 0.70:
        recommendations.append(f"Consider advisory retake for {name} view before maker review.")
    if missing_regions:
        recommendations.append(f"Check {name} view coverage for missing regions: {', '.join(missing_regions)}.")
    return dedupe(recommendations)


def prefix_messages(prefix: str, messages: list[str]) -> list[str]:
    return [f"{prefix}: {message}" for message in messages]


def tier_for_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def mean_score(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def round_score(score: float) -> float:
    return round(clamp(score), 4)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def dedupe(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
