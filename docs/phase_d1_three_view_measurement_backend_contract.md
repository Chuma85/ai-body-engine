# Phase D.1 Three-View Measurement Backend Contract

Phase D.1 adds a Body AI backend/API contract for the full mobile three-view scan package. The endpoint accepts front, side, and back views plus pose and validation metadata, but the current measurement estimator remains a compatibility path over the existing Phase 4H front/side-oriented synthetic-calibrated wrapper.

## Endpoint

`POST /v1/body-ai/measurements/three-view`

## Request

Required fields:

- `scanSessionId`
- `heightCm`
- `frontImage` or `frontImageStorageKey`
- `sideImage` or `sideImageStorageKey`
- `backImage` or `backImageStorageKey`

Optional fields:

- `weightKg`
- `requestPayloadVersion`
- `userId`
- `customerId`
- `orderId`
- `frontSourceType`, `sideSourceType`, `backSourceType`: `camera` or `upload`
- `frontPoseMetadata`, `sidePoseMetadata`, `backPoseMetadata`
- `frontValidationMetadata`, `sideValidationMetadata`, `backValidationMetadata`

Pose and validation metadata are flexible objects. The adapter recognizes common fields such as `confidenceScore`, `poseConfidence`, `qualityScore`, `validationScore`, `isValid`, `missingBodyRegions`, `warnings`, `errors`, and `retakeRecommendations`, while preserving room for mobile-specific metadata versions.

## Response

The response includes:

- `estimatedMeasurements`
- `perMeasurementConfidence`
- `overallScanConfidence`
- `scanQualitySummary`
- `poseSummary`
- `validationSummary`
- `engineVersion`
- `modelVersion`
- `realWorldValidationStatus: pending`
- `makerReviewRequired: true`
- `warnings`
- `errors`
- `compatibilityMode`
- `estimatorPath`
- `normalizedInputs`

## View Handling

All three views are required. The API rejects front-only and front+side payloads because back-view capture is part of the Phase D mobile scan contract.

The adapter normalizes each view into:

- source type
- inline image versus storage-key input kind
- image/storage-key completeness
- pose metadata availability
- validation metadata availability

The normalized package is retained in the in-process service store for traceability and future persistence wiring.

## Compatibility Limitation

The current estimator does not truly model-weight all three views. It still calls the Phase 4H Body AI inference wrapper, which validates a front and side image path and returns packaged synthetic-calibrated measurement predictions. To avoid a false production-accuracy claim, the Phase D.1 adapter:

- accepts and normalizes the back view
- logs that back view is present
- includes back-view quality in scan confidence
- returns explicit warnings that back view is not yet a learned model input
- keeps `makerReviewRequired` true
- keeps `realWorldValidationStatus` pending

This is an API and compatibility integration step, not a production measurement-accuracy upgrade.

## Confidence Calculation

Overall confidence is computed from:

- validation metadata quality, validity, warnings, errors, and missing body regions
- pose metadata confidence and missing body regions
- image/source completeness
- height availability

Pose metadata affects confidence through per-view pose scores and missing-region penalties. Validation metadata affects confidence through per-view validation scores, invalid-view penalties, warning/error penalties, missing-region penalties, and advisory retake recommendations.

Per-measurement confidence combines the current estimator's target confidence tier with the overall scan confidence. Targets that are unavailable or require manual input remain lower confidence even when scan quality is high.

## Logs

The adapter logs:

- received views and source types
- pose and validation metadata availability
- validation summary
- pose summary
- estimator/model path used
- confidence calculation basis

## Future Model-Training TODO

TODO: future true three-view weighting should replace the compatibility estimator with a model trained on synchronized front, side, and back captures plus pose/validation metadata. That model should consume the normalized Phase D payload directly, include back-view features in measurement prediction, and calibrate confidence against real-world validation data before any production-grade accuracy claims.
