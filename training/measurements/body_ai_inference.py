from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.measurement_confidence import DEFAULT_PHASE4D_PREDICTIONS, DEFAULT_RUN_NAME
from training.measurements.measurement_result_schema import (
    MeasurementConfidence,
    MeasurementInterval,
    MeasurementProductAction,
    MeasurementQualityFlag,
    MeasurementResult,
    MeasurementSource,
    MeasurementTargetResult,
    build_measurement_result_from_predictions,
    format_schema_summary,
    manual_user_input_result,
    write_json,
)
from training.measurements.measurement_targets import ProfileType, normalize_profile_type

SAMPLE_INFERENCE_RESULT_JSON = "sample_inference_result.json"
INFERENCE_SUMMARY_MD = "inference_wrapper_summary.md"
DEFAULT_OUTPUT_DIR = "artifacts/phase_4h_body_ai_inference_wrapper"
DEFAULT_SAMPLE_ID = "sample_000007"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class BodyAIInferenceError(ValueError):
    """Raised when an inference request is missing required local inputs."""


class BodyAIMeasurementService:
    def __init__(
        self,
        predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
        run_name: str = DEFAULT_RUN_NAME,
        model_version: str | None = None,
        pipeline_version: str | None = None,
    ) -> None:
        self.predictions_csv = Path(predictions_csv)
        self.run_name = run_name
        self.model_version = model_version
        self.pipeline_version = pipeline_version

    def predict(
        self,
        scan_id: str,
        front_image_path: str | Path,
        side_image_path: str | Path,
        height_cm: float | None = None,
        profile_type: str = ProfileType.UNSPECIFIED.value,
        user_id: str | None = None,
        order_id: str | None = None,
        demo_sample_id: str | None = None,
        generated_at: str | None = None,
    ) -> MeasurementResult:
        validate_inference_inputs(front_image_path, side_image_path)
        if not self.predictions_csv.exists():
            raise FileNotFoundError(f"Local demo prediction artifact is unavailable: {self.predictions_csv}")
        sample_id = demo_sample_id or sample_id_from_scan(scan_id) or DEFAULT_SAMPLE_ID
        result = build_measurement_result_from_predictions(
            self.predictions_csv,
            run_name=self.run_name,
            sample_id=sample_id,
            profile_type=profile_type,
            generated_at=generated_at,
        )
        result = self._apply_request_context(result, scan_id, user_id, order_id, height_cm)
        return self._apply_version_overrides(result)

    def predict_from_normalized_references(
        self,
        scan_id: str,
        height_cm: float | None = None,
        profile_type: str = ProfileType.UNSPECIFIED.value,
        user_id: str | None = None,
        order_id: str | None = None,
        demo_sample_id: str | None = None,
        generated_at: str | None = None,
    ) -> MeasurementResult:
        """Run the current demo estimator after an API layer has validated scan references."""
        if not self.predictions_csv.exists():
            raise FileNotFoundError(f"Local demo prediction artifact is unavailable: {self.predictions_csv}")
        sample_id = demo_sample_id or sample_id_from_scan(scan_id) or DEFAULT_SAMPLE_ID
        result = build_measurement_result_from_predictions(
            self.predictions_csv,
            run_name=self.run_name,
            sample_id=sample_id,
            profile_type=profile_type,
            generated_at=generated_at,
        )
        result = self._apply_request_context(result, scan_id, user_id, order_id, height_cm)
        return self._apply_version_overrides(result)

    def _apply_request_context(
        self,
        result: MeasurementResult,
        scan_id: str,
        user_id: str | None,
        order_id: str | None,
        height_cm: float | None,
    ) -> MeasurementResult:
        normalized_profile_type = normalize_profile_type(result.profile_type)
        targets = [height_result_with_input(target, height_cm) if target.target == "height" else target for target in result.targets]
        caveats = [
            *result.caveats,
            "Phase 4H local demo mode uses packaged synthetic artifact predictions until production model artifacts are wired.",
        ]
        if user_id:
            caveats.append(f"user_id: {user_id}")
        if order_id:
            caveats.append(f"order_id: {order_id}")
        return replace(
            result,
            result_id=f"body_ai_measurement_{scan_id}",
            sample_id=scan_id,
            targets=targets,
            profile_type=normalized_profile_type,
            caveats=caveats,
        )

    def _apply_version_overrides(self, result: MeasurementResult) -> MeasurementResult:
        metadata = result.metadata
        if self.model_version is not None:
            metadata = replace(metadata, model_version=self.model_version)
        if self.pipeline_version is not None:
            metadata = replace(metadata, pipeline_version=self.pipeline_version)
        return replace(result, metadata=metadata)


def run_body_ai_measurement(
    scan_id: str,
    front_image_path: str | Path,
    side_image_path: str | Path,
    height_cm: float | None = None,
    profile_type: str = ProfileType.UNSPECIFIED.value,
    user_id: str | None = None,
    order_id: str | None = None,
    predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
    run_name: str = DEFAULT_RUN_NAME,
    model_version: str | None = None,
    pipeline_version: str | None = "phase_4h_body_ai_inference_wrapper",
    demo_sample_id: str | None = None,
    generated_at: str | None = None,
) -> MeasurementResult:
    service = BodyAIMeasurementService(
        predictions_csv=predictions_csv,
        run_name=run_name,
        model_version=model_version,
        pipeline_version=pipeline_version,
    )
    return service.predict(
        scan_id=scan_id,
        user_id=user_id,
        order_id=order_id,
        front_image_path=front_image_path,
        side_image_path=side_image_path,
        height_cm=height_cm,
        profile_type=profile_type,
        demo_sample_id=demo_sample_id,
        generated_at=generated_at,
    )


def validate_inference_inputs(front_image_path: str | Path, side_image_path: str | Path) -> None:
    front = Path(front_image_path)
    side = Path(side_image_path)
    if not front.exists():
        raise BodyAIInferenceError(f"Missing front image: {front}")
    if not side.exists():
        raise BodyAIInferenceError(f"Missing side image: {side}")
    if not front.is_file():
        raise BodyAIInferenceError(f"Front image path is not a file: {front}")
    if not side.is_file():
        raise BodyAIInferenceError(f"Side image path is not a file: {side}")
    if front.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise BodyAIInferenceError(f"Front image path must be a PNG or JPEG file: {front}")
    if side.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise BodyAIInferenceError(f"Side image path must be a PNG or JPEG file: {side}")


def sample_id_from_scan(scan_id: str) -> str | None:
    if scan_id.startswith("sample_"):
        return scan_id
    return None


def height_result_with_input(target: MeasurementTargetResult, height_cm: float | None) -> MeasurementTargetResult:
    if height_cm is None:
        return manual_user_input_result("height")
    if height_cm <= 0:
        raise BodyAIInferenceError(f"height_cm must be positive when provided, got {height_cm}")
    estimate = round(float(height_cm), 4)
    return MeasurementTargetResult(
        target="height",
        estimate_cm=estimate,
        interval=MeasurementInterval(low_cm=estimate, high_cm=estimate, estimated_error_cm=0.0),
        confidence_tier=MeasurementConfidence.NOT_APPLICABLE,
        product_action=MeasurementProductAction.USER_INPUT_REQUIRED,
        source=MeasurementSource.MANUAL_USER_INPUT_REQUIRED,
        geometry_estimate_cm=None,
        residual_correction_cm=None,
        quality_flags=[MeasurementQualityFlag.USER_INPUT_REQUIRED],
        notes=[
            "Height was supplied by the caller and remains user-input-sourced rather than AI-estimated.",
            *target.notes,
        ],
    )


def to_dict(result: MeasurementResult) -> dict[str, Any]:
    return result.to_payload()


def to_json(result: MeasurementResult) -> str:
    return json.dumps(to_dict(result), indent=2, sort_keys=True) + "\n"


def save_result_json(result: MeasurementResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, to_dict(result))


def export_sample_inference_result(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    front_image_path: str | Path = "data/synthetic/phase_3t/images/front/sample_000001_front.png",
    side_image_path: str | Path = "data/synthetic/phase_3t/images/side/sample_000001_side.png",
    scan_id: str = "sample_000007",
    height_cm: float | None = 172.0,
    profile_type: str = ProfileType.UNSPECIFIED.value,
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = run_body_ai_measurement(
        scan_id=scan_id,
        front_image_path=front_image_path,
        side_image_path=side_image_path,
        height_cm=height_cm,
        profile_type=profile_type,
        generated_at=generated_at,
    )
    sample_json = output_path / SAMPLE_INFERENCE_RESULT_JSON
    summary_md = output_path / INFERENCE_SUMMARY_MD
    save_result_json(result, sample_json)
    summary_md.write_text(format_inference_summary(result), encoding="utf-8")
    return {
        "sample_inference_result_json": str(sample_json),
        "inference_wrapper_summary_md": str(summary_md),
        "payload": to_dict(result),
    }


def format_inference_summary(result: MeasurementResult) -> str:
    lines = [
        "# Phase 4H Body AI Inference Wrapper",
        "",
        f"Scan: `{result.sample_id}`",
        f"Pipeline: `{result.metadata.pipeline_version}`",
        f"Synthetic calibrated only: `{result.metadata.synthetic_calibrated_only}`",
        f"Real-world validated: `{result.metadata.real_world_validated}`",
        "",
        "## Measurement Targets",
        "",
        format_schema_summary(result),
        "## Inference Caveat",
        "",
        "This wrapper currently runs in deterministic local/demo mode using Phase 4D/4F artifacts. It is not a production model server.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local/demo Body AI measurement inference wrapper.")
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--front-image", required=True)
    parser.add_argument("--side-image", required=True)
    parser.add_argument("--height-cm", type=float)
    parser.add_argument("--profile-type", default=ProfileType.UNSPECIFIED.value, choices=[profile.value for profile in ProfileType])
    parser.add_argument("--user-id")
    parser.add_argument("--order-id")
    parser.add_argument("--predictions", default=DEFAULT_PHASE4D_PREDICTIONS)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_body_ai_measurement(
        scan_id=args.scan_id,
        user_id=args.user_id,
        order_id=args.order_id,
        front_image_path=args.front_image,
        side_image_path=args.side_image,
        height_cm=args.height_cm,
        profile_type=args.profile_type,
        predictions_csv=args.predictions,
        run_name=args.run_name,
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    save_result_json(result, output_path / SAMPLE_INFERENCE_RESULT_JSON)
    (output_path / INFERENCE_SUMMARY_MD).write_text(format_inference_summary(result), encoding="utf-8")
    print(f"Inference result: {output_path / SAMPLE_INFERENCE_RESULT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
