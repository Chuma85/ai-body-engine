from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

from training.measurements.measurement_confidence import (
    DEFAULT_PHASE4D_PREDICTIONS,
    DEFAULT_RUN_NAME,
    apply_confidence_policy,
    load_prediction_rows,
)
from training.measurements.measurement_uncertainty import (
    apply_uncertainty_policy,
    fit_uncertainty_policy,
)

SAMPLE_RESULT_JSON = "sample_measurement_result.json"
SCHEMA_SUMMARY_MD = "measurement_schema_summary.md"

SUPPORTED_TARGETS = [
    "chest",
    "waist",
    "hip",
    "thigh",
    "shoulder",
    "calf",
    "height",
    "inseam",
    "neck",
    "sleeve",
]
AI_RESIDUAL_TARGETS = {"chest", "waist", "hip", "thigh"}
LANDMARK_REQUIRED_TARGETS = {"inseam", "neck", "sleeve"}
UNAVAILABLE_TARGETS = {"shoulder", "calf"}
THIGH_CALIBRATION_RISK_NOTE = "Phase 4F synthetic interval coverage for thigh_cm was 0.8400, so thigh requires conservative review."


class MeasurementConfidence(str, Enum):
    HIGH = "high_confidence"
    MEDIUM = "medium_confidence"
    LOW = "low_confidence"
    NOT_APPLICABLE = "not_applicable"


class MeasurementQualityFlag(str, Enum):
    OK = "ok"
    SYNTHETIC_CALIBRATED_ONLY = "synthetic_calibrated_only"
    REAL_WORLD_NOT_VALIDATED = "real_world_not_validated"
    LARGE_RESIDUAL_CORRECTION = "large_residual_correction"
    GEOMETRY_QUALITY_FLAG = "geometry_quality_flag"
    LANDMARK_REQUIRED = "landmark_required"
    USER_INPUT_REQUIRED = "user_input_required"
    UNAVAILABLE = "unavailable"
    CALIBRATION_RISK = "calibration_risk"


class MeasurementProductAction(str, Enum):
    ACCEPT_AS_AI_ESTIMATE = "accept_as_ai_estimate"
    REQUIRE_MANUAL_CONFIRMATION = "require_manual_confirmation"
    REQUEST_RETAKE_OR_TAPE_MEASUREMENT = "request_retake_or_tape_measurement"
    USER_INPUT_REQUIRED = "user_input_required"
    MAKER_REVIEW_REQUIRED = "maker_review_required"


class MeasurementSource(str, Enum):
    AI_GEOMETRY_RESIDUAL = "ai_geometry_residual"
    AI_GEOMETRY_ONLY = "ai_geometry_only"
    MANUAL_USER_INPUT_REQUIRED = "manual_user_input_required"
    MANUAL_MAKER_VERIFIED = "manual_maker_verified"
    LANDMARK_REQUIRED = "landmark_required"
    UNAVAILABLE = "unavailable"


class ReadinessLevel(str, Enum):
    SYNTHETIC_CALIBRATED_RESEARCH = "synthetic_calibrated_research"
    ASSISTED_MEASUREMENT_REVIEW = "assisted_measurement_review"
    REAL_WORLD_VALIDATED = "real_world_validated"


@dataclass(frozen=True)
class MeasurementInterval:
    low_cm: float | None
    high_cm: float | None
    estimated_error_cm: float | None

    def validate(self, estimate_cm: float | None) -> None:
        if estimate_cm is None:
            return
        if self.low_cm is None or self.high_cm is None or self.estimated_error_cm is None:
            raise ValueError("AI measurement intervals require low, high, and estimated_error values.")
        if self.low_cm > estimate_cm or self.high_cm < estimate_cm:
            raise ValueError(
                f"Invalid interval: expected low <= estimate <= high, got {self.low_cm}, {estimate_cm}, {self.high_cm}."
            )


@dataclass(frozen=True)
class MeasurementTargetResult:
    target: str
    estimate_cm: float | None
    interval: MeasurementInterval
    confidence_tier: MeasurementConfidence
    product_action: MeasurementProductAction
    source: MeasurementSource
    geometry_estimate_cm: float | None = None
    residual_correction_cm: float | None = None
    quality_flags: list[MeasurementQualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_TARGETS:
            raise ValueError(f"Unsupported measurement target: {self.target}")
        if not self.product_action:
            raise ValueError(f"Measurement target {self.target} requires a product action.")
        is_ai_prediction = self.source in {MeasurementSource.AI_GEOMETRY_RESIDUAL, MeasurementSource.AI_GEOMETRY_ONLY}
        if is_ai_prediction and self.confidence_tier == MeasurementConfidence.NOT_APPLICABLE:
            raise ValueError(f"AI measurement target {self.target} requires a confidence tier.")
        if is_ai_prediction:
            self.interval.validate(self.estimate_cm)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence_tier"] = self.confidence_tier.value
        payload["product_action"] = self.product_action.value
        payload["source"] = self.source.value
        payload["quality_flags"] = [flag.value for flag in self.quality_flags]
        return payload


@dataclass(frozen=True)
class MeasurementModelMetadata:
    model_version: str
    pipeline_version: str
    calibration_version: str
    training_dataset_id: str
    readiness_level: ReadinessLevel = ReadinessLevel.SYNTHETIC_CALIBRATED_RESEARCH
    synthetic_calibrated_only: bool = True
    real_world_validated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.real_world_validated and self.synthetic_calibrated_only:
            raise ValueError("real_world_validated cannot be true while synthetic_calibrated_only is true.")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_level"] = self.readiness_level.value
        return payload


@dataclass(frozen=True)
class MeasurementResult:
    result_id: str
    sample_id: str
    dataset_split: str
    targets: list[MeasurementTargetResult]
    metadata: MeasurementModelMetadata
    caveats: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        target_names = [target.target for target in self.targets]
        if target_names != SUPPORTED_TARGETS:
            raise ValueError(f"Measurement result must include supported targets in stable order: {SUPPORTED_TARGETS}")
        if self.metadata.real_world_validated:
            raise ValueError("Body AI measurement results must default to real_world_validated=false until real validation exists.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "sample_id": self.sample_id,
            "dataset_split": self.dataset_split,
            "targets": [target.to_payload() for target in self.targets],
            "metadata": self.metadata.to_payload(),
            "caveats": list(self.caveats),
        }


def export_sample_measurement_result(
    predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
    output_dir: str | Path = "artifacts/phase_4g_measurement_result_contract",
    run_name: str = DEFAULT_RUN_NAME,
    sample_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = build_measurement_result_from_predictions(predictions_csv, run_name, sample_id, generated_at=generated_at)
    payload = result.to_payload()
    write_json(output_path / SAMPLE_RESULT_JSON, payload)
    (output_path / SCHEMA_SUMMARY_MD).write_text(format_schema_summary(result), encoding="utf-8")
    return {
        "sample_measurement_result_json": str(output_path / SAMPLE_RESULT_JSON),
        "measurement_schema_summary_md": str(output_path / SCHEMA_SUMMARY_MD),
        "payload": payload,
    }


def build_measurement_result_from_predictions(
    predictions_csv: str | Path = DEFAULT_PHASE4D_PREDICTIONS,
    run_name: str = DEFAULT_RUN_NAME,
    sample_id: str | None = None,
    generated_at: str | None = None,
) -> MeasurementResult:
    raw_rows = [row for row in load_prediction_rows(predictions_csv) if row["run_name"] == run_name]
    if not raw_rows:
        raise ValueError(f"No prediction rows found for run '{run_name}' in {predictions_csv}")
    confidence_rows = [apply_confidence_policy(row) for row in raw_rows]
    calibration_rows = [row for row in confidence_rows if row["dataset_split"] in {"train", "val"}]
    if not calibration_rows:
        raise ValueError("Need train/val rows to build uncertainty policy for measurement result export.")
    policy = fit_uncertainty_policy(calibration_rows)
    evaluated_rows = [apply_uncertainty_policy(row, policy) for row in confidence_rows]
    selected_sample_id = sample_id or first_sample_id(evaluated_rows)
    sample_rows = [row for row in evaluated_rows if row["sample_id"] == selected_sample_id]
    if not sample_rows:
        raise ValueError(f"No rows found for sample_id '{selected_sample_id}' in run '{run_name}'")
    row_by_target = {normalize_target_name(row["target"]): row for row in sample_rows}
    metadata = MeasurementModelMetadata(
        model_version="geometry_plus_residual_gradient_boosting",
        pipeline_version="phase_4g_measurement_result_contract",
        calibration_version="phase_4f_train_val_p90_intervals",
        training_dataset_id="phase_3t_realistic_1000",
        generated_at=generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    targets = [target_result_for_name(target, row_by_target.get(target)) for target in SUPPORTED_TARGETS]
    return MeasurementResult(
        result_id=f"body_ai_measurement_{selected_sample_id}",
        sample_id=selected_sample_id,
        dataset_split=str(sample_rows[0]["dataset_split"]),
        targets=targets,
        metadata=metadata,
        caveats=[
            "Synthetic calibrated-label validation only; not real-world production readiness.",
            "Manual confirmation is required before custom garment cutting decisions.",
            "Maker-only ease and allowance should be applied downstream, not baked into body measurements.",
        ],
    )


def first_sample_id(rows: list[dict[str, Any]]) -> str:
    test_rows = [row for row in rows if row.get("dataset_split") == "test"]
    selected_rows = test_rows or rows
    return str(sorted({row["sample_id"] for row in selected_rows})[0])


def target_result_for_name(target: str, row: dict[str, Any] | None) -> MeasurementTargetResult:
    if row is not None:
        return ai_geometry_residual_result(target, row)
    if target == "height":
        return manual_user_input_result(target)
    if target in LANDMARK_REQUIRED_TARGETS:
        return landmark_required_result(target)
    return unavailable_result(target)


def ai_geometry_residual_result(target: str, row: dict[str, Any]) -> MeasurementTargetResult:
    notes = ["AI estimate from front/side geometry estimate plus learned residual correction."]
    flags = flags_from_row(row)
    if target == "thigh":
        notes.append(THIGH_CALIBRATION_RISK_NOTE)
        flags.append(MeasurementQualityFlag.CALIBRATION_RISK)
    return MeasurementTargetResult(
        target=target,
        estimate_cm=round_float(row["final_estimate_cm"]),
        interval=MeasurementInterval(
            low_cm=round_float(row["prediction_interval_low_cm"]),
            high_cm=round_float(row["prediction_interval_high_cm"]),
            estimated_error_cm=round_float(row["estimated_error_cm"]),
        ),
        confidence_tier=MeasurementConfidence(str(row["confidence_tier"])),
        product_action=MeasurementProductAction(str(row["uncertainty_product_action"])),
        source=MeasurementSource.AI_GEOMETRY_RESIDUAL,
        geometry_estimate_cm=round_float(row["geometry_estimate_cm"]),
        residual_correction_cm=round_float(row["predicted_residual_cm"]),
        quality_flags=dedupe_flags(flags),
        notes=notes,
    )


def manual_user_input_result(target: str) -> MeasurementTargetResult:
    return MeasurementTargetResult(
        target=target,
        estimate_cm=None,
        interval=MeasurementInterval(None, None, None),
        confidence_tier=MeasurementConfidence.NOT_APPLICABLE,
        product_action=MeasurementProductAction.USER_INPUT_REQUIRED,
        source=MeasurementSource.MANUAL_USER_INPUT_REQUIRED,
        quality_flags=[MeasurementQualityFlag.USER_INPUT_REQUIRED],
        notes=["Height defaults to user input until a real-world height capture path is implemented."],
    )


def landmark_required_result(target: str) -> MeasurementTargetResult:
    return MeasurementTargetResult(
        target=target,
        estimate_cm=None,
        interval=MeasurementInterval(None, None, None),
        confidence_tier=MeasurementConfidence.NOT_APPLICABLE,
        product_action=MeasurementProductAction.REQUIRE_MANUAL_CONFIRMATION,
        source=MeasurementSource.LANDMARK_REQUIRED,
        quality_flags=[MeasurementQualityFlag.LANDMARK_REQUIRED],
        notes=[f"{target} requires an explicit landmark, proportion, or maker-confirmed measurement strategy."],
    )


def unavailable_result(target: str) -> MeasurementTargetResult:
    return MeasurementTargetResult(
        target=target,
        estimate_cm=None,
        interval=MeasurementInterval(None, None, None),
        confidence_tier=MeasurementConfidence.NOT_APPLICABLE,
        product_action=MeasurementProductAction.MAKER_REVIEW_REQUIRED,
        source=MeasurementSource.UNAVAILABLE,
        quality_flags=[MeasurementQualityFlag.UNAVAILABLE],
        notes=[f"{target} is present in the contract but is not produced by the Phase 4D geometry residual pipeline."],
    )


def flags_from_row(row: dict[str, Any]) -> list[MeasurementQualityFlag]:
    flags = [MeasurementQualityFlag.SYNTHETIC_CALIBRATED_ONLY, MeasurementQualityFlag.REAL_WORLD_NOT_VALIDATED]
    for field in ("geometry_quality_flags", "confidence_flags"):
        raw = str(row.get(field, ""))
        for token in raw.split(";"):
            flag = token.strip()
            if not flag or flag == "ok":
                continue
            flags.append(flag_for_token(flag))
    if len(flags) == 2:
        flags.append(MeasurementQualityFlag.OK)
    return dedupe_flags(flags)


def flag_for_token(token: str) -> MeasurementQualityFlag:
    mapping = {
        "large_residual_correction": MeasurementQualityFlag.LARGE_RESIDUAL_CORRECTION,
        "geometry_quality_flag": MeasurementQualityFlag.GEOMETRY_QUALITY_FLAG,
    }
    return mapping.get(token, MeasurementQualityFlag.GEOMETRY_QUALITY_FLAG if token.startswith("geometry_") else MeasurementQualityFlag.LARGE_RESIDUAL_CORRECTION)


def dedupe_flags(flags: list[MeasurementQualityFlag]) -> list[MeasurementQualityFlag]:
    return list(dict.fromkeys(flags))


def normalize_target_name(target: str) -> str:
    return target.removesuffix("_cm")


def round_float(value: Any) -> float:
    return round(float(value), 4)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def format_schema_summary(result: MeasurementResult) -> str:
    payload = result.to_payload()
    lines = [
        "# Phase 4G Measurement Result Contract",
        "",
        f"Sample: `{result.sample_id}`",
        f"Readiness: `{payload['metadata']['readiness_level']}`",
        f"Real-world validated: `{payload['metadata']['real_world_validated']}`",
        "",
        "| Target | Source | Action | Confidence | Estimate | Interval |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for target in payload["targets"]:
        interval = target["interval"]
        if target["estimate_cm"] is None:
            estimate = ""
            interval_text = ""
        else:
            estimate = f"{target['estimate_cm']:.4f}"
            interval_text = f"{interval['low_cm']:.4f} to {interval['high_cm']:.4f}"
        lines.append(
            f"| {target['target']} | {target['source']} | {target['product_action']} | {target['confidence_tier']} | {estimate} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "Synthetic calibrated-label validation only. Manual confirmation remains required before production garment decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a stable Body AI measurement result contract sample.")
    parser.add_argument("--predictions", default=DEFAULT_PHASE4D_PREDICTIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--sample-id")
    args = parser.parse_args(argv)
    result = export_sample_measurement_result(args.predictions, args.output, args.run_name, args.sample_id)
    print(f"Sample measurement result: {result['sample_measurement_result_json']}")
    print(f"Schema summary: {result['measurement_schema_summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
