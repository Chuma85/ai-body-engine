import csv
import json
from pathlib import Path

from training.measurements import measurement_uncertainty as uncertainty


def test_interval_calculation_is_deterministic() -> None:
    rows = [
        _row("train", "chest_cm", "high_confidence", abs_error=1.0),
        _row("val", "chest_cm", "high_confidence", abs_error=3.0),
        _row("train", "chest_cm", "low_confidence", abs_error=7.0),
        _row("val", "chest_cm", "low_confidence", abs_error=9.0),
    ]

    confidence_rows = [_with_confidence(row) for row in rows]
    first = uncertainty.fit_uncertainty_policy(confidence_rows)
    second = uncertainty.fit_uncertainty_policy(confidence_rows)

    assert first == second
    assert first["target_tier_error_stats"]["chest_cm"]["low_confidence"]["p90"] > first["target_tier_error_stats"]["chest_cm"]["high_confidence"]["p90"]


def test_test_split_is_not_used_for_fitting_thresholds() -> None:
    calibration_rows = [_with_confidence(_row("train", abs_error=1.0)), _with_confidence(_row("val", abs_error=2.0))]
    with_test_rows = [*calibration_rows, _with_confidence(_row("test", abs_error=100.0))]

    policy_from_calibration = uncertainty.fit_uncertainty_policy(calibration_rows)
    policy_from_filtered = uncertainty.fit_uncertainty_policy([row for row in with_test_rows if row["dataset_split"] in uncertainty.CALIBRATION_SPLITS])

    assert policy_from_calibration == policy_from_filtered


def test_prediction_interval_contains_final_estimate() -> None:
    policy = uncertainty.fit_uncertainty_policy([_with_confidence(_row("train", abs_error=2.0)), _with_confidence(_row("val", abs_error=2.0))])

    evaluated = uncertainty.apply_uncertainty_policy(_with_confidence(_row("test", final_estimate=100.0, calibrated_label=101.0)), policy)

    assert evaluated["prediction_interval_low_cm"] <= 100.0 <= evaluated["prediction_interval_high_cm"]


def test_wider_confidence_buckets_produce_wider_intervals() -> None:
    rows = [
        _with_confidence(_row("train", "chest_cm", "high_confidence", abs_error=2.0)),
        _with_confidence(_row("val", "chest_cm", "high_confidence", abs_error=2.0)),
        _with_confidence(_row("train", "chest_cm", "low_confidence", abs_error=2.0)),
        _with_confidence(_row("val", "chest_cm", "low_confidence", abs_error=2.0)),
    ]
    policy = uncertainty.fit_uncertainty_policy(rows)

    high = uncertainty.apply_uncertainty_policy(_with_confidence(_row("test", "chest_cm", "high_confidence")), policy)
    low = uncertainty.apply_uncertainty_policy(_with_confidence(_row("test", "chest_cm", "low_confidence")), policy)

    assert low["estimated_error_cm"] > high["estimated_error_cm"]


def test_product_action_mapping_uses_interval_width() -> None:
    assert uncertainty.uncertainty_product_action("high_confidence", 2.0) == "accept_as_ai_estimate"
    assert uncertainty.uncertainty_product_action("high_confidence", 4.0) == "require_manual_confirmation"
    assert uncertainty.uncertainty_product_action("medium_confidence", 2.0) == "require_manual_confirmation"
    assert uncertainty.uncertainty_product_action("low_confidence", 2.0) == "request_retake_or_tape_measurement"
    assert uncertainty.uncertainty_product_action("high_confidence", 6.0) == "request_retake_or_tape_measurement"


def test_uncertainty_output_schema_is_stable(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    rows = [
        _row("train", "chest_cm", "high_confidence", abs_error=1.0),
        _row("val", "chest_cm", "high_confidence", abs_error=2.0),
        _row("test", "chest_cm", "high_confidence", abs_error=1.5),
        _row("train", "waist_cm", "low_confidence", abs_error=5.0),
        _row("val", "waist_cm", "low_confidence", abs_error=8.0),
        _row("test", "waist_cm", "low_confidence", abs_error=6.0),
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = uncertainty.calibrate_measurement_uncertainty(predictions_path, tmp_path / "out")

    for key in (
        "uncertainty_policy_json",
        "uncertainty_eval_json",
        "uncertainty_eval_csv",
        "per_target_summary_csv",
        "coverage_summary_md",
        "product_action_policy_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["uncertainty_eval_json"]).read_text(encoding="utf-8"))
    assert summary["calibration_splits"] == ["train", "val"]
    with Path(result["uncertainty_eval_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        output_rows = list(csv.DictReader(csv_file))
    assert {"target", "confidence_tier", "coverage", "mean_estimated_error_cm"} <= set(output_rows[0])


def _row(
    split: str,
    target: str = "chest_cm",
    tier: str = "high_confidence",
    abs_error: float = 1.0,
    final_estimate: float = 100.0,
    calibrated_label: float | None = None,
) -> dict[str, str | float]:
    calibrated = final_estimate + abs_error if calibrated_label is None else calibrated_label
    predicted_residual = 2.0 if tier == "high_confidence" else 10.0
    return {
        "sample_id": f"{split}_{target}_{tier}_{abs_error}",
        "dataset_split": split,
        "target": target,
        "model_name": "gradient_boosting",
        "run_name": "geometry_plus_residual__gradient_boosting",
        "geometry_estimate_cm": final_estimate - predicted_residual,
        "calibrated_label_cm": calibrated,
        "residual_cm": predicted_residual,
        "predicted_residual_cm": predicted_residual,
        "final_estimate_cm": final_estimate,
        "abs_error_cm": abs_error,
        "confidence_flags": "ok" if tier != "low_confidence" else "large_residual_correction",
        "geometry_quality_flags": "ok",
    }


def _with_confidence(row: dict[str, str | float]) -> dict[str, str | float]:
    from training.measurements.measurement_confidence import apply_confidence_policy

    return apply_confidence_policy(row)
