import csv
import json
from pathlib import Path

from training.measurements import measurement_confidence as confidence


def test_confidence_tier_assignment_is_deterministic() -> None:
    row = _prediction_row(predicted_residual_cm=2.0, final_estimate_cm=100.0)

    first = confidence.apply_confidence_policy(row)
    second = confidence.apply_confidence_policy(row)

    assert first["confidence_tier"] == "high_confidence"
    assert first == second


def test_large_residual_triggers_low_confidence() -> None:
    row = _prediction_row(predicted_residual_cm=12.0, final_estimate_cm=100.0)

    evaluated = confidence.apply_confidence_policy(row)

    assert evaluated["confidence_tier"] == "low_confidence"
    assert evaluated["product_action"] == "request_retake_or_tape_measurement"
    assert "large_absolute_residual" in evaluated["confidence_reasons"]


def test_missing_or_invalid_geometry_triggers_low_confidence() -> None:
    row = _prediction_row(predicted_residual_cm=1.0, final_estimate_cm=100.0, geometry_quality_flags="missing_height")

    evaluated = confidence.apply_confidence_policy(row)

    assert evaluated["confidence_tier"] == "low_confidence"
    assert evaluated["product_action"] == "request_retake_or_tape_measurement"


def test_product_action_mapping_is_deterministic() -> None:
    assert confidence.product_action_for_tier("high_confidence") == "accept_as_ai_estimate"
    assert confidence.product_action_for_tier("medium_confidence") == "require_manual_confirmation"
    assert confidence.product_action_for_tier("low_confidence") == "request_retake_or_tape_measurement"


def test_confidence_evaluation_output_schema_is_stable(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    rows = [
        _prediction_row("sample_1", "chest_cm", predicted_residual_cm=2.0, final_estimate_cm=100.0, abs_error_cm=1.0),
        _prediction_row("sample_2", "chest_cm", predicted_residual_cm=6.0, final_estimate_cm=100.0, abs_error_cm=2.0),
        _prediction_row("sample_3", "chest_cm", predicted_residual_cm=12.0, final_estimate_cm=100.0, abs_error_cm=5.0),
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = confidence.evaluate_measurement_confidence(predictions_path, tmp_path / "out", run_name="geometry_plus_residual__gradient_boosting")

    for key in (
        "confidence_policy_json",
        "confidence_eval_json",
        "confidence_eval_csv",
        "per_target_confidence_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["confidence_eval_json"]).read_text(encoding="utf-8"))
    assert summary["policy"]["tiers"] == ["high_confidence", "medium_confidence", "low_confidence"]
    with Path(result["confidence_eval_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        output_rows = list(csv.DictReader(csv_file))
    assert {"target", "confidence_tier", "product_action", "prediction_count", "mae_cm"} <= set(output_rows[0])


def _prediction_row(
    sample_id: str = "sample_1",
    target: str = "chest_cm",
    predicted_residual_cm: float = 0.0,
    final_estimate_cm: float = 100.0,
    abs_error_cm: float = 1.0,
    geometry_quality_flags: str = "ok",
) -> dict[str, str | float]:
    return {
        "sample_id": sample_id,
        "dataset_split": "test",
        "target": target,
        "model_name": "gradient_boosting",
        "run_name": "geometry_plus_residual__gradient_boosting",
        "geometry_estimate_cm": 98.0,
        "calibrated_label_cm": 100.0,
        "residual_cm": 2.0,
        "predicted_residual_cm": predicted_residual_cm,
        "final_estimate_cm": final_estimate_cm,
        "abs_error_cm": abs_error_cm,
        "confidence_flags": "ok",
        "geometry_quality_flags": geometry_quality_flags,
    }
