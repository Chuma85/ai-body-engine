import csv
import json
from pathlib import Path
import zlib

from training.evaluate_candidate_model import evaluate_candidate_model
from training.measurements import body_ai_inference as inference
from training.train_candidate_model import DEFAULT_TARGET_COLUMNS, train_candidate_model


def test_candidate_evaluation_loads_candidate_and_writes_reports(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=15)
    training = train_candidate_model(
        dataset_root,
        tmp_path / "candidate",
        dataset_version="v1",
        generated_at="2026-06-08T00:00:00Z",
    )

    result = evaluate_candidate_model(
        dataset_root,
        training["model_path"],
        tmp_path / "evaluation",
        generated_at="2026-06-08T00:05:00Z",
    )

    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert Path(result["metrics_path"]).exists()
    assert Path(result["leakage_audit_path"]).exists()
    assert Path(result["split_audit_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert metrics["candidateModelVersion"] == "candidate_model_v1"
    assert metrics["baselineMetrics"]["test"]["overallMae"] >= 0
    assert metrics["candidateMetrics"]["test"]["overallMae"] >= 0
    assert set(metrics["comparison"]["perTarget"]) == set(DEFAULT_TARGET_COLUMNS)
    assert metrics["compatibilityMode"]["pixelsConsumed"] is False
    assert metrics["compatibilityMode"]["backImagePixelWeighted"] is False


def test_leakage_audit_flags_correction_delta_target_leakage(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=15, correction_delta_equals_target=True)
    training = train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1")

    result = evaluate_candidate_model(dataset_root, training["model_path"], tmp_path / "evaluation")

    leakage = result["leakage_audit"]
    assert leakage["riskDetected"] is True
    assert leakage["riskLevel"] == "high"
    assert result["recommendation"]["decision"] == "leakage_risk"
    assert any(finding["kind"] == "feature_equals_target" for finding in leakage["findings"])


def test_split_audit_catches_duplicate_profile_scan_and_order_ids(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=15, duplicate_identity=True)
    training = train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1")

    result = evaluate_candidate_model(dataset_root, training["model_path"], tmp_path / "evaluation")

    split_audit = result["split_audit"]
    assert split_audit["valid"] is False
    fields = {finding["field"] for finding in split_audit["duplicateFindings"]}
    assert {"profileId", "scanSessionId", "orderId"} <= fields
    assert result["recommendation"]["promoteAllowed"] is False


def test_insufficient_test_data_blocks_promotion(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=8)
    training = train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1")

    result = evaluate_candidate_model(dataset_root, training["model_path"], tmp_path / "evaluation", min_test_records=3)

    assert result["metrics"]["candidateMetrics"]["test"]
    assert result["recommendation"]["decision"] == "needs_more_data"
    assert result["recommendation"]["promoteAllowed"] is False


def test_regression_blocks_promotion_recommendation(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, count=15)
    training = train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1")
    model_path = Path(training["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["intercepts"] = [1000.0 for _target in model["targetColumns"]]
    model["coefficients"] = [[0.0 for _target in model["targetColumns"]] for _feature in model["featureNames"]]
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    result = evaluate_candidate_model(dataset_root, model_path, tmp_path / "evaluation")

    assert result["metrics"]["comparison"]["overallRegression"] is True
    assert result["recommendation"]["decision"] == "regression_detected"
    assert result["recommendation"]["promoteAllowed"] is False


def test_evaluation_does_not_change_live_inference_behavior(tmp_path: Path) -> None:
    predictions = _prediction_fixture(tmp_path)
    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    front.write_bytes(b"front")
    side.write_bytes(b"side")
    before = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-06-08T00:00:00Z",
    ).to_payload()
    dataset_root = _write_verified_dataset(tmp_path, count=15)
    training = train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1")

    evaluate_candidate_model(dataset_root, training["model_path"], tmp_path / "evaluation")

    after = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-06-08T00:00:00Z",
    ).to_payload()
    assert after == before
    assert after["metadata"]["pipeline_version"] == "phase_4h_body_ai_inference_wrapper"


def _write_verified_dataset(
    tmp_path: Path,
    *,
    count: int,
    correction_delta_equals_target: bool = False,
    duplicate_identity: bool = False,
) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(1, count + 1):
        sample_id = f"verified-{index:03d}"
        for view in ("front", "side", "back"):
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png")
        final = {
            target: round(70.0 + index * 0.8 + target_index * 2.25, 4)
            for target_index, target in enumerate(DEFAULT_TARGET_COLUMNS)
        }
        if correction_delta_equals_target:
            correction_deltas = dict(final)
        else:
            correction_deltas = {
                target: round(0.25 + (index % 3) * 0.1 + target_index * 0.01, 4)
                for target_index, target in enumerate(DEFAULT_TARGET_COLUMNS)
            }
        records.append(
            {
                "sample_id": sample_id,
                "dataset_version": "v1",
                "profileId": "duplicate-profile" if duplicate_identity else f"profile-{index:03d}",
                "scanSessionId": "duplicate-scan" if duplicate_identity else f"scan-{index:03d}",
                "orderId": "duplicate-order" if duplicate_identity else f"order-{index:03d}",
                "front_image_reference": f"images/{sample_id}-front.png",
                "side_image_reference": f"images/{sample_id}-side.png",
                "back_image_reference": f"images/{sample_id}-back.png",
                "pose_metadata_summary": {
                    "front": {"pose_confidence": 0.70 + (index % 5) * 0.01},
                    "side": {"pose_confidence": 0.68 + (index % 4) * 0.01},
                    "back": {"pose_confidence": 0.66 + (index % 3) * 0.01},
                },
                "validation_metadata_summary": {
                    "front": {"quality_score": 0.72 + (index % 4) * 0.01, "is_valid": True},
                    "side": {"quality_score": 0.71 + (index % 5) * 0.01},
                    "back": {"quality_score": 0.69 + (index % 6) * 0.01},
                },
                "verification_metadata_summary": {"verified": True, "maker_review_score": 0.9},
                "lineage": {
                    "ai_estimate": {target: value - correction_deltas[target] for target, value in final.items()},
                    "customer_edit": {target: value - 0.15 for target, value in final.items()},
                    "maker_adjustment": {target: value for target, value in final.items()},
                    "final_approved": final,
                },
                "correction_deltas": correction_deltas,
                "confidence_metadata": {"confidence_tier": _tier_for_index(index)},
                "eligibility_metadata": {"eligible_for_training": True},
            }
        )
    (dataset_root / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return dataset_root


def _tier_for_index(index: int) -> str:
    if index % 3 == 0:
        return "low_confidence"
    if index % 2 == 0:
        return "medium_confidence"
    return "high_confidence"


def _prediction_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.csv"
    rows = []
    for split, sample_id, error in (("train", "sample_001", 1.0), ("val", "sample_002", 2.0), ("test", "sample_003", 1.5)):
        for target in ("chest_cm", "waist_cm", "hip_cm", "thigh_cm"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_split": split,
                    "target": target,
                    "model_name": "gradient_boosting",
                    "run_name": "geometry_plus_residual__gradient_boosting",
                    "geometry_estimate_cm": "95.0",
                    "calibrated_label_cm": str(100.0 + error),
                    "residual_cm": "5.0",
                    "predicted_residual_cm": "5.0",
                    "final_estimate_cm": "100.0",
                    "abs_error_cm": str(error),
                    "confidence_flags": "ok",
                    "geometry_quality_flags": "ok",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_png(path: Path) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")

    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    raw_scanline = b"\x00\x00\x00\x00\xff"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )
