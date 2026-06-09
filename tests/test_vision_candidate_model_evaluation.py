import csv
import json
from pathlib import Path
import zlib

from training.measurements import body_ai_inference as inference
from training.train_candidate_model import DEFAULT_TARGET_COLUMNS


def test_vision_evaluation_compares_production_metadata_and_vision_candidates(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
        generated_at="2026-06-09T00:00:00Z",
    )

    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert Path(result["benchmark_report_path"]).exists()
    assert Path(result["ablation_report_path"]).exists()
    assert Path(result["view_contribution_report_path"]).exists()
    assert Path(result["confidence_calibration_report_path"]).exists()
    assert Path(result["promotion_recommendation_path"]).exists()
    assert metrics["generatedAt"] == "2026-06-09T00:00:00Z"
    assert metrics["metadataCandidateModelVersion"] == "candidate_model_v1"
    assert metrics["visionCandidateModelVersion"] == "vision_candidate_model_v1"
    assert set(metrics["benchmark"]["testPerMeasurementMae"]) == set(DEFAULT_TARGET_COLUMNS)
    assert set(metrics["benchmark"]["testOverallMae"]) == {"productionBaseline", "metadataCandidate", "visionCandidate"}
    assert metrics["productionModelUpdated"] is False
    assert metrics["liveApiBehaviorChanged"] is False


def test_ablation_and_view_contribution_reports_are_generated(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
    )

    ablation = json.loads(Path(result["ablation_report_path"]).read_text(encoding="utf-8"))
    contribution = json.loads(Path(result["view_contribution_report_path"]).read_text(encoding="utf-8"))
    assert set(ablation["ablations"]) == {"metadata_only", "images_only", "images_metadata"}
    assert set(contribution["views"]) == {"front_only", "front_side", "front_side_back"}
    assert "backViewHelped" in contribution
    assert ablation["method"].startswith("evaluation-time branch masking")


def test_vision_leakage_audit_blocks_promotion(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()
    vision_model_path = Path(vision_training["model_json_path"])
    model = json.loads(vision_model_path.read_text(encoding="utf-8"))
    model["trainingConfig"]["inputs"]["correctionDeltas"] = True
    vision_model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_model_path,
        tmp_path / "evaluation",
    )

    assert result["leakage_audit"]["riskDetected"] is True
    assert result["recommendation"]["decision"] == "leakage_risk"
    assert result["recommendation"]["promoteAllowed"] is False


def test_split_duplicate_blocks_vision_promotion(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15, duplicate_identity=True)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
    )

    assert result["split_audit"]["valid"] is False
    fields = {finding["field"] for finding in result["split_audit"]["duplicateFindings"]}
    assert {"profileId", "scanSessionId", "orderId"} <= fields
    assert result["recommendation"]["promoteAllowed"] is False


def test_vision_regression_blocks_promotion(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()
    vision_model_path = Path(vision_training["model_json_path"])
    model = json.loads(vision_model_path.read_text(encoding="utf-8"))
    model["targetNormalization"]["mean"] = [1000.0 for _target in model["targetColumns"]]
    model["targetNormalization"]["std"] = [1.0 for _target in model["targetColumns"]]
    vision_model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_model_path,
        tmp_path / "evaluation",
    )

    assert result["metrics"]["benchmark"]["productionVsVision"]["overallRegression"] is True
    assert result["recommendation"]["decision"] == "regression_detected"
    assert result["recommendation"]["promoteAllowed"] is False


def test_insufficient_vision_test_data_blocks_promotion(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=8)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
        min_test_records=3,
    )

    assert result["recommendation"]["decision"] == "needs_more_data"
    assert result["recommendation"]["promoteAllowed"] is False


def test_promotion_recommendation_is_conservative(tmp_path: Path) -> None:
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    result = evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
    )

    recommendation = result["recommendation"]
    assert recommendation["decision"] in {
        "promote_candidate",
        "do_not_promote",
        "needs_more_data",
        "leakage_risk",
        "regression_detected",
        "confidence_not_calibrated",
    }
    if recommendation["promoteAllowed"]:
        assert recommendation["eligibilityCriteria"] == {
            "beatsProductionBaseline": True,
            "beatsMetadataCandidate": True,
            "noLeakageRisk": True,
            "splitIntegrityValid": True,
            "noMajorPerMeasurementRegression": True,
            "enoughTestRecords": True,
            "confidenceCalibrated": True,
        }


def test_vision_evaluation_does_not_change_live_inference_behavior(tmp_path: Path) -> None:
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
        generated_at="2026-06-09T00:00:00Z",
    ).to_payload()
    dataset_root = _write_multimodal_dataset(tmp_path, count=15)
    metadata_model_path = _write_metadata_candidate_model(tmp_path / "metadata-candidate")
    vision_training = _train_vision(dataset_root, tmp_path / "vision-candidate")
    evaluator = _evaluator()

    evaluator.evaluate_vision_candidate_model(
        dataset_root,
        metadata_model_path,
        vision_training["model_json_path"],
        tmp_path / "evaluation",
    )

    after = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-06-09T00:00:00Z",
    ).to_payload()
    assert after == before
    assert after["metadata"]["pipeline_version"] == "phase_4h_body_ai_inference_wrapper"


def _evaluator():
    from training import evaluate_vision_candidate_model

    return evaluate_vision_candidate_model


def _train_vision(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    from training.train_vision_candidate_model import train_vision_candidate_model

    return train_vision_candidate_model(
        dataset_root,
        output_dir,
        dataset_version="v1",
        model_version="vision_candidate_model_v1",
        random_seed=7,
        image_size=8,
        epochs=1,
        branch_dim=4,
        fusion_dim=8,
        device="cpu",
        generated_at="2026-06-09T00:00:00Z",
    )


def _write_metadata_candidate_model(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["image_reference.front_exists"]
    model = {
        "artifactType": "candidate_body_ai_measurement_model",
        "modelFamily": "verified_measurement_metadata_ridge_regressor",
        "modelVersion": "candidate_model_v1",
        "datasetVersion": "v1",
        "targetColumns": list(DEFAULT_TARGET_COLUMNS),
        "featureNames": feature_names,
        "featureMeans": [1.0],
        "featureStds": [1.0],
        "intercepts": [82.0 + index * 2.0 for index, _target in enumerate(DEFAULT_TARGET_COLUMNS)],
        "coefficients": [[0.0 for _target in DEFAULT_TARGET_COLUMNS]],
        "imageUsage": {"pixelsConsumed": False},
        "candidateOnly": True,
        "isProduction": False,
        "trainingConfig": {
            "datasetVersion": "v1",
            "randomSeed": 7,
            "splitPolicy": {"method": "deterministic_shuffle", "valSize": 0.2, "testSize": 0.2},
            "targetColumns": list(DEFAULT_TARGET_COLUMNS),
            "featurePipeline": {
                "usesImagePixels": False,
                "featureNames": feature_names,
            },
        },
    }
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model_path


def _write_multimodal_dataset(
    tmp_path: Path,
    *,
    count: int,
    duplicate_identity: bool = False,
) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(1, count + 1):
        sample_id = f"vision-eval-{index:03d}"
        colors = {"front": (255, index % 255, 0, 255), "side": (0, 255, index % 255, 255), "back": (index % 255, 0, 255, 255)}
        for view, rgba in colors.items():
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png", rgba=rgba)
        final = {
            target: round(70.0 + index * 0.8 + target_index * 2.25, 4)
            for target_index, target in enumerate(DEFAULT_TARGET_COLUMNS)
        }
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
                "verification_metadata_summary": {"verified": True, "review_score": 0.9},
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


def _write_png(path: Path, *, width: int = 8, height: int = 8, rgba: tuple[int, int, int, int] = (255, 0, 0, 255)) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + checksum.to_bytes(4, "big")

    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    pixel = bytes(rgba)
    raw_scanline = b"".join(b"\x00" + (pixel * width) for _row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_scanline))
        + chunk(b"IEND", b"")
    )
