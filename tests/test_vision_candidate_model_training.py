import csv
import json
from pathlib import Path
import zlib

import pytest

from training.measurements import body_ai_inference as inference
from training.train_candidate_model import DEFAULT_TARGET_COLUMNS


VISION_CONFIG_JSON = "vision_training_config.json"
VISION_METRICS_JSON = "vision_training_metrics.json"
VISION_MODEL_JSON = "vision_model.json"
VISION_MODEL_WEIGHTS = "vision_model.pt"
VISION_REGISTRY_JSON = "vision_candidate_model_registry.json"


def test_vision_candidate_training_writes_artifacts_metrics_and_registry(tmp_path: Path) -> None:
    trainer = _trainer()
    dataset_root = _write_multimodal_dataset(tmp_path, dataset_version="v1", count=8)
    output_dir = tmp_path / "artifacts" / "vision-candidate"

    result = trainer.train_vision_candidate_model(
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
        generated_at="2026-06-08T00:00:00Z",
    )

    model = json.loads(Path(result["model_json_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    registry = json.loads(Path(result["registry_path"]).read_text(encoding="utf-8"))

    assert (output_dir / VISION_MODEL_JSON).exists()
    assert (output_dir / VISION_MODEL_WEIGHTS).exists()
    assert (output_dir / VISION_CONFIG_JSON).exists()
    assert (output_dir / VISION_METRICS_JSON).exists()
    assert (output_dir / VISION_REGISTRY_JSON).exists()
    assert Path(result["report_path"]).exists()
    assert model["modelVersion"] == "vision_candidate_model_v1"
    assert model["candidateType"] == "vision_multimodal"
    assert model["datasetVersion"] == "v1"
    assert model["trainingTimestamp"] == "2026-06-08T00:00:00Z"
    assert model["recordCount"] == 8
    assert model["imageUsage"]["pixelsConsumed"] is True
    assert model["imageUsage"]["separateViewBranches"] is True
    assert model["candidateOnly"] is True
    assert model["isProduction"] is False
    assert model["architecture"]["frontImageEncoder"] == "ViewImageEncoder"
    assert model["architecture"]["sideImageEncoder"] == "ViewImageEncoder"
    assert model["architecture"]["backImageEncoder"] == "ViewImageEncoder"
    assert model["architecture"]["metadataFeatureEncoder"] == "MetadataEncoder"
    assert model["architecture"]["targetCount"] == len(DEFAULT_TARGET_COLUMNS)
    assert config["inputs"]["frontImageTensor"] is True
    assert config["inputs"]["sideImageTensor"] is True
    assert config["inputs"]["backImageTensor"] is True
    assert config["inputs"]["finalApprovedMeasurementsAsTargets"] is True
    assert config["inputs"]["finalApprovedMeasurementsAsInputs"] is False
    assert config["inputs"]["correctionDeltas"] is False
    assert metrics["sampleCounts"] == {"train": 4, "val": 2, "test": 2}
    assert metrics["metric"] == "mean_absolute_error_cm"
    assert metrics["pixelsConsumed"] is True
    for split in ("train", "val", "test"):
        assert "overallMae" in metrics[split]
        assert set(metrics[split]["maeByTarget"]) == set(DEFAULT_TARGET_COLUMNS)
    assert registry["productionModelUpdated"] is False
    assert registry["productionModelVersion"] is None
    assert registry["candidates"][0]["candidateType"] == "vision_multimodal"
    assert registry["candidates"][0]["pixelsConsumed"] is True
    assert registry["candidates"][0]["productionModelUpdated"] is False
    assert registry["candidates"][0]["readyForEvaluation"] is True


def test_multimodal_ready_dataset_is_required(tmp_path: Path) -> None:
    trainer = _trainer()
    dataset_root = _write_multimodal_dataset(tmp_path, dataset_version="v1", count=4)
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["records"][0]["verification_metadata_summary"] = {}
    (dataset_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(trainer.VisionCandidateTrainingError, match="multimodal_ready"):
        trainer.train_vision_candidate_model(
            dataset_root,
            tmp_path / "vision-candidate",
            dataset_version="v1",
            image_size=8,
            epochs=1,
            device="cpu",
        )


def test_missing_image_tensors_block_training_gate() -> None:
    trainer = _trainer()

    with pytest.raises(trainer.VisionCandidateTrainingError, match="front/side/back image tensors"):
        trainer.require_image_tensors(
            [
                {
                    "sampleId": "sample-without-front",
                    "frontImage": {"tensor": None},
                    "sideImage": {"tensor": object()},
                    "backImage": {"tensor": object()},
                }
            ]
        )


def test_separate_front_side_back_and_metadata_branches_are_constructed() -> None:
    import torch

    trainer = _trainer()
    model = trainer.VisionMultimodalRegressor(metadata_dim=3, target_count=2, branch_dim=4, fusion_dim=8)

    assert model.front_image_encoder is not model.side_image_encoder
    assert model.side_image_encoder is not model.back_image_encoder
    assert hasattr(model, "metadata_feature_encoder")
    assert hasattr(model, "fusion_layer")
    assert hasattr(model, "measurement_prediction_head")

    output = model(
        torch.ones((2, 3, 8, 8)),
        torch.ones((2, 3, 8, 8)) * 0.5,
        torch.ones((2, 3, 8, 8)) * 0.25,
        torch.ones((2, 3)),
    )
    assert tuple(output.shape) == (2, 2)


def test_leakage_controls_exclude_final_lineage_and_corrections(tmp_path: Path) -> None:
    trainer = _trainer()
    dataset_root = _write_multimodal_dataset(tmp_path, dataset_version="v1", count=8)
    result = trainer.train_vision_candidate_model(
        dataset_root,
        tmp_path / "vision-candidate",
        dataset_version="v1",
        random_seed=7,
        image_size=8,
        epochs=1,
        branch_dim=4,
        fusion_dim=8,
        device="cpu",
        generated_at="2026-06-08T00:00:00Z",
    )

    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    feature_blob = " ".join(config["metadataFeatureNames"]).lower()

    assert "final" not in feature_blob
    assert "customer" not in feature_blob
    assert "maker_adjustment" not in feature_blob
    assert "correction" not in feature_blob
    assert config["inputs"]["lineageMeasurements"] is False
    assert config["inputs"]["correctionDeltas"] is False
    assert config["inputs"]["finalApprovedMeasurementsAsInputs"] is False


def test_forbidden_metadata_feature_names_fail_fast(tmp_path: Path) -> None:
    trainer = _trainer()
    dataset_root = _write_multimodal_dataset(tmp_path, dataset_version="v1", count=4)
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["records"][0]["verification_metadata_summary"]["finalApprovedChestCm"] = 101.0
    (dataset_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(trainer.VisionCandidateTrainingError, match="Forbidden leakage-prone metadata features"):
        trainer.train_vision_candidate_model(
            dataset_root,
            tmp_path / "vision-candidate",
            dataset_version="v1",
            image_size=8,
            epochs=1,
            device="cpu",
        )


def test_vision_candidate_training_does_not_change_live_inference_behavior(tmp_path: Path) -> None:
    trainer = _trainer()
    predictions = _prediction_fixture(tmp_path)
    front, side = _inference_image_fixtures(tmp_path)
    before = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-06-08T00:00:00Z",
    ).to_payload()
    dataset_root = _write_multimodal_dataset(tmp_path, dataset_version="v1", count=8)

    trainer.train_vision_candidate_model(
        dataset_root,
        tmp_path / "vision-candidate",
        dataset_version="v1",
        image_size=8,
        epochs=1,
        branch_dim=4,
        fusion_dim=8,
        device="cpu",
        generated_at="2026-06-08T00:00:00Z",
    )

    after = inference.run_body_ai_measurement(
        "sample_003",
        front,
        side,
        height_cm=171.0,
        predictions_csv=predictions,
        generated_at="2026-06-08T00:00:00Z",
    ).to_payload()
    assert after == before
    assert after["metadata"]["real_world_validated"] is False


def _trainer():
    from training import train_vision_candidate_model

    return train_vision_candidate_model


def _write_multimodal_dataset(tmp_path: Path, *, dataset_version: str, count: int) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    records = _records(dataset_root, dataset_version=dataset_version, start=1, count=count)
    (dataset_root / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return dataset_root


def _records(dataset_root: Path, *, dataset_version: str, start: int, count: int) -> list[dict[str, object]]:
    records = []
    for index in range(start, start + count):
        sample_id = f"vision-{index:03d}"
        colors = {"front": (255, 0, index % 255, 255), "side": (0, 255, index % 255, 255), "back": (0, 0, 255, 255)}
        for view, rgba in colors.items():
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png", rgba=rgba)
        final = {
            target: round(70.0 + index + target_index * 1.75, 4)
            for target_index, target in enumerate(DEFAULT_TARGET_COLUMNS)
        }
        records.append(
            {
                "sample_id": sample_id,
                "dataset_version": dataset_version,
                "front_image_reference": f"images/{sample_id}-front.png",
                "side_image_reference": f"images/{sample_id}-side.png",
                "back_image_reference": f"images/{sample_id}-back.png",
                "pose_metadata_summary": {
                    "front": {"pose_confidence": 0.80 + index * 0.001, "missing_body_regions": []},
                    "side": {"pose_confidence": 0.78 + index * 0.001},
                    "back": {"pose_confidence": 0.76 + index * 0.001},
                },
                "validation_metadata_summary": {
                    "front": {"quality_score": 0.82 + index * 0.001, "is_valid": True},
                    "side": {"quality_score": 0.81 + index * 0.001, "warning_count": index % 2},
                    "back": {"quality_score": 0.79 + index * 0.001},
                },
                "verification_metadata_summary": {"verified": True, "review_score": 0.9},
                "lineage": {
                    "ai_estimate": {target: value - 1.0 for target, value in final.items()},
                    "customer_edit": {target: value - 0.5 for target, value in final.items()},
                    "maker_adjustment": {target: value for target, value in final.items()},
                    "final_approved": final,
                },
                "correction_deltas": {
                    target: round(final[target] - (final[target] - 1.0), 4)
                    for target in DEFAULT_TARGET_COLUMNS
                },
                "confidence_metadata": {"confidence_tier": "high_confidence" if index % 2 else "medium_confidence"},
                "eligibility_metadata": {"eligible_for_training": True, "holdout_candidate": index % 3 == 0},
            }
        )
    return records


def _inference_image_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    front.write_bytes(b"front")
    side.write_bytes(b"side")
    return front, side


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
