import csv
import json
from pathlib import Path
import zlib

import pytest

from training.measurements import body_ai_inference as inference
from training.train_candidate_model import (
    CandidateTrainingError,
    DEFAULT_TARGET_COLUMNS,
    train_candidate_model,
)


def test_candidate_training_creates_artifacts_metrics_and_registry(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, dataset_version="v1", count=8)
    output_dir = tmp_path / "artifacts" / "candidate"

    result = train_candidate_model(
        dataset_root,
        output_dir,
        dataset_version="v1",
        model_version="candidate_model_v1",
        random_seed=7,
        generated_at="2026-06-08T00:00:00Z",
    )

    model = json.loads(Path(result["model_path"]).read_text(encoding="utf-8"))
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    registry = json.loads(Path(result["registry_path"]).read_text(encoding="utf-8"))

    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["config_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert model["modelVersion"] == "candidate_model_v1"
    assert model["datasetVersion"] == "v1"
    assert model["trainingTimestamp"] == "2026-06-08T00:00:00Z"
    assert model["recordCount"] == 8
    assert model["candidateOnly"] is True
    assert model["isProduction"] is False
    assert model["imageUsage"]["pixelsConsumed"] is False
    assert config["randomSeed"] == 7
    assert config["datasetVersion"] == "v1"
    assert config["featurePipeline"]["usesImagePixels"] is False
    assert config["featurePipeline"]["validatedImageReferences"] == ["front", "side", "back"]
    assert metrics["sampleCounts"] == {"train": 4, "val": 2, "test": 2}
    assert metrics["metric"] == "mean_absolute_error_cm"
    assert set(metrics["test"]["maeByTarget"]) == set(DEFAULT_TARGET_COLUMNS)
    assert registry["productionModelUpdated"] is False
    assert registry["candidates"][0]["candidateStatus"] == "ready_for_evaluation"
    assert registry["candidates"][0]["productionStatus"] == "not_production"


def test_candidate_training_filters_requested_dataset_version(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, dataset_version="v1", count=4)
    _append_verified_records(dataset_root, dataset_version="v2", start=5, count=4)

    result = train_candidate_model(
        dataset_root,
        tmp_path / "candidate-v2",
        dataset_version="v2",
        model_version="candidate_model_v2",
        generated_at="2026-06-08T00:00:00Z",
    )

    assert result["model"]["datasetVersion"] == "v2"
    assert result["model"]["recordCount"] == 4
    assert result["training_config"]["datasetVersion"] == "v2"


def test_candidate_training_requires_dataset_version_when_multiple_versions_exist(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, dataset_version="v1", count=4)
    _append_verified_records(dataset_root, dataset_version="v2", start=5, count=4)

    with pytest.raises(CandidateTrainingError, match="Multiple dataset versions"):
        train_candidate_model(dataset_root, tmp_path / "candidate")


def test_candidate_registry_versions_increment_without_production_promotion(tmp_path: Path) -> None:
    dataset_root = _write_verified_dataset(tmp_path, dataset_version="v1", count=8)
    output_dir = tmp_path / "candidate"

    first = train_candidate_model(dataset_root, output_dir, dataset_version="v1", generated_at="2026-06-08T00:00:00Z")
    second = train_candidate_model(dataset_root, output_dir, dataset_version="v1", generated_at="2026-06-08T00:01:00Z")

    registry = json.loads(Path(second["registry_path"]).read_text(encoding="utf-8"))
    assert first["model"]["modelVersion"] == "candidate_model_v1"
    assert second["model"]["modelVersion"] == "candidate_model_v2"
    assert [entry["modelVersion"] for entry in registry["candidates"]] == ["candidate_model_v1", "candidate_model_v2"]
    assert registry["productionModelVersion"] is None
    assert registry["productionModelUpdated"] is False
    assert not (output_dir / "production_model.json").exists()


def test_candidate_training_does_not_change_live_inference_behavior(tmp_path: Path) -> None:
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
    dataset_root = _write_verified_dataset(tmp_path, dataset_version="v1", count=8)

    train_candidate_model(dataset_root, tmp_path / "candidate", dataset_version="v1", generated_at="2026-06-08T00:00:00Z")

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
    assert after["metadata"]["real_world_validated"] is False


def _write_verified_dataset(tmp_path: Path, *, dataset_version: str, count: int) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    records = _records(dataset_root, dataset_version=dataset_version, start=1, count=count)
    (dataset_root / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return dataset_root


def _append_verified_records(dataset_root: Path, *, dataset_version: str, start: int, count: int) -> None:
    payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload["records"].extend(_records(dataset_root, dataset_version=dataset_version, start=start, count=count))
    (dataset_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _records(dataset_root: Path, *, dataset_version: str, start: int, count: int) -> list[dict[str, object]]:
    records = []
    for index in range(start, start + count):
        sample_id = f"verified-{index:03d}"
        for view in ("front", "side", "back"):
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png")
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
                "verification_metadata_summary": {"verified": True, "maker_review_score": 0.9},
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
