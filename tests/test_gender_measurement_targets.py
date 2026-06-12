import csv
import json
from pathlib import Path
import zlib

from synthetic.generator.generate_dataset import generate_dataset
from synthetic.generator.validate_dataset import validate_dataset
from training.datasets.verified_measurement_dataset import VerifiedMeasurementDatasetLoader
from training.measurements.measurement_targets import (
    GENDER_MEASUREMENT_SCHEMA_VERSION,
    ProfileType,
    target_available_for_profile,
    targets_for_profile,
)
from training.train_candidate_model import DEFAULT_TARGET_COLUMNS, train_candidate_model
from training.train_vision_candidate_model import require_target_coverage


def test_profile_target_availability_keeps_gender_specific_fields_optional() -> None:
    assert target_available_for_profile("bust_cm", ProfileType.FEMALE.value) is True
    assert target_available_for_profile("bust_cm", ProfileType.MALE.value) is False
    assert target_available_for_profile("jacket_length_cm", ProfileType.MALE.value) is True
    assert target_available_for_profile("jacket_length_cm", ProfileType.FEMALE.value) is False
    assert "abdomen_cm" in targets_for_profile(ProfileType.UNSPECIFIED.value)
    assert "wrist_cm" in targets_for_profile(ProfileType.UNSPECIFIED.value)
    assert "sleeve_shoulder_to_wrist_cm" in targets_for_profile(ProfileType.UNSPECIFIED.value)


def test_synthetic_generation_emits_gender_schema_and_capture_variation(tmp_path: Path) -> None:
    labels_csv = generate_dataset(count=4, output_dir=str(tmp_path / "synthetic"), width=96, height=144)
    rows = list(csv.DictReader(labels_csv.open("r", newline="", encoding="utf-8")))

    assert {row["profile_type"] for row in rows} == {"male", "female"}
    assert {row["measurement_schema_version"] for row in rows} == {GENDER_MEASUREMENT_SCHEMA_VERSION}
    assert validate_dataset(str(labels_csv))["valid"] is True

    male = next(row for row in rows if row["profile_type"] == "male")
    female = next(row for row in rows if row["profile_type"] == "female")
    assert male["bust_cm"] == ""
    assert male["jacket_length_cm"] != ""
    assert female["bust_cm"] != ""
    assert female["jacket_length_cm"] == ""
    for field in ("abdomen_cm", "stomach_cm", "wrist_cm", "thigh_cm", "calf_cm", "ankle_cm", "knee_cm"):
        assert male[field] != ""
        assert female[field] != ""
    for field in ("camera_angle_degrees", "camera_distance_m", "body_rotation_degrees", "phone_framing_scale"):
        assert rows[0][field] != ""


def test_verified_loader_accepts_gender_schema_and_reports_profile_availability(tmp_path: Path) -> None:
    dataset_root = _write_gender_verified_dataset(tmp_path, count=4)
    loader = VerifiedMeasurementDatasetLoader(dataset_root)

    assert loader.statistics()["dataset_versions"] == {GENDER_MEASUREMENT_SCHEMA_VERSION: 4}
    assert loader.statistics()["profile_type_distribution"] == {"male": 2, "female": 2}
    male = next(sample for sample in loader if sample["profile_type"] == "male")
    female = next(sample for sample in loader if sample["profile_type"] == "female")
    assert "bust_cm" in male["target_availability"]["missingAvailableTargets"] or "bust_cm" not in male["target_availability"]["availableTargets"]
    assert "bust_cm" in female["target_availability"]["presentTargets"]


def test_candidate_training_does_not_require_female_targets_for_male_profiles(tmp_path: Path) -> None:
    dataset_root = _write_gender_verified_dataset(tmp_path, count=10)

    result = train_candidate_model(
        dataset_root,
        tmp_path / "candidate",
        dataset_version=GENDER_MEASUREMENT_SCHEMA_VERSION,
        model_version="candidate_model_v1",
        target_columns=["chest_cm", "bust_cm"],
        random_seed=3,
    )

    assert result["model"]["candidateOnly"] is True
    assert result["model"]["isProduction"] is False
    assert set(result["metrics"]["test"]["maeByTarget"]) == {"chest_cm", "bust_cm"}


def test_vision_target_coverage_skips_profile_incompatible_targets() -> None:
    samples = [
        {"sampleId": "m1", "profileType": "male", "finalApprovedMeasurements": {"chest_cm": 96.0}},
        {"sampleId": "m2", "profileType": "male", "finalApprovedMeasurements": {"chest_cm": 98.0}},
        {"sampleId": "f1", "profileType": "female", "finalApprovedMeasurements": {"chest_cm": 90.0, "bust_cm": 96.0}},
        {"sampleId": "f2", "profileType": "female", "finalApprovedMeasurements": {"chest_cm": 92.0, "bust_cm": 99.0}},
    ]

    require_target_coverage(samples, ["chest_cm", "bust_cm"])


def _write_gender_verified_dataset(tmp_path: Path, *, count: int) -> Path:
    dataset_root = tmp_path / "verified"
    (dataset_root / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(1, count + 1):
        profile_type = "male" if index % 2 else "female"
        sample_id = f"gender-{index:03d}"
        for view in ("front", "side", "back"):
            _write_png(dataset_root / "images" / f"{sample_id}-{view}.png")
        final = {target: round(70.0 + index + target_index * 1.5, 4) for target_index, target in enumerate(DEFAULT_TARGET_COLUMNS)}
        if profile_type == "female":
            final["bust_cm"] = round(94.0 + index, 4)
            final["underbust_cm"] = round(82.0 + index, 4)
        else:
            final["jacket_length_cm"] = round(68.0 + index, 4)
        records.append(
            {
                "sample_id": sample_id,
                "dataset_version": GENDER_MEASUREMENT_SCHEMA_VERSION,
                "profileType": profile_type,
                "front_image_reference": f"images/{sample_id}-front.png",
                "side_image_reference": f"images/{sample_id}-side.png",
                "back_image_reference": f"images/{sample_id}-back.png",
                "pose_metadata_summary": {"front": {"pose_confidence": 0.9 + index * 0.001}},
                "validation_metadata_summary": {"front": {"quality_score": 0.88 + index * 0.001}},
                "verification_metadata_summary": {"verified": True},
                "lineage": {
                    "ai_estimate": {target: value - 1.0 for target, value in final.items()},
                    "customer_edit": {target: value - 0.5 for target, value in final.items()},
                    "maker_adjustment": final,
                    "final_approved": final,
                },
                "correction_deltas": {target: 1.0 for target in final},
                "confidence_metadata": {"confidence_tier": "high_confidence"},
                "eligibility_metadata": {"eligible_for_training": True},
            }
        )
    (dataset_root / "manifest.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return dataset_root


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
