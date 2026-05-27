import csv
import json
from pathlib import Path

from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from synthetic.validate_synthetic_dataset import validate_dataset
from training.experiments.build_phase3t_benchmark_report import (
    TARGET_COLUMNS,
    build_phase3t_benchmark_report,
    promotion_gate,
)


def test_phase3t_config_loads_with_realistic_controls() -> None:
    config_path = Path("synthetic/blender/configs/phase_3t_realistic_1000_config.example.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["sample_count"] == 1000
    assert config["output_dir"] == "data/synthetic/phase_3t"
    assert config["body_seed"] != config["render_seed"]
    assert config["variation_controls"]["enabled"] is True
    assert config["render_realism"]["enabled"] is True


def test_promotion_gate_logic_is_deterministic() -> None:
    assert promotion_gate(6.1)["gate"] == "research_only"
    assert promotion_gate(4.2)["gate"] == "assisted_sizing_manual_confirmation"
    assert promotion_gate(2.1)["gate"] == "stronger_production_candidate"


def test_validation_catches_missing_image_pair(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, count=3)
    missing_side = dataset_root / "images" / "side" / "sample_000002_side.png"
    missing_side.unlink()

    result = validate_dataset(dataset_root)

    assert result["valid"] is False
    assert "sample_000002" in result["unpaired_front_samples"]


def test_phase3t_report_writes_stable_outputs(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, count=12)
    manifest = build_dataset_manifest(dataset_root)
    assert manifest["valid"] is True
    classical_dir = _write_classical_artifacts(tmp_path)
    cnn_dir = _write_cnn_artifacts(tmp_path)

    result = build_phase3t_benchmark_report(
        dataset_root,
        tmp_path / "artifacts" / "phase_3t",
        classical_dir=classical_dir,
        cnn_dir=cnn_dir,
    )

    for key in (
        "dataset_validation_json",
        "dataset_validation_csv",
        "dataset_validation_md",
        "benchmark_results_json",
        "benchmark_results_csv",
        "benchmark_results_md",
        "per_target_results_csv",
        "promotion_readiness_md",
    ):
        assert Path(result[key]).exists()

    benchmark = json.loads(Path(result["benchmark_results_json"]).read_text(encoding="utf-8"))
    assert benchmark["benchmark_rows"][0]["test_mae"] <= benchmark["benchmark_rows"][-1]["test_mae"]
    with Path(result["per_target_results_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        target_rows = list(csv.DictReader(csv_file))
    assert {row["target"] for row in target_rows} == set(TARGET_COLUMNS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3t_fixture"
    front_dir = dataset_root / "images" / "front"
    side_dir = dataset_root / "images" / "side"
    labels_dir = dataset_root / "labels"
    front_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for index in range(1, count + 1):
            sample_id = f"sample_{index:06d}"
            _write_rect_image(front_dir / f"{sample_id}_front.png", (20, 8, 44, 58), (64, 64))
            _write_rect_image(side_dir / f"{sample_id}_side.png", (24, 8, 38, 58), (64, 64))
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(85 + index),
                    "waist_cm": str(70 + index),
                    "hip_cm": str(90 + index),
                    "shoulder_cm": str(40 + (index % 6)),
                    "inseam_cm": str(70 + (index % 8)),
                    "sleeve_cm": str(55 + (index % 7)),
                    "neck_cm": str(33 + (index % 4)),
                    "thigh_cm": str(48 + (index % 8)),
                    "calf_cm": str(33 + (index % 6)),
                    "body_shape": "average",
                    "generator_version": "test",
                }
            )
            writer.writerow(row)
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (40, 40, 40))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)


def _write_classical_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "classical"
    artifact_dir.mkdir()
    with (artifact_dir / "results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset",
                "dataset_path",
                "feature_config",
                "model_type",
                "feature_count",
                "train_mae",
                "val_mae",
                "test_mae",
                "worst_target",
                "worst_target_mae",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "phase_3t_fixture",
                "dataset_path": "data/synthetic/phase_3t_fixture",
                "feature_config": "raw_scale_camera",
                "model_type": "elasticnet",
                "feature_count": 8,
                "train_mae": 4.0,
                "val_mae": 4.5,
                "test_mae": 4.8,
                "worst_target": "height_cm",
                "worst_target_mae": 8.0,
            }
        )
    with (artifact_dir / "per_target_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["dataset", "feature_config", "model_type", "target", "test_mae"])
        writer.writeheader()
        for index, target in enumerate(TARGET_COLUMNS):
            writer.writerow(
                {
                    "dataset": "phase_3t_fixture",
                    "feature_config": "raw_scale_camera",
                    "model_type": "elasticnet",
                    "target": target,
                    "test_mae": 3.0 + index / 10.0,
                }
            )
    return artifact_dir


def _write_cnn_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "cnn"
    artifact_dir.mkdir()
    metrics = {
        "model_type": "dual_branch_cnn",
        "sample_counts": {"train": 8, "val": 2, "test": 2},
        "train": {"overall_mae": 5.0, "mae_by_target": {target: 5.0 for target in TARGET_COLUMNS}},
        "val": {"overall_mae": 5.5, "mae_by_target": {target: 5.5 for target in TARGET_COLUMNS}},
        "test": {"overall_mae": 5.2, "mae_by_target": {target: 5.2 for target in TARGET_COLUMNS}},
        "epoch_metrics": [{"epoch": 1, "train_overall_mae": 5.0, "val_overall_mae": 5.5}],
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (artifact_dir / "config.json").write_text(json.dumps({"dataset": "data/synthetic/phase_3t_fixture"}), encoding="utf-8")
    return artifact_dir
