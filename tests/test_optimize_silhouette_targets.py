import csv
import json
from pathlib import Path

from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments import optimize_silhouette_targets as optimizer


def test_silhouette_target_list_excludes_weak_targets() -> None:
    assert optimizer.SILHOUETTE_TARGETS == ["chest_cm", "waist_cm", "hip_cm", "thigh_cm", "shoulder_cm", "calf_cm"]
    for excluded in ("height_cm", "weight_kg", "inseam_cm", "sleeve_cm", "neck_cm"):
        assert excluded not in optimizer.SILHOUETTE_TARGETS


def test_grouped_mae_and_promotion_gate_are_deterministic() -> None:
    assert optimizer.promotion_gate(5.1)["gate"] == "research_only"
    assert optimizer.promotion_gate(4.0)["gate"] == "assisted_manual_confirmation"
    assert optimizer.promotion_gate(2.0)["gate"] == "stronger_candidate"


def test_per_target_selection_is_deterministic() -> None:
    rows = [
        {"run_name": "run_b", "target": "chest_cm", "test_mae": 3.0, "model_type": "ridge", "feature_config": "raw", "mode": "multi_output", "promotion_gate": "assisted"},
        {"run_name": "run_a", "target": "chest_cm", "test_mae": 3.0, "model_type": "ridge", "feature_config": "raw", "mode": "multi_output", "promotion_gate": "assisted"},
    ]

    best = optimizer.select_best_per_target(rows)

    assert best[0]["run_name"] == "run_a"


def test_optimizer_tiny_fixture_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    dataset_root = _write_dataset(tmp_path, 24)
    output_dir = tmp_path / "artifacts" / "phase_3w"
    monkeypatch.setattr(optimizer, "DEFAULT_MODEL_FEATURE_COMBOS", [("raw_scale_camera", "ridge")])

    result = optimizer.optimize_silhouette_targets(dataset_root, output_dir, cnn_metrics=None)

    for key in (
        "benchmark_results_json",
        "benchmark_results_csv",
        "per_target_results_csv",
        "best_model_per_target_csv",
        "error_analysis_csv",
        "promotion_gate_summary_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["benchmark_results_json"]).read_text(encoding="utf-8"))
    assert summary["silhouette_targets"] == optimizer.SILHOUETTE_TARGETS
    assert set(summary["excluded_targets"]) == {"height_cm", "weight_kg", "inseam_cm", "sleeve_cm", "neck_cm"}
    with Path(result["per_target_results_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["target"] for row in rows} == set(optimizer.SILHOUETTE_TARGETS)


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3w_fixture"
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
            front_width = 16 + (index % 10)
            side_width = 8 + (index % 6)
            _write_rect_image(front_dir / f"{sample_id}_front.png", (24, 8, 24 + front_width, 58), (96, 96))
            _write_rect_image(side_dir / f"{sample_id}_side.png", (32, 8, 32 + side_width, 58), (96, 96))
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(80 + front_width),
                    "waist_cm": str(70 + front_width),
                    "hip_cm": str(88 + front_width),
                    "shoulder_cm": str(40 + index / 3),
                    "inseam_cm": str(70 + index / 4),
                    "sleeve_cm": str(55 + index / 5),
                    "neck_cm": str(33 + index / 8),
                    "thigh_cm": str(45 + side_width),
                    "calf_cm": str(32 + side_width / 2),
                    "body_shape": "average" if index % 2 else "broad",
                    "generator_version": "test",
                }
            )
            writer.writerow(row)

    manifest = build_dataset_manifest(dataset_root)
    assert manifest["valid"] is True
    return dataset_root


def _write_rect_image(path: Path, rect: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, (40, 40, 40))
    pixels = image.load()
    x_min, y_min, x_max, y_max = rect
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[x, y] = (220, 220, 220)
    image.save(path)
