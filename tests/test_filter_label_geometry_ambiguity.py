import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments import filter_label_geometry_ambiguity as ambiguity


def test_ambiguity_score_calculation_is_deterministic() -> None:
    sample_ids = ["a", "b", "c", "d"]
    proxy_names = ["waist_band_00_y42_front_norm_width", "waist_band_00_y42_side_norm_depth"]
    proxy_matrix = np.asarray([[1.0, 1.0], [1.01, 1.0], [4.0, 4.0], [4.1, 4.0]], dtype=np.float64)
    target_matrix = np.asarray(
        [
            [0.0, 70.0, 0.0, 0.0],
            [0.0, 105.0, 0.0, 0.0],
            [0.0, 75.0, 0.0, 0.0],
            [0.0, 76.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    splits = {sample_id: "train" for sample_id in sample_ids}

    first = ambiguity.calculate_ambiguity_scores(sample_ids, proxy_names, proxy_matrix, target_matrix, splits, ambiguity_percentile=75.0)
    second = ambiguity.calculate_ambiguity_scores(sample_ids, proxy_names, proxy_matrix, target_matrix, splits, ambiguity_percentile=75.0)

    assert first["rows"] == second["rows"]
    row_b = next(row for row in first["rows"] if row["sample_id"] == "b")
    assert row_b["waist_cm_nearest_ambiguous_neighbor"] == "a"
    assert row_b["waist_cm_nearest_label_diff"] == 35.0


def test_similar_geometry_different_label_pairs_are_detected() -> None:
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    scores, neighbor_ids, label_diffs = ambiguity.target_ambiguity_scores(
        sample_ids,
        np.asarray([[1.0, 1.0], [1.0, 1.01], [5.0, 5.0]], dtype=np.float64),
        np.asarray([70.0, 102.0, 71.0], dtype=np.float64),
        nearest_neighbors=1,
    )

    assert neighbor_ids[0] == "sample_b"
    assert label_diffs[0] == 32.0
    assert scores[0] > scores[2]


def test_filtered_manifests_preserve_columns_and_test_split_behavior() -> None:
    rows = [
        {"sample_id": "train_a", "dataset_split": "train", "front_image_path": "a", "side_image_path": "b", "label_row_index": "0"},
        {"sample_id": "test_a", "dataset_split": "test", "front_image_path": "c", "side_image_path": "d", "label_row_index": "1"},
    ]
    ambiguous_ids = {"train_a", "test_a"}

    clean_train = ambiguity.filter_manifest_rows(rows, ambiguous_ids, mode="clean_train_only")
    clean_all = ambiguity.filter_manifest_rows(rows, ambiguous_ids, mode="clean_train_clean_test")

    assert [row["sample_id"] for row in clean_train] == ["test_a"]
    assert clean_train[0]["dataset_split"] == "test"
    assert clean_all == []
    assert set(clean_train[0]) == set(rows[0])


def test_missing_phase3y_optional_artifacts_warns_without_crashing(tmp_path: Path) -> None:
    warnings = ambiguity.validate_optional_phase3y_artifacts(tmp_path / "missing_phase3y")

    assert warnings
    assert "recomputed geometry proxies" in warnings[0]


def test_filtering_tiny_fixture_writes_expected_outputs(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 30)
    output_dir = tmp_path / "artifacts" / "phase_3z"

    result = ambiguity.filter_label_geometry_ambiguity(
        dataset_root,
        output_dir,
        phase3y_artifacts=tmp_path / "missing_phase3y",
        model_types=["ridge"],
        ambiguity_percentile=80.0,
        ambiguity_pairs_per_target=2,
    )

    for key in (
        "ambiguity_scores_csv",
        "filtered_manifest_clean_train_csv",
        "filtered_manifest_clean_all_csv",
        "ambiguous_pairs_csv",
        "benchmark_results_json",
        "benchmark_results_csv",
        "per_target_results_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()
    summary = json.loads(Path(result["benchmark_results_json"]).read_text(encoding="utf-8"))
    assert summary["targets"] == ambiguity.TARGETS
    assert {"run_name", "variant", "model_type", "test_group_mae"} <= set(summary["benchmark_results"][0])
    with Path(result["filtered_manifest_clean_train_csv"]).open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    test_rows = [row for row in rows if row["dataset_split"] == "test"]
    assert len(test_rows) == 3


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3z_fixture"
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
            front_width = 14 + (index % 9)
            side_width = 7 + (index % 6)
            _write_rect_image(front_dir / f"{sample_id}_front.png", (24, 8, 24 + front_width, 58), (96, 96))
            _write_rect_image(side_dir / f"{sample_id}_side.png", (32, 8, 32 + side_width, 58), (96, 96))
            label_noise = 20 if index in {5, 15, 25} else 0
            row = {column: "" for column in LABEL_COLUMNS}
            row.update(
                {
                    "sample_id": sample_id,
                    "front_image_path": (front_dir / f"{sample_id}_front.png").as_posix(),
                    "side_image_path": (side_dir / f"{sample_id}_side.png").as_posix(),
                    "height_cm": str(160 + index),
                    "weight_kg": str(55 + index),
                    "chest_cm": str(80 + front_width + label_noise),
                    "waist_cm": str(70 + front_width + label_noise),
                    "hip_cm": str(88 + front_width + label_noise),
                    "shoulder_cm": str(40 + index / 3),
                    "inseam_cm": str(70 + index / 4),
                    "sleeve_cm": str(55 + index / 5),
                    "neck_cm": str(33 + index / 8),
                    "thigh_cm": str(45 + side_width + label_noise),
                    "calf_cm": str(32 + side_width / 2),
                    "body_shape": "average",
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
