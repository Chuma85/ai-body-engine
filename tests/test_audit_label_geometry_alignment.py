import csv
from pathlib import Path

import numpy as np
from PIL import Image

from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS
from synthetic.build_dataset_manifest import build_dataset_manifest
from training.experiments.audit_label_geometry_alignment import (
    audit_label_geometry_alignment,
    build_label_geometry_correlations,
    build_monotonicity_rows,
    ellipse_circumference_proxy,
    find_geometry_label_ambiguity_pairs,
    same_height_different_target_samples,
)


def test_label_geometry_correlation_schema_is_stable() -> None:
    proxy_names = ["chest_band_00_y28_front_norm_width", "chest_band_00_y28_side_norm_depth"]
    proxy_matrix = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float64)
    target_matrix = np.asarray([[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    rows = build_label_geometry_correlations(proxy_matrix, target_matrix, proxy_names)

    assert {"target", "band_name", "proxy", "proxy_role", "correlation", "abs_correlation"} <= set(rows[0])
    assert rows[0]["correlation"] > 0.99
    assert rows[1]["correlation"] < -0.99


def test_monotonicity_checker_detects_increasing_proxy() -> None:
    proxy_names = ["waist_band_00_y42_front_norm_width"]
    proxy_matrix = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float64)
    target_matrix = np.asarray(
        [[0.0, 10.0, 0.0, 0.0], [0.0, 11.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 21.0, 0.0, 0.0], [0.0, 30.0, 0.0, 0.0], [0.0, 31.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    rows = build_monotonicity_rows(proxy_matrix, target_matrix, proxy_names)

    assert rows[0]["monotonic_increasing"] is True
    assert rows[0]["monotonic_score"] == 1.0


def test_ambiguity_detection_finds_similar_geometry_different_labels() -> None:
    sample_ids = ["sample_a", "sample_b", "sample_c"]
    proxy_names = ["hip_band_00_y56_front_norm_width", "hip_band_00_y56_side_norm_depth"]
    proxy_matrix = np.asarray([[1.0, 1.0], [1.01, 1.0], [5.0, 5.0]], dtype=np.float64)
    target_matrix = np.asarray([[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 35.0, 0.0], [0.0, 0.0, 11.0, 0.0]], dtype=np.float64)

    rows = find_geometry_label_ambiguity_pairs(sample_ids, proxy_matrix, target_matrix, proxy_names, ambiguity_pairs_per_target=1)

    assert rows[0]["target"] == "hip_cm"
    assert {rows[0]["sample_id_a"], rows[0]["sample_id_b"]} == {"sample_a", "sample_b"}
    assert rows[0]["label_diff"] == 25.0


def test_ellipse_proxy_is_positive_for_width_and_depth() -> None:
    assert ellipse_circumference_proxy(0.4, 0.2) > 0.0
    assert ellipse_circumference_proxy(0.0, 0.2) == 0.0


def test_same_height_different_target_samples_are_selected() -> None:
    samples = [
        {"sample_id": "a", "measurements": {"height_cm": 170.0, "waist_cm": 70.0}},
        {"sample_id": "b", "measurements": {"height_cm": 171.0, "waist_cm": 105.0}},
        {"sample_id": "c", "measurements": {"height_cm": 190.0, "waist_cm": 110.0}},
    ]

    selected = same_height_different_target_samples(samples, "waist_cm", max_pairs=1)

    assert [sample["sample_id"] for sample in selected] == ["a", "b"]


def test_audit_writes_outputs_on_fixture_dataset(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, 16)
    output_dir = tmp_path / "alignment"

    result = audit_label_geometry_alignment(dataset_root, output_dir, ambiguity_pairs_per_target=2)

    for key in (
        "correlations_json",
        "correlations_csv",
        "correlations_md",
        "monotonicity_csv",
        "ambiguity_csv",
        "summary_md",
    ):
        assert Path(result[key]).exists()


def _write_dataset(tmp_path: Path, count: int) -> Path:
    dataset_root = tmp_path / "data" / "synthetic" / "phase_3y_fixture"
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
            front_width = 16 + (index % 8)
            side_width = 8 + (index % 5)
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
