import csv

from synthetic.generator.generate_dataset import LABEL_COLUMNS, generate_dataset
from synthetic.generator.validate_dataset import validate_dataset


def test_generate_tiny_synthetic_dataset(tmp_path) -> None:
    output_dir = tmp_path / "phase_2a"

    labels_csv = generate_dataset(count=5, output_dir=str(output_dir), width=128, height=192)

    assert labels_csv.exists()

    with labels_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert len(rows) == 5
    assert set(LABEL_COLUMNS).issubset(reader.fieldnames or [])

    for row in rows:
        assert (tmp_path / "phase_2a" / "images" / "front" / f"{row['sample_id']}_front.png").exists()
        assert (tmp_path / "phase_2a" / "images" / "side" / f"{row['sample_id']}_side.png").exists()
        assert not (tmp_path / "phase_2a" / "images" / "back" / f"{row['sample_id']}_back.png").exists()
        assert row["front_image_path"]
        assert row["side_image_path"]
        assert row["back_image_path"] == ""
        assert row["has_back"] == "False"
        assert row["capture_views"] == "front,side"

    validation = validate_dataset(str(labels_csv))

    assert validation["valid"] is True
    assert validation["row_count"] == 5
    assert validation["missing_files"] == []
    assert validation["missing_columns"] == []


def test_generate_tiny_synthetic_dataset_with_optional_back_view(tmp_path) -> None:
    output_dir = tmp_path / "phase_3t_back_smoke"

    labels_csv = generate_dataset(
        count=3,
        output_dir=str(output_dir),
        width=128,
        height=192,
        include_back_view=True,
    )

    with labels_csv.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3
    for row in rows:
        sample_id = row["sample_id"]
        assert (output_dir / "images" / "front" / f"{sample_id}_front.png").exists()
        assert (output_dir / "images" / "side" / f"{sample_id}_side.png").exists()
        assert (output_dir / "images" / "back" / f"{sample_id}_back.png").exists()
        assert row["front_image_path"].endswith(f"{sample_id}_front.png")
        assert row["side_image_path"].endswith(f"{sample_id}_side.png")
        assert row["back_image_path"].endswith(f"{sample_id}_back.png")
        assert row["has_front"] == "True"
        assert row["has_side"] == "True"
        assert row["has_back"] == "True"
        assert row["capture_views"] == "front,side,back"

    validation = validate_dataset(str(labels_csv))

    assert validation["valid"] is True
    assert validation["missing_files"] == []
