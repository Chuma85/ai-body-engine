import csv
import json
from pathlib import Path

import pytest

from synthetic.audit_synthetic_variation import (
    audit_synthetic_variation,
    numeric_summary,
)
from synthetic.blender.scripts.render_parametric_body import LABEL_COLUMNS


def test_audit_parses_headerless_labels_csv(tmp_path) -> None:
    dataset = _write_labels_dataset(tmp_path, _rows(), include_header=False)

    summary = audit_synthetic_variation(dataset)

    assert summary["sample_count"] == 3
    assert "height_cm" in summary["numeric_columns"]
    assert any("labels.csv has no header" in warning for warning in summary["warnings"])


def test_numeric_summary_statistics_are_correct() -> None:
    stats = numeric_summary([160.0, 170.0, 180.0])

    assert stats["count"] == 3
    assert stats["min"] == 160.0
    assert stats["max"] == 180.0
    assert stats["mean"] == 170.0
    assert stats["std"] == pytest.approx(8.1649658)
    assert stats["range"] == 20.0


def test_low_variation_warning_is_emitted(tmp_path) -> None:
    rows = [_row(index, height_cm="170", weight_kg="70") for index in range(1, 4)]
    dataset = _write_labels_dataset(tmp_path, rows)

    summary = audit_synthetic_variation(dataset)

    assert any("Low variation: height_cm" in warning for warning in summary["warnings"])
    assert summary["measurement_stats"]["height_cm"]["range"] == 0.0


def test_outlier_warning_is_emitted(tmp_path) -> None:
    rows = _rows()
    rows[0]["height_cm"] = "250"
    dataset = _write_labels_dataset(tmp_path, rows)

    summary = audit_synthetic_variation(dataset)

    assert any("Outlier values: height_cm" in warning for warning in summary["warnings"])


def test_audit_writes_summary_json_and_report_md(tmp_path) -> None:
    dataset = _write_labels_dataset(tmp_path, _rows())
    output_dir = tmp_path / "artifacts" / "analysis" / "audit"

    summary = audit_synthetic_variation(dataset, output_dir)

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    assert summary_path.exists()
    assert report_path.exists()
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["sample_count"] == 3
    assert "Synthetic Variation Audit" in report_path.read_text(encoding="utf-8")
    assert summary["summary_path"] == str(summary_path)
    assert summary["report_path"] == str(report_path)


def test_missing_and_non_numeric_fields_are_reported(tmp_path) -> None:
    rows = _rows()
    rows[0]["waist_cm"] = ""
    rows[1]["hip_cm"] = "wide"
    dataset = _write_labels_dataset(tmp_path, rows)

    summary = audit_synthetic_variation(dataset)

    assert summary["missing_fields"]["waist_cm"] == 1
    assert summary["non_numeric_fields"] == [
        {
            "row_index": 1,
            "sample_id": "sample_000002",
            "column": "hip_cm",
            "value": "wide",
        }
    ]


def _rows() -> list[dict[str, str]]:
    return [
        _row(1, height_cm="160", weight_kg="60", chest_cm="90", waist_cm="70", hip_cm="95"),
        _row(2, height_cm="170", weight_kg="75", chest_cm="100", waist_cm="82", hip_cm="105"),
        _row(3, height_cm="180", weight_kg="90", chest_cm="110", waist_cm="94", hip_cm="115"),
    ]


def _row(index: int, **overrides: str) -> dict[str, str]:
    row = {column: "" for column in LABEL_COLUMNS}
    row.update(
        {
            "sample_id": f"sample_{index:06d}",
            "front_image_path": f"images/front/sample_{index:06d}_front.png",
            "side_image_path": f"images/side/sample_{index:06d}_side.png",
            "height_cm": "170",
            "weight_kg": "70",
            "chest_cm": "95",
            "waist_cm": "80",
            "hip_cm": "100",
            "shoulder_cm": "45",
            "inseam_cm": "80",
            "sleeve_cm": "62",
            "neck_cm": "38",
            "thigh_cm": "55",
            "calf_cm": "38",
            "body_shape": "average",
            "generator_version": "test",
        }
    )
    row.update(overrides)
    return row


def _write_labels_dataset(tmp_path: Path, rows: list[dict[str, str]], include_header: bool = True) -> Path:
    dataset = tmp_path / "data" / "synthetic" / "phase_audit"
    labels_dir = dataset / "labels"
    labels_dir.mkdir(parents=True)

    with (labels_dir / "labels.csv").open("w", newline="", encoding="utf-8") as labels_file:
        if include_header:
            writer = csv.DictWriter(labels_file, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(labels_file)
            for row in rows:
                writer.writerow([row.get(column, "") for column in LABEL_COLUMNS])

    return dataset
