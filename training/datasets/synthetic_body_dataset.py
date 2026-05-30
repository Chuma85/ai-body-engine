from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Iterator

from synthetic.validate_synthetic_dataset import _read_label_rows

VALID_SPLITS = {"train", "val", "test", "all"}
MEASUREMENT_COLUMNS = [
    "height_cm",
    "weight_kg",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
    "sleeve_cm",
    "neck_cm",
    "thigh_cm",
    "calf_cm",
]


class SyntheticBodyDataset:
    def __init__(self, dataset_root: str | Path, split: str = "all", load_images: bool = False) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.load_images = load_images

        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of: {', '.join(sorted(VALID_SPLITS))}.")
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.manifest_path = self.dataset_root / "manifest.csv"
        self.labels_path = self.dataset_root / "labels" / "labels.csv"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.csv: {self.manifest_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Missing labels.csv: {self.labels_path}")

        self._labels_by_sample = self._load_labels()
        self._manifest_rows = self._load_manifest()

    def __len__(self) -> int:
        return len(self._manifest_rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        manifest_row = self._manifest_rows[index]
        sample_id = manifest_row["sample_id"]
        label_row = self._labels_by_sample.get(sample_id)
        if label_row is None:
            raise ValueError(f"Manifest sample '{sample_id}' is missing from labels.csv.")

        front_image_path = self._resolve_path(manifest_row["front_image_path"])
        side_image_path = self._resolve_path(manifest_row["side_image_path"])
        back_image_path = self._resolve_optional_path(manifest_row.get("back_image_path", ""))
        self._require_existing_file(front_image_path, "front image", sample_id)
        self._require_existing_file(side_image_path, "side image", sample_id)
        if back_image_path is not None:
            self._require_existing_file(back_image_path, "back image", sample_id)

        sample = {
            "sample_id": sample_id,
            "front_image_path": front_image_path,
            "side_image_path": side_image_path,
            "back_image_path": back_image_path,
            "dataset_split": manifest_row["dataset_split"],
            "label_row_index": int(manifest_row["label_row_index"]),
            "labels": label_row,
            "measurements": _parse_measurements(label_row),
            "body_shape": label_row.get("body_shape", ""),
            "generator_version": label_row.get("generator_version", ""),
        }

        if self.load_images:
            sample["front_image_bytes"] = front_image_path.read_bytes()
            sample["side_image_bytes"] = side_image_path.read_bytes()
            if back_image_path is not None:
                sample["back_image_bytes"] = back_image_path.read_bytes()

        return sample

    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "val": 0, "test": 0}
        for row in self._load_all_manifest_rows():
            dataset_split = row["dataset_split"]
            if dataset_split in counts:
                counts[dataset_split] += 1
        return counts

    def _load_labels(self) -> dict[str, dict[str, str]]:
        parse_result: dict[str, Any] = {"errors": [], "warnings": []}
        label_rows = _read_label_rows(self.labels_path, parse_result)
        if parse_result["errors"]:
            raise ValueError(f"Could not parse labels.csv: {'; '.join(parse_result['errors'])}")

        labels_by_sample: dict[str, dict[str, str]] = {}
        for row in label_rows:
            sample_id = row.get("sample_id", "")
            if sample_id:
                labels_by_sample[sample_id] = row
        return labels_by_sample

    def _load_manifest(self) -> list[dict[str, str]]:
        rows = self._load_all_manifest_rows()
        filtered_rows = rows if self.split == "all" else [row for row in rows if row["dataset_split"] == self.split]
        for row in filtered_rows:
            sample_id = row["sample_id"]
            if sample_id not in self._labels_by_sample:
                raise ValueError(f"Manifest sample '{sample_id}' is missing from labels.csv.")
        return filtered_rows

    def _load_all_manifest_rows(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)
            missing_columns = [column for column in ("sample_id", "front_image_path", "side_image_path", "label_row_index", "dataset_split") if column not in (reader.fieldnames or [])]
            if missing_columns:
                raise ValueError(f"manifest.csv is missing columns: {', '.join(missing_columns)}")
            rows = list(reader)

        return rows

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def _resolve_optional_path(self, path_value: str) -> Path | None:
        if path_value in ("", None):
            return None
        return self._resolve_path(path_value)

    @staticmethod
    def _require_existing_file(path: Path, label: str, sample_id: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} for sample '{sample_id}': {path}")


def _parse_measurements(label_row: dict[str, str]) -> dict[str, float]:
    measurements: dict[str, float] = {}
    for column in MEASUREMENT_COLUMNS:
        value = label_row.get(column)
        if value not in ("", None):
            measurements[column] = float(value)
    return measurements


def format_sample_summary(sample: dict[str, Any]) -> str:
    measurements = sample["measurements"]
    summary = (
        f"{sample['sample_id']} | split={sample['dataset_split']} | "
        f"front={sample['front_image_path']} | side={sample['side_image_path']} | "
        f"height_cm={measurements.get('height_cm', '')} | weight_kg={measurements.get('weight_kg', '')} | "
        f"body_shape={sample['body_shape']}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test the synthetic front/side body dataset loader.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="all", choices=sorted(VALID_SPLITS))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)

    dataset = SyntheticBodyDataset(args.dataset, split=args.split)
    counts = dataset.split_counts()
    print(f"Dataset: {dataset.dataset_root}")
    print(f"Split: {args.split}")
    print(f"Loaded samples: {len(dataset)}")
    print(f"Split counts: train={counts['train']} val={counts['val']} test={counts['test']}")
    for index, sample in enumerate(dataset):
        if index >= args.limit:
            break
        print(format_sample_summary(sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
