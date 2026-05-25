from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from training.datasets.synthetic_body_dataset import MEASUREMENT_COLUMNS, SyntheticBodyDataset
from training.deep.dependencies import import_torch

TARGET_COLUMNS = MEASUREMENT_COLUMNS


class SyntheticBodyImageDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        image_size: int = 128,
        as_tensors: bool = False,
        limit_samples: int | None = None,
        target_columns: list[str] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if limit_samples is not None and limit_samples <= 0:
            raise ValueError("limit_samples must be positive when provided.")

        self.base_dataset = SyntheticBodyDataset(dataset_root, split=split)
        self.image_size = image_size
        self.as_tensors = as_tensors
        self.target_columns = target_columns or list(TARGET_COLUMNS)
        self.samples = list(self.base_dataset)
        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        front_image = load_normalized_image(sample["front_image_path"], self.image_size)
        side_image = load_normalized_image(sample["side_image_path"], self.image_size)
        targets = target_vector(sample, self.target_columns)

        if self.as_tensors:
            torch = import_torch()
            front_value: Any = torch.from_numpy(front_image)
            side_value: Any = torch.from_numpy(side_image)
            target_value: Any = torch.from_numpy(targets)
        else:
            front_value = front_image
            side_value = side_image
            target_value = targets

        return {
            "sample_id": sample["sample_id"],
            "split": sample["dataset_split"],
            "front_image": front_value,
            "side_image": side_value,
            "targets": target_value,
            "target_names": self.target_columns,
        }


def load_normalized_image(image_path: str | Path, image_size: int) -> np.ndarray:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {path}") from error

    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def target_vector(sample: dict[str, Any], target_columns: list[str] | None = None) -> np.ndarray:
    columns = target_columns or list(TARGET_COLUMNS)
    measurements = sample["measurements"]
    try:
        values = [float(measurements[target]) for target in columns]
    except KeyError as error:
        raise ValueError(f"Sample {sample.get('sample_id', '')} is missing target {error}.") from error
    return np.asarray(values, dtype=np.float32)
