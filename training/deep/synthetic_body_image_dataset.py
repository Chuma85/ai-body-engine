from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError

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
        augment: bool = False,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        shift_pixels: int = 0,
        noise_std: float = 0.0,
        horizontal_flip_prob: float = 0.0,
        augment_seed: int = 42,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if limit_samples is not None and limit_samples <= 0:
            raise ValueError("limit_samples must be positive when provided.")
        if brightness_jitter < 0.0:
            raise ValueError("brightness_jitter must be non-negative.")
        if contrast_jitter < 0.0:
            raise ValueError("contrast_jitter must be non-negative.")
        if shift_pixels < 0:
            raise ValueError("shift_pixels must be non-negative.")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0.0 and 1.0.")

        self.base_dataset = SyntheticBodyDataset(dataset_root, split=split)
        self.split = split
        self.image_size = image_size
        self.as_tensors = as_tensors
        self.target_columns = target_columns or list(TARGET_COLUMNS)
        self.augment = augment and split == "train"
        self.augmentation_settings = {
            "enabled": self.augment,
            "brightness_jitter": brightness_jitter,
            "contrast_jitter": contrast_jitter,
            "shift_pixels": shift_pixels,
            "noise_std": noise_std,
            "horizontal_flip_prob": horizontal_flip_prob,
            "augment_seed": augment_seed,
        }
        self.samples = list(self.base_dataset)
        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        front_image = load_normalized_image(
            sample["front_image_path"],
            self.image_size,
            augmentation_settings=self.augmentation_settings if self.augment else None,
            rng=_augmentation_rng(self.augmentation_settings["augment_seed"], index, view_offset=1),
        )
        side_image = load_normalized_image(
            sample["side_image_path"],
            self.image_size,
            augmentation_settings=self.augmentation_settings if self.augment else None,
            rng=_augmentation_rng(self.augmentation_settings["augment_seed"], index, view_offset=2),
        )
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


def load_normalized_image(
    image_path: str | Path,
    image_size: int,
    augmentation_settings: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            augmented = apply_image_augmentation(resized, augmentation_settings, rng)
            array = np.asarray(augmented, dtype=np.float32) / 255.0
            if augmentation_settings and augmentation_settings.get("enabled") and augmentation_settings.get("noise_std", 0.0) > 0.0:
                noise_rng = rng or np.random.default_rng()
                array = array + noise_rng.normal(0.0, float(augmentation_settings["noise_std"]), size=array.shape).astype(np.float32)
                array = np.clip(array, 0.0, 1.0)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Could not read image file: {path}") from error

    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def apply_image_augmentation(
    image: Image.Image,
    augmentation_settings: dict[str, Any] | None,
    rng: np.random.Generator | None,
) -> Image.Image:
    if not augmentation_settings or not augmentation_settings.get("enabled"):
        return image
    generator = rng or np.random.default_rng()
    augmented = image

    shift_pixels = int(augmentation_settings.get("shift_pixels", 0))
    if shift_pixels > 0:
        dx = int(generator.integers(-shift_pixels, shift_pixels + 1))
        augmented = augmented.transform(
            augmented.size,
            Image.AFFINE,
            (1, 0, -dx, 0, 1, 0),
            resample=Image.BILINEAR,
            fillcolor=(50, 50, 50),
        )

    horizontal_flip_prob = float(augmentation_settings.get("horizontal_flip_prob", 0.0))
    if horizontal_flip_prob > 0.0 and float(generator.random()) < horizontal_flip_prob:
        augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

    brightness_jitter = float(augmentation_settings.get("brightness_jitter", 0.0))
    if brightness_jitter > 0.0:
        factor = 1.0 + float(generator.uniform(-brightness_jitter, brightness_jitter))
        augmented = ImageEnhance.Brightness(augmented).enhance(factor)

    contrast_jitter = float(augmentation_settings.get("contrast_jitter", 0.0))
    if contrast_jitter > 0.0:
        factor = 1.0 + float(generator.uniform(-contrast_jitter, contrast_jitter))
        augmented = ImageEnhance.Contrast(augmented).enhance(factor)

    return augmented


def _augmentation_rng(seed: int, index: int, view_offset: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + index * 1009 + view_offset * 9176)


def target_vector(sample: dict[str, Any], target_columns: list[str] | None = None) -> np.ndarray:
    columns = target_columns or list(TARGET_COLUMNS)
    measurements = sample["measurements"]
    try:
        values = [float(measurements[target]) for target in columns]
    except KeyError as error:
        raise ValueError(f"Sample {sample.get('sample_id', '')} is missing target {error}.") from error
    return np.asarray(values, dtype=np.float32)
