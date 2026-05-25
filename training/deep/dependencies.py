from __future__ import annotations

from typing import Any

MISSING_TORCH_MESSAGE = (
    "PyTorch is required for the deep image model scaffold. "
    "Install the repository deep-learning dependencies with `pip install -r requirements.txt` "
    "or install PyTorch for your platform from https://pytorch.org/get-started/locally/."
)


class DeepLearningDependencyError(RuntimeError):
    """Raised when an optional deep-learning dependency is unavailable."""


def import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DeepLearningDependencyError(MISSING_TORCH_MESSAGE) from error
    return torch
