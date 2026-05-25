from __future__ import annotations

from typing import Any

from training.deep.dependencies import import_torch


class DualBranchCNN:
    def __new__(
        cls,
        target_count: int,
        embedding_dim: int = 96,
        shared_encoder: bool = False,
        dropout: float = 0.2,
    ) -> Any:
        torch = import_torch()
        nn = torch.nn
        model_class = _build_model_class(nn)
        return model_class(
            target_count=target_count,
            embedding_dim=embedding_dim,
            shared_encoder=shared_encoder,
            dropout=dropout,
        )


def _build_model_class(nn: Any) -> type:
    class ImageEncoder(nn.Module):
        def __init__(self, embedding_dim: int, dropout: float) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(96, embedding_dim),
                nn.ReLU(),
            )

        def forward(self, images: Any) -> Any:
            return self.encoder(images)

    class _DualBranchCNN(nn.Module):
        def __init__(
            self,
            target_count: int,
            embedding_dim: int = 96,
            shared_encoder: bool = False,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if target_count <= 0:
                raise ValueError("target_count must be positive.")
            if embedding_dim <= 0:
                raise ValueError("embedding_dim must be positive.")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be greater than or equal to 0.0 and less than 1.0.")

            self.shared_encoder = shared_encoder
            self.dropout = dropout
            self.front_encoder = ImageEncoder(embedding_dim, dropout)
            self.side_encoder = self.front_encoder if shared_encoder else ImageEncoder(embedding_dim, dropout)
            self.regressor = nn.Sequential(
                nn.Linear(embedding_dim * 2, 192),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, target_count),
            )

        def forward(self, front_images: Any, side_images: Any) -> Any:
            torch = import_torch()
            front_embedding = self.front_encoder(front_images)
            side_embedding = self.side_encoder(side_images)
            combined = torch.cat([front_embedding, side_embedding], dim=1)
            return self.regressor(combined)

    return _DualBranchCNN
