"""Dataset PyTorch e DataLoaders para treino da MLP."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChurnDataset(Dataset):
    """Dataset PyTorch para dados tabulares de churn.

    Args:
        X: Features pré-processadas (numpy array float32).
        y: Targets binários 0/1 (numpy array).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"X e y têm tamanhos diferentes: {len(X)} vs {len(y)}")

        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Constrói DataLoader a partir de arrays numpy."""
    dataset = ChurnDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
