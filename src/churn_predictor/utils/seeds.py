"""Utilitários para reprodutibilidade.

Fixa seeds em todas as bibliotecas relevantes (Python, NumPy, PyTorch).
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch

from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def set_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """Fixa seeds para reprodutibilidade global.

    Args:
        seed: Valor da semente.
        deterministic: Se True, usa algoritmos determinísticos no PyTorch
            (mais lento, mas garante bit-exact reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Variáveis de ambiente úteis para frameworks como cuBLAS
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Determinismo total no PyTorch (a partir de 1.8+)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with contextlib.suppress(AttributeError):
            torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("seeds_configured", seed=seed, deterministic=deterministic)


def get_device() -> torch.device:
    """Retorna o device disponível (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.debug("device_selected", device=str(device))
    return device
