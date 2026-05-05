"""Carregamento e cache do modelo + pipeline para inferência."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from churn_predictor.models.mlp import ChurnMLP
from churn_predictor.utils.config import settings
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """Container com tudo que a API precisa para inferência."""

    pipeline: Any  # sklearn.pipeline.Pipeline
    model: ChurnMLP
    threshold: float
    metadata: dict[str, Any]
    device: torch.device

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Pipeline completo de inferência: features → probabilidades + classes.

        Returns:
            (probas, predictions) — ambos arrays 1D.
        """
        # Pré-processamento
        X_arr = self.pipeline.transform(X)
        X_tensor = torch.from_numpy(np.ascontiguousarray(X_arr)).float().to(self.device)

        # Inferência
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = torch.sigmoid(logits).cpu().numpy().ravel()

        predictions = (probas >= self.threshold).astype(int)
        return probas, predictions


def load_model_bundle(
    pipeline_path: Path | None = None,
    model_path: Path | None = None,
) -> ModelBundle:
    """Carrega pipeline + modelo do disco.

    Args:
        pipeline_path: Caminho do pipeline.joblib.
        model_path: Caminho do model.pt.

    Returns:
        ModelBundle pronto para inferência.

    Raises:
        FileNotFoundError: Se algum artefato não existir.
    """
    pipeline_path = pipeline_path or settings.api_pipeline_path
    model_path = model_path or settings.api_model_path

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline não encontrado em {pipeline_path}. "
            f"Treine o modelo primeiro com `make train`."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            f"Treine o modelo primeiro com `make train`."
        )

    # Pipeline sklearn
    pipeline = joblib.load(pipeline_path)
    logger.info("pipeline_loaded", path=str(pipeline_path))

    # Modelo PyTorch
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        use_batchnorm=checkpoint.get("use_batchnorm", True),
    )
    model.load_state_dict(checkpoint["state_dict"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    bundle = ModelBundle(
        pipeline=pipeline,
        model=model,
        threshold=checkpoint["optimal_threshold"],
        metadata=checkpoint["metadata"],
        device=device,
    )

    logger.info(
        "model_loaded",
        path=str(model_path),
        input_dim=checkpoint["input_dim"],
        threshold=checkpoint["optimal_threshold"],
        device=str(device),
    )
    return bundle


def classify_risk(probability: float) -> str:
    """Classifica nível de risco baseado em probabilidade."""
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"
