"""Smoke tests: pipeline completo treina e prediz sem quebrar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from churn_predictor.features.pipeline import build_preprocessing_pipeline
from churn_predictor.models.baselines import get_baseline_models
from churn_predictor.models.mlp import ChurnMLP
from churn_predictor.training.dataset import make_dataloader
from churn_predictor.training.trainer import train_mlp


@pytest.mark.smoke
def test_preprocessing_pipeline_smoke(synthetic_training_data: tuple[pd.DataFrame, pd.Series]):
    """Pipeline de pré-processamento fita e transforma sem erros."""
    X, _y = synthetic_training_data
    pipeline = build_preprocessing_pipeline()

    X_transformed = pipeline.fit_transform(X)

    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > 0
    assert not np.isnan(X_transformed).any(), "Pipeline produziu NaNs"


@pytest.mark.smoke
def test_baselines_smoke(synthetic_training_data: tuple[pd.DataFrame, pd.Series]):
    """Todos os baselines treinam e predizem sem erros."""
    X, y = synthetic_training_data
    pipeline = build_preprocessing_pipeline()
    X_arr = pipeline.fit_transform(X)

    for name, model in get_baseline_models().items():
        model.fit(X_arr, y.values)
        preds = model.predict(X_arr)
        probas = model.predict_proba(X_arr)

        assert preds.shape == (len(y),), f"{name}: preds shape errado"
        assert probas.shape == (len(y), 2), f"{name}: probas shape errado"
        assert ((probas >= 0) & (probas <= 1)).all(), f"{name}: probas fora de [0,1]"


@pytest.mark.smoke
def test_mlp_forward_smoke():
    """MLP forward pass funciona com dimensões esperadas."""
    model = ChurnMLP(input_dim=20, hidden_dims=(32, 16), dropout=0.2)
    x = torch.randn(8, 20)
    out = model(x)
    assert out.shape == (8, 1), f"Shape de saída errado: {out.shape}"


@pytest.mark.smoke
def test_mlp_predict_proba_smoke():
    """MLP.predict_proba retorna valores em [0, 1]."""
    model = ChurnMLP(input_dim=20, hidden_dims=(32, 16))
    x = torch.randn(8, 20)
    proba = model.predict_proba(x)
    assert proba.shape == (8, 1)
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.smoke
@pytest.mark.slow
def test_full_training_smoke(synthetic_training_data: tuple[pd.DataFrame, pd.Series]):
    """Treino end-to-end da MLP com dados sintéticos."""
    X, y = synthetic_training_data
    pipeline = build_preprocessing_pipeline()
    X_arr = pipeline.fit_transform(X).astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Split simples
    split = len(X_arr) // 2
    train_loader = make_dataloader(X_arr[:split], y_arr[:split], batch_size=16)
    val_loader = make_dataloader(X_arr[split:], y_arr[split:], batch_size=16, shuffle=False)

    model = ChurnMLP(input_dim=X_arr.shape[1], hidden_dims=(16, 8), dropout=0.1)
    model, history = train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_arr[:split],
        max_epochs=3,  # Smoke: poucas épocas
        early_stopping_patience=10,
    )

    assert len(history.train_loss) >= 1
    assert history.best_epoch >= 1
    assert all(loss > 0 for loss in history.train_loss)
