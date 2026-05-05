"""Testes unitários de métricas, modelos e reprodutibilidade."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from churn_predictor.evaluation.metrics import (
    compute_business_cost,
    compute_metrics,
    find_optimal_threshold,
)
from churn_predictor.models.mlp import ChurnMLP
from churn_predictor.utils.seeds import set_seeds


# =============================================================================
# Métricas
# =============================================================================
class TestMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_metrics(y_true, y_proba, threshold=0.5)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_all_negatives(self):
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        m = compute_metrics(y_true, y_proba, threshold=0.5)
        assert m.true_negatives == 4
        assert m.false_positives == 0

    def test_threshold_changes_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.4, 0.6, 0.55, 0.7])
        m_low = compute_metrics(y_true, y_proba, threshold=0.5)
        m_high = compute_metrics(y_true, y_proba, threshold=0.65)
        # Threshold maior → menos positivos preditos
        assert m_high.true_positives <= m_low.true_positives


class TestBusinessCost:
    def test_cost_calculation(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])  # 1 FP, 1 FN
        cost = compute_business_cost(y_true, y_pred, cost_fn=500.0, cost_fp=50.0)
        assert cost["total_cost"] == 550.0
        assert cost["n_fp"] == 1
        assert cost["n_fn"] == 1

    def test_optimal_threshold_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.uniform(0, 1, 1000)
        threshold, cost = find_optimal_threshold(y_true, y_proba)
        assert 0.05 <= threshold <= 0.95
        assert cost >= 0


# =============================================================================
# MLP
# =============================================================================
class TestChurnMLP:
    def test_mlp_construction(self):
        model = ChurnMLP(input_dim=10, hidden_dims=(32, 16), dropout=0.2)
        assert model.input_dim == 10
        assert model.hidden_dims == (32, 16)
        assert model.num_parameters() > 0

    def test_mlp_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            ChurnMLP(input_dim=10, dropout=1.5)

    def test_mlp_empty_hidden_dims(self):
        with pytest.raises(ValueError, match="hidden_dims"):
            ChurnMLP(input_dim=10, hidden_dims=())

    def test_mlp_forward_batch(self):
        model = ChurnMLP(input_dim=10, hidden_dims=(8,))
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_mlp_predict_classes(self):
        model = ChurnMLP(input_dim=10, hidden_dims=(8,))
        x = torch.randn(4, 10)
        preds = model.predict(x, threshold=0.5)
        assert preds.shape == (4, 1)
        assert ((preds == 0) | (preds == 1)).all()


# =============================================================================
# Reprodutibilidade
# =============================================================================
class TestReproducibility:
    def test_set_seeds_makes_torch_deterministic(self):
        set_seeds(42)
        a = torch.rand(5)
        set_seeds(42)
        b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_set_seeds_makes_numpy_deterministic(self):
        set_seeds(42)
        a = np.random.rand(5)
        set_seeds(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_mlp_deterministic_init(self):
        set_seeds(42)
        m1 = ChurnMLP(input_dim=10, hidden_dims=(16,))
        set_seeds(42)
        m2 = ChurnMLP(input_dim=10, hidden_dims=(16,))

        # Pesos devem ser idênticos com mesma seed
        for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.allclose(p1, p2)
