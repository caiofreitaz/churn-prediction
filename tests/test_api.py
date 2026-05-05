"""Testes da API FastAPI."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class MockBundle:
    """Mock de ModelBundle que retorna predições determinísticas."""

    threshold: ClassVar[float] = 0.5
    metadata: ClassVar[dict] = {"dataset_hash": "mock123"}

    def predict(self, X):
        n = len(X)
        probas = np.array([0.7] * n)
        preds = (probas >= self.threshold).astype(int)
        return probas, preds


@pytest.fixture
def mock_bundle():
    return MockBundle()


@pytest.fixture
def client(mock_bundle):
    """TestClient com modelo mockado via patch no load_model_bundle."""
    from churn_predictor.api import main as api_main

    with (
        patch.object(api_main, "load_model_bundle", return_value=mock_bundle),
        TestClient(api_main.app) as c,
    ):
        yield c


# =============================================================================
# Endpoints simples
# =============================================================================
class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "churn-predictor"
        assert "version" in body


class TestHealthEndpoint:
    def test_health_with_model(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_health_without_model(self):
        """Quando load_model_bundle falha, /health retorna unhealthy."""
        from churn_predictor.api import main as api_main

        with patch.object(
            api_main,
            "load_model_bundle",
            side_effect=FileNotFoundError("no model"),
        ), TestClient(api_main.app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"


# =============================================================================
# Endpoint /predict
# =============================================================================
class TestPredictEndpoint:
    def test_predict_valid_payload(self, client: TestClient, sample_customer: dict):
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200, response.text

        body = response.json()
        assert "churn_probability" in body
        assert "churn_prediction" in body
        assert "risk_level" in body
        assert 0 <= body["churn_probability"] <= 1
        assert body["churn_prediction"] in (0, 1)
        assert body["risk_level"] in ("low", "medium", "high")

    def test_predict_invalid_gender(self, client: TestClient, sample_customer: dict):
        sample_customer["gender"] = "Other"  # Inválido
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 422

    def test_predict_negative_tenure(self, client: TestClient, sample_customer: dict):
        sample_customer["tenure"] = -1
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 422

    def test_predict_missing_field(self, client: TestClient, sample_customer: dict):
        del sample_customer["gender"]
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 422

    def test_predict_returns_request_id_header(
        self, client: TestClient, sample_customer: dict
    ):
        response = client.post("/predict", json=sample_customer)
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time-Ms" in response.headers

    def test_predict_without_model_returns_503(self, sample_customer: dict):
        """Quando o modelo falha ao carregar, /predict retorna 503."""
        from churn_predictor.api import main as api_main

        with patch.object(
            api_main,
            "load_model_bundle",
            side_effect=FileNotFoundError("no model"),
        ), TestClient(api_main.app) as client:
            response = client.post("/predict", json=sample_customer)
        assert response.status_code == 503


# =============================================================================
# Endpoint /predict/batch
# =============================================================================
class TestBatchEndpoint:
    def test_batch_predict(self, client: TestClient, sample_customer: dict):
        payload = {"customers": [sample_customer, sample_customer]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert body["n_processed"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_empty_fails(self, client: TestClient):
        response = client.post("/predict/batch", json={"customers": []})
        assert response.status_code == 422

    def test_batch_too_large_fails(self, client: TestClient, sample_customer: dict):
        # >1000 customers
        payload = {"customers": [sample_customer] * 1001}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422
