"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_predictor.utils.seeds import set_seeds


@pytest.fixture(autouse=True)
def fix_seeds():
    """Fixa seeds em todos os testes para reprodutibilidade."""
    set_seeds(42)


@pytest.fixture
def sample_customer() -> dict:
    """Exemplo válido de cliente para predição."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.10,
        "TotalCharges": 1069.20,
    }


@pytest.fixture
def sample_dataframe(sample_customer: dict) -> pd.DataFrame:
    """DataFrame com 10 clientes sintéticos para smoke tests."""
    np.random.seed(42)
    rows = []
    for _i in range(10):
        row = sample_customer.copy()
        row["tenure"] = int(np.random.randint(1, 72))
        row["MonthlyCharges"] = float(np.random.uniform(20, 120))
        row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético pequeno para testes rápidos de treino."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(1, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": np.random.uniform(20, 120, n),
            "TotalCharges": np.random.uniform(0, 8000, n),
        }
    )
    # Cria target com sinal: contratos mensais e fiber optic têm mais churn
    score = (
        (df["Contract"] == "Month-to-month").astype(int) * 0.5
        + (df["InternetService"] == "Fiber optic").astype(int) * 0.3
        + np.random.normal(0, 0.3, n)
    )
    y = pd.Series((score > 0.4).astype(int), name="Churn")
    return df, y
