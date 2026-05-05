"""Schema de validação de dados via Pandera.

Garante que o dataset de entrada tem colunas, tipos e ranges esperados
antes de qualquer processamento. Falhas ficam evidentes no fail-fast.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# ----------------------------------------------------------------------------
# Schema do dataset bruto (Telco Customer Churn da IBM)
# ----------------------------------------------------------------------------
RAW_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(str, nullable=False, unique=True),
        "gender": Column(str, checks=pa.Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, checks=pa.Check.isin([0, 1])),
        "Partner": Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "Dependents": Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "tenure": Column(int, checks=pa.Check.in_range(0, 100)),
        "PhoneService": Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No phone service"]),
        ),
        "InternetService": Column(
            str,
            checks=pa.Check.isin(["DSL", "Fiber optic", "No"]),
        ),
        "OnlineSecurity": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "OnlineBackup": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "DeviceProtection": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "TechSupport": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingTV": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "StreamingMovies": Column(
            str,
            checks=pa.Check.isin(["Yes", "No", "No internet service"]),
        ),
        "Contract": Column(
            str,
            checks=pa.Check.isin(["Month-to-month", "One year", "Two year"]),
        ),
        "PaperlessBilling": Column(str, checks=pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            checks=pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, checks=pa.Check.in_range(0, 1000)),
        "TotalCharges": Column(
            float,
            checks=pa.Check.in_range(0, 100000),
            nullable=True,  # Tem valores vazios após limpeza
        ),
        "Churn": Column(str, checks=pa.Check.isin(["Yes", "No"])),
    },
    strict=True,  # Falha se colunas extras aparecerem
    coerce=False,  # Tipos devem bater exatamente
)


# ----------------------------------------------------------------------------
# Schema do dataset processado (após limpeza, antes do pipeline sklearn)
# ----------------------------------------------------------------------------
PROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(str),
        "SeniorCitizen": Column(int),
        "Partner": Column(str),
        "Dependents": Column(str),
        "tenure": Column(int, checks=pa.Check.ge(0)),
        "PhoneService": Column(str),
        "MultipleLines": Column(str),
        "InternetService": Column(str),
        "OnlineSecurity": Column(str),
        "OnlineBackup": Column(str),
        "DeviceProtection": Column(str),
        "TechSupport": Column(str),
        "StreamingTV": Column(str),
        "StreamingMovies": Column(str),
        "Contract": Column(str),
        "PaperlessBilling": Column(str),
        "PaymentMethod": Column(str),
        "MonthlyCharges": Column(float, checks=pa.Check.ge(0)),
        "TotalCharges": Column(float, checks=pa.Check.ge(0)),
        "Churn": Column(int, checks=pa.Check.isin([0, 1])),
    },
    strict=True,
)


# ----------------------------------------------------------------------------
# Schema mínimo para predição na API (sem target nem ID)
# ----------------------------------------------------------------------------
INFERENCE_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES: list[str] = [
    f for f in INFERENCE_FEATURES if f not in NUMERIC_FEATURES
]
