"""Schemas Pydantic para validação de entrada e saída da API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Features de entrada para predição de churn.

    Reflete as colunas do dataset Telco após limpeza (sem customerID, sem Churn).
    """

    gender: Literal["Male", "Female"] = Field(..., description="Gênero do cliente")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Cliente sênior? 1=sim, 0=não")
    Partner: Literal["Yes", "No"] = Field(..., description="Tem cônjuge?")
    Dependents: Literal["Yes", "No"] = Field(..., description="Tem dependentes?")
    tenure: int = Field(..., ge=0, le=120, description="Meses como cliente")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Tem serviço de telefone?")
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=1000)
    TotalCharges: float = Field(..., ge=0, le=100000)

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: float) -> float:
        if v < 0:
            raise ValueError("TotalCharges não pode ser negativo")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Resposta da predição."""

    churn_probability: float = Field(..., ge=0, le=1, description="Probabilidade de churn [0,1]")
    churn_prediction: int = Field(..., ge=0, le=1, description="Classe predita (0 ou 1)")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Nível de risco")
    threshold_used: float = Field(..., description="Threshold aplicado para binarização")
    model_version: str = Field(..., description="Versão do modelo")

    model_config = {
        "protected_namespaces": (),  # Permite usar 'model_version' como nome de campo
        "json_schema_extra": {
            "example": {
                "churn_probability": 0.78,
                "churn_prediction": 1,
                "risk_level": "high",
                "threshold_used": 0.45,
                "model_version": "0.1.0",
            }
        },
    }


class BatchPredictionRequest(BaseModel):
    """Predição em lote."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Resposta da predição em lote."""

    predictions: list[PredictionResponse]
    n_processed: int
    n_high_risk: int


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    pipeline_loaded: bool
    model_version: str
    uptime_seconds: float

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada."""

    error: str
    detail: str | None = None
    request_id: str | None = None
