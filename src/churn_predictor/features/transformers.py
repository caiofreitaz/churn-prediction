"""Transformadores customizados sklearn-compatíveis.

Encapsulam lógica de feature engineering específica do domínio de churn,
mantendo compatibilidade com `sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TenureBucketizer(BaseEstimator, TransformerMixin):
    """Converte tenure (meses) em buckets de relacionamento.

    Hipótese: clientes em janelas distintas de relacionamento têm
    padrões de churn diferentes (lua-de-mel, consolidação, fidelizado).
    """

    BUCKETS: ClassVar[list[tuple[float, str]]] = [
        (6, "0-6m"),
        (12, "6-12m"),
        (24, "1-2y"),
        (48, "2-4y"),
        (float("inf"), "4y+"),
    ]

    def __init__(self, column: str = "tenure", output_column: str = "tenure_bucket"):
        self.column = column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TenureBucketizer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        bins = [-1] + [b[0] for b in self.BUCKETS]
        labels = [b[1] for b in self.BUCKETS]
        X[self.output_column] = pd.cut(
            X[self.column], bins=bins, labels=labels, include_lowest=True
        ).astype(str)
        return X


class ChargesRatioFeature(BaseEstimator, TransformerMixin):
    """Calcula features de razão entre charges.

    Cria:
    - charges_per_month_ratio: TotalCharges / max(tenure, 1)
        (quanto o cliente pagou em média por mês — captura desvios do
        MonthlyCharges atual e identifica clientes que upgradaram/downgradaram).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChargesRatioFeature:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Evita divisão por zero
        safe_tenure = X["tenure"].replace(0, 1)
        X["charges_per_month_ratio"] = X["TotalCharges"] / safe_tenure
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Combina todas as features customizadas em um único transformador."""

    def __init__(self) -> None:
        self.tenure_bucketizer_ = TenureBucketizer()
        self.charges_ratio_ = ChargesRatioFeature()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        self.tenure_bucketizer_.fit(X, y)
        self.charges_ratio_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.tenure_bucketizer_.transform(X)
        X = self.charges_ratio_.transform(X)
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([])
        return np.array([*list(input_features), "tenure_bucket", "charges_per_month_ratio"])
