"""Testes de validação de schema (Pandera) e features."""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from churn_predictor.data.loader import clean_data
from churn_predictor.data.schema import (
    CATEGORICAL_FEATURES,
    INFERENCE_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_SCHEMA,
)
from churn_predictor.features.transformers import (
    ChargesRatioFeature,
    FeatureEngineer,
    TenureBucketizer,
)


# =============================================================================
# Schema de inferência
# =============================================================================
class TestInferenceSchema:
    def test_inference_features_count(self):
        """Devem ser 19 features (sem customerID e Churn)."""
        assert len(INFERENCE_FEATURES) == 19

    def test_features_are_disjoint(self):
        """Features numéricas e categóricas não se sobrepõem."""
        assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))

    def test_features_cover_all(self):
        """Soma de numéricas + categóricas = total de inference features."""
        total = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
        assert total == set(INFERENCE_FEATURES)


# =============================================================================
# Pandera schemas
# =============================================================================
class TestProcessedSchema:
    def test_valid_data_passes(self, sample_dataframe: pd.DataFrame):
        """DataFrame válido (após clean_data) passa no schema."""
        df = sample_dataframe.copy()
        df["Churn"] = [0, 1] * 5

        # Garante tipos exatos esperados pelo schema
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("int64")
        df["tenure"] = df["tenure"].astype("int64")
        df["Churn"] = df["Churn"].astype("int64")
        df["MonthlyCharges"] = df["MonthlyCharges"].astype("float64")
        df["TotalCharges"] = df["TotalCharges"].astype("float64")

        validated = PROCESSED_SCHEMA.validate(df, lazy=True)
        assert len(validated) == len(df)

    def test_invalid_target_fails(self, sample_dataframe: pd.DataFrame):
        """Target fora de {0, 1} é rejeitado."""
        df = sample_dataframe.copy()
        df["Churn"] = [99] * len(df)  # Inválido

        with pytest.raises(pandera.errors.SchemaErrors):
            PROCESSED_SCHEMA.validate(df, lazy=True)

    def test_negative_tenure_fails(self, sample_dataframe: pd.DataFrame):
        """Tenure negativo é rejeitado."""
        df = sample_dataframe.copy()
        df["Churn"] = 0
        df.loc[0, "tenure"] = -1

        with pytest.raises(pandera.errors.SchemaErrors):
            PROCESSED_SCHEMA.validate(df, lazy=True)


# =============================================================================
# Limpeza de dados
# =============================================================================
class TestCleanData:
    def test_clean_data_drops_id(self):
        """customerID é removido."""
        df = pd.DataFrame(
            {
                "customerID": ["a", "b"],
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [1, 12],
                "PhoneService": ["Yes", "Yes"],
                "MultipleLines": ["No", "Yes"],
                "InternetService": ["DSL", "Fiber optic"],
                "OnlineSecurity": ["No", "Yes"],
                "OnlineBackup": ["No", "Yes"],
                "DeviceProtection": ["No", "Yes"],
                "TechSupport": ["No", "Yes"],
                "StreamingTV": ["No", "Yes"],
                "StreamingMovies": ["No", "Yes"],
                "Contract": ["Month-to-month", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "MonthlyCharges": [29.85, 56.95],
                "TotalCharges": [29.85, 1889.50],
                "Churn": ["No", "Yes"],
            }
        )
        cleaned = clean_data(df)
        assert "customerID" not in cleaned.columns

    def test_clean_data_encodes_target(self):
        """Churn 'Yes'/'No' vira 1/0."""
        df = pd.DataFrame(
            {
                "customerID": ["a", "b"],
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [1, 12],
                "PhoneService": ["Yes", "Yes"],
                "MultipleLines": ["No", "Yes"],
                "InternetService": ["DSL", "Fiber optic"],
                "OnlineSecurity": ["No", "Yes"],
                "OnlineBackup": ["No", "Yes"],
                "DeviceProtection": ["No", "Yes"],
                "TechSupport": ["No", "Yes"],
                "StreamingTV": ["No", "Yes"],
                "StreamingMovies": ["No", "Yes"],
                "Contract": ["Month-to-month", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "MonthlyCharges": [29.85, 56.95],
                "TotalCharges": [29.85, 1889.50],
                "Churn": ["No", "Yes"],
            }
        )
        cleaned = clean_data(df)
        assert cleaned["Churn"].dtype.kind == "i"
        assert set(cleaned["Churn"].unique()) == {0, 1}


# =============================================================================
# Transformadores customizados
# =============================================================================
class TestTenureBucketizer:
    def test_bucket_assignment(self):
        bucketizer = TenureBucketizer()
        df = pd.DataFrame({"tenure": [0, 3, 8, 15, 30, 60]})
        out = bucketizer.fit_transform(df)
        expected = ["0-6m", "0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
        assert out["tenure_bucket"].tolist() == expected


class TestChargesRatioFeature:
    def test_ratio_calculation(self):
        transformer = ChargesRatioFeature()
        df = pd.DataFrame({"tenure": [10, 0, 5], "TotalCharges": [100.0, 50.0, 25.0]})
        out = transformer.fit_transform(df)
        # tenure=0 vira 1 para evitar divisão por zero
        assert out["charges_per_month_ratio"].tolist() == [10.0, 50.0, 5.0]


class TestFeatureEngineer:
    def test_engineer_combined(self, sample_dataframe: pd.DataFrame):
        fe = FeatureEngineer()
        out = fe.fit_transform(sample_dataframe)
        assert "tenure_bucket" in out.columns
        assert "charges_per_month_ratio" in out.columns
        assert len(out) == len(sample_dataframe)
