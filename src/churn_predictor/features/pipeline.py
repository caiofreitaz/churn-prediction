"""Pipeline de pré-processamento sklearn.

Encapsula feature engineering + scaling + encoding em um Pipeline único,
garantindo zero data leakage e reprodutibilidade.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_predictor.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from churn_predictor.features.transformers import FeatureEngineer
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def build_preprocessing_pipeline() -> Pipeline:
    """Constrói pipeline de pré-processamento.

    Etapas:
    1. FeatureEngineer: cria features customizadas (tenure_bucket, ratios)
    2. ColumnTransformer:
       - Numéricas: StandardScaler
       - Categóricas: OneHotEncoder(handle_unknown='ignore')

    Returns:
        Pipeline pronto para `.fit_transform(X_train)` e `.transform(X_test)`.
    """
    # Após FeatureEngineer, ganhamos novas colunas
    numeric_features = [*NUMERIC_FEATURES, "charges_per_month_ratio"]
    categorical_features = [*CATEGORICAL_FEATURES, "tenure_bucket"]

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    logger.info(
        "preprocessing_pipeline_built",
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
    )
    return pipeline
