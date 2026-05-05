"""Carregamento, limpeza e split do dataset Telco Customer Churn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_predictor.data.schema import PROCESSED_SCHEMA, RAW_SCHEMA
from churn_predictor.utils.config import RAW_DATA_DIR, settings
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplit:
    """Container para os splits de treino/validação/teste."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def summary(self) -> dict[str, int]:
        return {
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "test_size": len(self.X_test),
            "n_features": self.X_train.shape[1],
            "train_pos_rate": float(self.y_train.mean()),
            "val_pos_rate": float(self.y_val.mean()),
            "test_pos_rate": float(self.y_test.mean()),
        }


def load_raw_data(path: Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Carrega o dataset bruto do CSV.

    Args:
        path: Caminho do CSV. Default: usa settings.dataset_filename.
        validate: Se True, valida contra RAW_SCHEMA.

    Returns:
        DataFrame bruto, sem transformações.
    """
    if path is None:
        path = RAW_DATA_DIR / settings.dataset_filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. "
            f"Execute `make download-data` ou `python scripts/download_data.py`."
        )

    df = pd.read_csv(path)
    logger.info("dataset_loaded", path=str(path), shape=df.shape)

    # Limpeza mínima para passar no schema bruto
    # TotalCharges vem como string com espaços vazios
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if validate:
        try:
            RAW_SCHEMA.validate(df, lazy=True)
            logger.info("raw_schema_validation_passed")
        except Exception as e:
            logger.error("raw_schema_validation_failed", error=str(e))
            raise

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e prepara o dataset para o pipeline.

    - Remove customerID (não tem valor preditivo)
    - Imputa TotalCharges faltante com 0 (clientes novos com tenure=0)
    - Codifica target Churn como 0/1

    Args:
        df: DataFrame bruto validado.

    Returns:
        DataFrame limpo, validado contra PROCESSED_SCHEMA.
    """
    df = df.copy()

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Imputação de TotalCharges (clientes com tenure=0 têm TotalCharges vazio)
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing > 0:
        logger.info("imputing_total_charges", n_missing=int(n_missing))
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0).astype(float)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Validação
    try:
        PROCESSED_SCHEMA.validate(df, lazy=True)
        logger.info("processed_schema_validation_passed", shape=df.shape)
    except Exception as e:
        logger.error("processed_schema_validation_failed", error=str(e))
        raise

    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
) -> DataSplit:
    """Divide dados em treino/validação/teste de forma estratificada.

    Args:
        df: DataFrame limpo com a coluna alvo.
        target: Nome da coluna alvo.
        test_size: Fração para teste (default: settings.test_size).
        val_size: Fração do treino para validação (default: settings.val_size).
        random_state: Seed (default: settings.random_seed).

    Returns:
        DataSplit com X_train/val/test e y_train/val/test.
    """
    test_size = test_size if test_size is not None else settings.test_size
    val_size = val_size if val_size is not None else settings.val_size
    random_state = random_state if random_state is not None else settings.random_seed

    X = df.drop(columns=[target])
    y = df[target]

    # Primeiro: separa teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Segundo: separa validação do treino
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    split = DataSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )

    logger.info("data_split_completed", **split.summary())
    return split


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Calcula hash determinístico do dataset (para versionamento no MLflow)."""
    import hashlib

    # Ordena colunas e converte para bytes determinísticos
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_and_split(
    path: Path | None = None,
    random_state: int | None = None,
) -> tuple[DataSplit, dict[str, str | int]]:
    """Pipeline completo: carrega → limpa → divide.

    Returns:
        (split, metadata) onde metadata contém info para tracking.
    """
    df_raw = load_raw_data(path)
    df_clean = clean_data(df_raw)
    split = split_data(df_clean, random_state=random_state)

    metadata = {
        "dataset_hash": compute_dataset_hash(df_clean),
        "n_rows": len(df_clean),
        "n_features": len(df_clean.columns) - 1,
        "positive_rate": float(df_clean["Churn"].mean()),
    }
    return split, metadata
