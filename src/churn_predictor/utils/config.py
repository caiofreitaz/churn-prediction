"""Configuração centralizada do projeto via Pydantic Settings.

Single source of truth para todos os parâmetros configuráveis.
Pode ser sobrescrito por variáveis de ambiente ou arquivo .env.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ----------------------------------------------------------------------------
# Caminhos do projeto (calculados em tempo de import)
# ----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"


class Settings(BaseSettings):
    """Configurações globais do projeto.

    Todos os campos podem ser sobrescritos por variáveis de ambiente
    com o mesmo nome (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- Reprodutibilidade -----
    random_seed: int = 42

    # ----- Dataset -----
    dataset_name: str = "telco_churn"
    dataset_filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    target_column: str = "Churn"
    test_size: float = 0.2
    val_size: float = 0.15  # Fração do treino que vira validação
    cv_folds: int = 5

    # ----- MLflow -----
    mlflow_tracking_uri: str = Field(default_factory=lambda: f"file://{MLRUNS_DIR}")
    mlflow_experiment_name: str = "churn-prediction"

    # ----- MLP (PyTorch) -----
    mlp_hidden_dims: tuple[int, ...] = (128, 64, 32)
    mlp_dropout: float = 0.3
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-5
    mlp_batch_size: int = 64
    mlp_max_epochs: int = 100
    mlp_early_stopping_patience: int = 10

    # ----- API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"
    api_model_path: Path = MODELS_DIR / "best_model.pt"
    api_pipeline_path: Path = MODELS_DIR / "preprocessing_pipeline.joblib"

    # ----- Logging -----
    log_level: str = "INFO"
    log_format: str = "json"  # "json" ou "console"

    # ----- Análise de custo (negócio) -----
    # Custo médio: perder cliente que iria cancelar (FN) é mais caro
    # que oferecer desconto a quem ficaria mesmo (FP).
    cost_false_negative: float = 500.0  # Perda de receita média anual (R$)
    cost_false_positive: float = 50.0  # Custo da campanha de retenção (R$)


# Instância singleton — importar `settings` em todos os módulos
settings = Settings()
