"""Script para baixar o dataset Telco Customer Churn.

O dataset original da IBM está disponível publicamente em vários
mirrors. Este script tenta múltiplas fontes para garantir disponibilidade.

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Permite executar standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_predictor.utils.config import RAW_DATA_DIR, settings  # noqa: E402
from churn_predictor.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


# Mirrors públicos do dataset Telco Churn
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv",
]


def download_dataset() -> Path:
    """Baixa o dataset e salva em data/raw/."""
    output_path = RAW_DATA_DIR / settings.dataset_filename
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("dataset_already_exists", path=str(output_path))
        return output_path

    last_error: Exception | None = None
    for url in DATASET_URLS:
        try:
            logger.info("downloading_dataset", url=url)
            df = pd.read_csv(url)
            df.to_csv(output_path, index=False)
            logger.info(
                "dataset_downloaded",
                path=str(output_path),
                shape=df.shape,
            )
            return output_path
        except Exception as e:  # noqa: BLE001
            logger.warning("download_failed", url=url, error=str(e))
            last_error = e
            continue

    raise RuntimeError(
        f"Não foi possível baixar o dataset de nenhum mirror. "
        f"Último erro: {last_error}"
    )


if __name__ == "__main__":
    path = download_dataset()
    print(f"\n✓ Dataset disponível em: {path}")
