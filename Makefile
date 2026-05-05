# ============================================================
# Churn Predictor - Makefile
# ============================================================
# Uso: make <target>
# ============================================================

.PHONY: help install install-dev clean lint format test test-fast \
        test-cov train serve mlflow-ui download-data setup all

# Cor para output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
NC     := \033[0m # No Color

# Python e diretórios
PYTHON       := python
SRC_DIR      := src/churn_predictor
TESTS_DIR    := tests
NOTEBOOKS    := notebooks

help: ## Mostra este help
	@echo "$(GREEN)Churn Predictor - Comandos disponíveis$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================
# Setup
# ============================================================
install: ## Instala dependências de produção
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev: ## Instala dependências de desenvolvimento
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

setup: install-dev ## Setup completo do ambiente
	@echo "$(GREEN)✓ Ambiente configurado$(NC)"
	@echo "  Próximo passo: make download-data"

download-data: ## Baixa o dataset Telco Customer Churn
	$(PYTHON) scripts/download_data.py

# ============================================================
# Code Quality
# ============================================================
lint: ## Roda ruff check
	ruff check $(SRC_DIR) $(TESTS_DIR)

format: ## Formata código com ruff
	ruff format $(SRC_DIR) $(TESTS_DIR)
	ruff check --fix $(SRC_DIR) $(TESTS_DIR)

# ============================================================
# Testing
# ============================================================
test: ## Roda todos os testes
	pytest

test-fast: ## Roda apenas testes rápidos (exclui slow)
	pytest -m "not slow"

test-smoke: ## Roda apenas smoke tests
	pytest -m smoke -v

test-cov: ## Roda testes com relatório de cobertura HTML
	pytest --cov-report=html
	@echo "$(GREEN)Relatório em htmlcov/index.html$(NC)"

# ============================================================
# Training & Serving
# ============================================================
train: ## Treina o modelo MLP completo
	$(PYTHON) -m churn_predictor.training.train

train-baselines: ## Treina apenas baselines
	$(PYTHON) -m churn_predictor.training.train_baselines

serve: ## Inicia API FastAPI localmente
	uvicorn churn_predictor.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Inicia API em modo produção
	uvicorn churn_predictor.api.main:app --host 0.0.0.0 --port 8000 --workers 4

mlflow-ui: ## Abre interface do MLflow
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ============================================================
# Cleanup
# ============================================================
clean: ## Remove arquivos temporários
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage build dist
	@echo "$(GREEN)✓ Limpeza concluída$(NC)"

clean-all: clean ## Remove tudo, incluindo modelos e MLflow
	rm -rf mlruns models/*.pt models/*.joblib
	@echo "$(GREEN)✓ Limpeza completa concluída$(NC)"

# ============================================================
# CI Pipeline
# ============================================================
ci: lint test ## Roda lint + tests (usado no CI)
	@echo "$(GREEN)✓ CI passou$(NC)"

all: clean install-dev lint test ## Setup completo + validação
	@echo "$(GREEN)✓ Tudo pronto$(NC)"
