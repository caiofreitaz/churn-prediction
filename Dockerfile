# =============================================================================
# Churn Predictor — Multi-stage Dockerfile
# =============================================================================
# Build:  docker build -t churn-predictor:latest .
# Run:    docker run -p 8000:8000 churn-predictor:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Dependências de sistema necessárias para compilar wheels (ex.: numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências em diretório isolado
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Metadados
LABEL org.opencontainers.image.title="churn-predictor"
LABEL org.opencontainers.image.description="ML API for telecom churn prediction"
LABEL org.opencontainers.image.source="https://github.com/your-org/churn-prediction"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Pacote curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash appuser

# Copia dependências e código
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# PATH para os binários instalados via pip --user
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json

USER appuser

EXPOSE 8000

# Healthcheck: API responde em /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Inicia API em modo produção
CMD ["uvicorn", "churn_predictor.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-config", "/dev/null"]
