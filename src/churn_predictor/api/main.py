"""Aplicação FastAPI para servir o modelo de predição de churn.

Endpoints:
    GET  /             — informações da API
    GET  /health       — health check (modelo carregado, uptime)
    POST /predict      — predição individual
    POST /predict/batch — predição em lote (até 1000 clientes)

Uso:
    uvicorn churn_predictor.api.main:app --reload
    # ou
    churn-serve
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from churn_predictor import __version__
from churn_predictor.api.middleware import LoggingMiddleware
from churn_predictor.api.model_loader import (
    ModelBundle,
    classify_risk,
    load_model_bundle,
)
from churn_predictor.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from churn_predictor.utils.config import settings
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)

# Estado global da aplicação (thread-safe via lifespan)
_app_state: dict[str, Any] = {
    "model_bundle": None,
    "start_time": None,
}


# =============================================================================
# Lifespan: carrega modelo no startup
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo no startup e libera no shutdown."""
    logger.info("api_starting")
    _app_state["start_time"] = time.time()

    try:
        bundle = load_model_bundle()
        _app_state["model_bundle"] = bundle
        logger.info("api_ready", model_loaded=True)
    except FileNotFoundError as e:
        logger.warning("model_not_found", error=str(e))
        # Permite que a API suba mesmo sem modelo (para health check)
        _app_state["model_bundle"] = None

    yield

    logger.info("api_shutdown")


# =============================================================================
# App
# =============================================================================
app = FastAPI(
    title="Churn Predictor API",
    description="API de inferência para predição de churn (telecom).",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)


# =============================================================================
# Exception handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "HTTP error",
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID")
    logger.exception("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


# =============================================================================
# Helpers
# =============================================================================
def _get_bundle() -> ModelBundle:
    """Retorna o ModelBundle carregado ou levanta 503."""
    bundle = _app_state.get("model_bundle")
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado. Verifique se os artefatos existem.",
        )
    return bundle


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/", tags=["info"])
async def root() -> dict[str, str]:
    """Informações básicas da API."""
    return {
        "service": "churn-predictor",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["info"],
)
async def health() -> HealthResponse:
    """Health check com detalhes do estado do modelo."""
    bundle = _app_state.get("model_bundle")
    start_time = _app_state.get("start_time", time.time())

    is_healthy = bundle is not None
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        pipeline_loaded=is_healthy,
        model_version=__version__,
        uptime_seconds=time.time() - start_time,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    tags=["inference"],
)
async def predict(features: CustomerFeatures) -> PredictionResponse:
    """Prediz probabilidade de churn para um único cliente."""
    bundle = _get_bundle()

    # Converte para DataFrame (1 linha)
    df = pd.DataFrame([features.model_dump()])

    try:
        probas, predictions = bundle.predict(df)
    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro durante predição: {e}",
        ) from e

    proba = float(probas[0])
    pred = int(predictions[0])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        risk_level=classify_risk(proba),
        threshold_used=bundle.threshold,
        model_version=__version__,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Prediz churn para múltiplos clientes (até 1000)."""
    bundle = _get_bundle()

    df = pd.DataFrame([c.model_dump() for c in request.customers])

    try:
        probas, predictions = bundle.predict(df)
    except Exception as e:
        logger.error("batch_prediction_failed", error=str(e), n_customers=len(df))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro durante predição em lote: {e}",
        ) from e

    results = [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            churn_prediction=int(pred),
            risk_level=classify_risk(float(p)),
            threshold_used=bundle.threshold,
            model_version=__version__,
        )
        for p, pred in zip(probas, predictions, strict=True)
    ]

    return BatchPredictionResponse(
        predictions=results,
        n_processed=len(results),
        n_high_risk=sum(1 for r in results if r.risk_level == "high"),
    )


# =============================================================================
# Entry-point para `churn-serve`
# =============================================================================
def run() -> None:
    """Inicia uvicorn programaticamente."""
    import uvicorn

    uvicorn.run(
        "churn_predictor.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.api_log_level,
        reload=False,
    )


if __name__ == "__main__":
    run()
