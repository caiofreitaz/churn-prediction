"""Logging estruturado via structlog.

Substitui `print()` por logs JSON em produção e logs legíveis em dev.
Uso: `logger = get_logger(__name__)` em qualquer módulo.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from churn_predictor.utils.config import settings

# Flag para garantir que a configuração só roda uma vez
_CONFIGURED = False


def configure_logging() -> None:
    """Configura logging estruturado globalmente.

    Idempotente: pode ser chamado múltiplas vezes sem efeitos colaterais.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configuração base do logging do stdlib
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Processadores comuns
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Renderer depende do formato configurado
    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retorna logger estruturado configurado.

    Args:
        name: Nome do logger (use `__name__` no módulo chamador).

    Returns:
        Logger pronto para uso com `.info(...)`, `.error(...)`, etc.
    """
    configure_logging()
    return structlog.get_logger(name)
