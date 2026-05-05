"""Métricas de avaliação e análise de trade-off de custo."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_predictor.utils.config import settings


@dataclass
class ClassificationMetrics:
    """Container de métricas de classificação binária."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calcula todas as métricas de classificação.

    Args:
        y_true: Targets reais (0/1).
        y_proba: Probabilidades preditas (entre 0 e 1).
        threshold: Threshold para binarização.

    Returns:
        ClassificationMetrics com todas as métricas.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        threshold=float(threshold),
    )


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
) -> dict[str, float]:
    """Calcula custo de negócio com base em FN/FP.

    Hipóteses:
    - FN (cliente que iria cancelar e não foi detectado) → perda de receita.
    - FP (cliente que ficaria mesmo, mas recebeu retenção) → custo de campanha.

    Args:
        y_true: Targets reais.
        y_pred: Predições binárias.
        cost_fn: Custo unitário de FN (default: settings.cost_false_negative).
        cost_fp: Custo unitário de FP (default: settings.cost_false_positive).

    Returns:
        Dict com custos detalhados.
    """
    cost_fn = cost_fn if cost_fn is not None else settings.cost_false_negative
    cost_fp = cost_fp if cost_fp is not None else settings.cost_false_positive

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = fn * cost_fn + fp * cost_fp

    # Custo de não fazer nada (deixar todos os FN acontecerem)
    baseline_cost = (fn + tp) * cost_fn

    # Economia gerada pelo modelo (clientes salvos x custo evitado, líquido de FP)
    savings = tp * cost_fn - fp * cost_fp

    return {
        "total_cost": float(total_cost),
        "baseline_cost_no_model": float(baseline_cost),
        "savings_vs_no_model": float(savings),
        "cost_per_fn": float(cost_fn),
        "cost_per_fp": float(cost_fp),
        "n_fn": int(fn),
        "n_fp": int(fp),
        "n_tp": int(tp),
        "n_tn": int(tn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Busca threshold que minimiza o custo de negócio.

    Returns:
        (threshold_ótimo, custo_mínimo).
    """
    cost_fn = cost_fn if cost_fn is not None else settings.cost_false_negative
    cost_fp = cost_fp if cost_fp is not None else settings.cost_false_positive

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        result = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
        costs.append(result["total_cost"])

    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
