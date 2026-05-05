"""Modelos baseline (Scikit-Learn) para comparação com a MLP.

Inclui:
- DummyClassifier (estratégia mais ingênua possível)
- LogisticRegression (baseline linear)
- RandomForestClassifier (baseline não-linear / árvore)
- GradientBoostingClassifier (baseline ensemble)
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_predictor.utils.config import settings


def get_baseline_models(random_state: int | None = None) -> dict[str, Any]:
    """Retorna dicionário {nome: estimador} dos baselines.

    Args:
        random_state: Seed (default: settings.random_seed).

    Returns:
        Dict com 4 modelos prontos para `.fit(X, y)`.
    """
    rs = random_state if random_state is not None else settings.random_seed

    return {
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=rs),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=rs,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=rs,
        ),
    }
