"""Script principal de treinamento.

Orquestra:
1. Carrega e divide dados
2. Aplica pipeline de pré-processamento
3. Treina baselines + MLP
4. Avalia e compara
5. Registra tudo no MLflow
6. Persiste o melhor modelo

Uso:
    python -m churn_predictor.training.train
    # ou
    churn-train
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import mlflow
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from churn_predictor.data.loader import load_and_split
from churn_predictor.evaluation.metrics import (
    compute_business_cost,
    compute_metrics,
    find_optimal_threshold,
)
from churn_predictor.features.pipeline import build_preprocessing_pipeline
from churn_predictor.models.baselines import get_baseline_models
from churn_predictor.models.mlp import ChurnMLP
from churn_predictor.training.dataset import make_dataloader
from churn_predictor.training.trainer import train_mlp
from churn_predictor.utils.config import MODELS_DIR, settings
from churn_predictor.utils.logging import get_logger
from churn_predictor.utils.seeds import get_device, set_seeds

logger = get_logger(__name__)


# =============================================================================
# Setup MLflow
# =============================================================================
def setup_mlflow() -> None:
    """Configura MLflow tracking URI e experimento."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(
        "mlflow_configured",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )


# =============================================================================
# Treinamento dos baselines
# =============================================================================
def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Treina e avalia todos os baselines, registrando no MLflow."""
    results = {}
    models = get_baseline_models()

    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline_{name}", nested=False):
            logger.info("training_baseline", model=name)

            # Tags e metadata
            mlflow.set_tag("model_type", "baseline")
            mlflow.set_tag("model_family", name)
            mlflow.log_params({"dataset_hash": metadata["dataset_hash"]})

            # Validação cruzada estratificada
            skf = StratifiedKFold(
                n_splits=settings.cv_folds,
                shuffle=True,
                random_state=settings.random_seed,
            )
            cv_pr_aucs = []
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                model_cv = type(model)(**model.get_params())
                model_cv.fit(X_train[train_idx], y_train[train_idx])
                if hasattr(model_cv, "predict_proba"):
                    proba = model_cv.predict_proba(X_train[val_idx])[:, 1]
                else:
                    proba = model_cv.predict(X_train[val_idx])
                m = compute_metrics(y_train[val_idx], proba)
                cv_pr_aucs.append(m.pr_auc)
                mlflow.log_metric(f"cv_fold_{fold_idx}_pr_auc", m.pr_auc)

            mlflow.log_metric("cv_pr_auc_mean", float(np.mean(cv_pr_aucs)))
            mlflow.log_metric("cv_pr_auc_std", float(np.std(cv_pr_aucs)))

            # Treina no full train
            model.fit(X_train, y_train)

            # Avalia em val e test
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]

            val_metrics = compute_metrics(y_val, val_proba)
            test_metrics = compute_metrics(y_test, test_proba)

            # Custo de negócio no test
            test_pred = (test_proba >= 0.5).astype(int)
            cost = compute_business_cost(y_test, test_pred)

            # Log no MLflow
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.to_dict().items() if isinstance(v, float)}
            )
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.to_dict().items() if isinstance(v, float)}
            )
            mlflow.log_metrics(
                {f"cost_{k}": v for k, v in cost.items() if isinstance(v, int | float)}
            )

            mlflow.sklearn.log_model(model, name="model")

            results[name] = {
                "val_metrics": val_metrics.to_dict(),
                "test_metrics": test_metrics.to_dict(),
                "cost": cost,
                "cv_pr_auc_mean": float(np.mean(cv_pr_aucs)),
            }
            logger.info(
                "baseline_completed",
                model=name,
                test_pr_auc=round(test_metrics.pr_auc, 4),
                test_f1=round(test_metrics.f1, 4),
            )

    return results


# =============================================================================
# Treinamento da MLP
# =============================================================================
def train_mlp_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], ChurnMLP]:
    """Treina MLP em PyTorch com tracking completo no MLflow."""

    with mlflow.start_run(run_name="mlp_pytorch"):
        logger.info("training_mlp_started")

        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("model_family", "mlp_pytorch")
        mlflow.set_tag("framework", "pytorch")

        # Hiperparâmetros
        params = {
            "hidden_dims": str(settings.mlp_hidden_dims),
            "dropout": settings.mlp_dropout,
            "learning_rate": settings.mlp_learning_rate,
            "weight_decay": settings.mlp_weight_decay,
            "batch_size": settings.mlp_batch_size,
            "max_epochs": settings.mlp_max_epochs,
            "early_stopping_patience": settings.mlp_early_stopping_patience,
            "input_dim": X_train.shape[1],
            "dataset_hash": metadata["dataset_hash"],
            "random_seed": settings.random_seed,
        }
        mlflow.log_params(params)

        # Constrói modelo
        model = ChurnMLP(
            input_dim=X_train.shape[1],
            hidden_dims=settings.mlp_hidden_dims,
            dropout=settings.mlp_dropout,
        )
        mlflow.log_param("n_parameters", model.num_parameters())

        # DataLoaders
        train_loader = make_dataloader(
            X_train, y_train, batch_size=settings.mlp_batch_size, shuffle=True
        )
        val_loader = make_dataloader(
            X_val, y_val, batch_size=settings.mlp_batch_size, shuffle=False
        )
        test_loader = make_dataloader(
            X_test, y_test, batch_size=settings.mlp_batch_size, shuffle=False
        )

        # Treina
        device = get_device()
        model, history = train_mlp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            y_train=y_train,
            max_epochs=settings.mlp_max_epochs,
            learning_rate=settings.mlp_learning_rate,
            weight_decay=settings.mlp_weight_decay,
            early_stopping_patience=settings.mlp_early_stopping_patience,
            device=device,
        )

        # Log curvas de treinamento
        for epoch, (tl, vl, prc, roc) in enumerate(
            zip(history.train_loss, history.val_loss, history.val_pr_auc, history.val_roc_auc, strict=True),
            start=1,
        ):
            mlflow.log_metric("train_loss", tl, step=epoch)
            mlflow.log_metric("val_loss", vl, step=epoch)
            mlflow.log_metric("val_pr_auc", prc, step=epoch)
            mlflow.log_metric("val_roc_auc", roc, step=epoch)

        mlflow.log_metric("best_epoch", history.best_epoch)
        mlflow.log_metric("stopped_early", int(history.stopped_early))

        # Avalia no val e test
        import torch.nn as nn

        from churn_predictor.training.trainer import evaluate

        criterion = nn.BCEWithLogitsLoss()
        _, _, val_proba = evaluate(model, val_loader, criterion, device)
        _, _, test_proba = evaluate(model, test_loader, criterion, device)

        val_metrics = compute_metrics(y_val, val_proba)
        test_metrics = compute_metrics(y_test, test_proba)

        # Threshold ótimo via custo de negócio
        opt_threshold, _opt_cost = find_optimal_threshold(y_val, val_proba)
        test_metrics_opt = compute_metrics(y_test, test_proba, threshold=opt_threshold)
        test_pred_opt = (test_proba >= opt_threshold).astype(int)
        cost_opt = compute_business_cost(y_test, test_pred_opt)

        # Log no MLflow
        mlflow.log_metrics(
            {f"val_{k}": v for k, v in val_metrics.to_dict().items() if isinstance(v, float)}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.to_dict().items() if isinstance(v, float)}
        )
        mlflow.log_metric("optimal_threshold", opt_threshold)
        mlflow.log_metrics(
            {
                f"test_opt_{k}": v
                for k, v in test_metrics_opt.to_dict().items()
                if isinstance(v, float)
            }
        )
        mlflow.log_metrics(
            {f"cost_{k}": v for k, v in cost_opt.items() if isinstance(v, int | float)}
        )

        # Salva modelo
        mlflow.pytorch.log_model(model, name="model")

        logger.info(
            "mlp_training_completed",
            test_pr_auc=round(test_metrics.pr_auc, 4),
            test_f1=round(test_metrics.f1, 4),
            optimal_threshold=round(opt_threshold, 3),
        )

        return {
            "val_metrics": val_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "test_metrics_opt": test_metrics_opt.to_dict(),
            "cost_opt": cost_opt,
            "optimal_threshold": opt_threshold,
            "history": {
                "best_epoch": history.best_epoch,
                "best_val_pr_auc": history.best_val_pr_auc,
                "stopped_early": history.stopped_early,
            },
        }, model


# =============================================================================
# Persistência do melhor modelo
# =============================================================================
def save_artifacts(
    pipeline: Any,
    mlp_model: ChurnMLP,
    optimal_threshold: float,
    metadata: dict[str, Any],
) -> None:
    """Salva pipeline + modelo + metadata para a API consumir."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pipeline sklearn (joblib)
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    # Modelo PyTorch (state_dict + config)
    model_path = MODELS_DIR / "best_model.pt"
    torch.save(
        {
            "state_dict": mlp_model.state_dict(),
            "input_dim": mlp_model.input_dim,
            "hidden_dims": mlp_model.hidden_dims,
            "dropout": mlp_model.dropout,
            "use_batchnorm": mlp_model.use_batchnorm,
            "optimal_threshold": optimal_threshold,
            "metadata": metadata,
        },
        model_path,
    )

    logger.info(
        "artifacts_saved",
        pipeline=str(pipeline_path),
        model=str(model_path),
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Pipeline completo de treinamento."""
    logger.info("training_pipeline_started")
    set_seeds(settings.random_seed)
    setup_mlflow()

    # 1. Dados
    split, metadata = load_and_split()
    logger.info("data_loaded", **metadata)

    # 2. Pipeline de pré-processamento (fit no train apenas — sem leakage)
    pipeline = build_preprocessing_pipeline()
    X_train_arr = pipeline.fit_transform(split.X_train)
    X_val_arr = pipeline.transform(split.X_val)
    X_test_arr = pipeline.transform(split.X_test)
    y_train_arr = split.y_train.values
    y_val_arr = split.y_val.values
    y_test_arr = split.y_test.values

    logger.info(
        "preprocessing_done",
        train_shape=X_train_arr.shape,
        val_shape=X_val_arr.shape,
        test_shape=X_test_arr.shape,
    )

    # 3. Treina baselines
    baseline_results = train_baselines(
        X_train_arr, y_train_arr,
        X_val_arr, y_val_arr,
        X_test_arr, y_test_arr,
        metadata,
    )

    # 4. Treina MLP
    mlp_results, mlp_model = train_mlp_pipeline(
        X_train_arr, y_train_arr,
        X_val_arr, y_val_arr,
        X_test_arr, y_test_arr,
        metadata,
    )

    # 5. Comparação final
    comparison = compare_models(baseline_results, mlp_results)
    logger.info("model_comparison", **{"comparison_table": comparison})

    # 6. Salva artefatos
    save_artifacts(
        pipeline=pipeline,
        mlp_model=mlp_model,
        optimal_threshold=mlp_results["optimal_threshold"],
        metadata=metadata,
    )

    # 7. Salva relatório
    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "baselines": baseline_results,
                "mlp": mlp_results,
                "comparison": comparison,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("training_pipeline_completed", report=str(report_path))


def compare_models(
    baseline_results: dict[str, dict[str, Any]],
    mlp_results: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Gera tabela comparativa de modelos."""
    rows = []
    for name, res in baseline_results.items():
        m = res["test_metrics"]
        rows.append(
            {
                "model": name,
                "test_pr_auc": m["pr_auc"],
                "test_roc_auc": m["roc_auc"],
                "test_f1": m["f1"],
                "test_recall": m["recall"],
                "test_precision": m["precision"],
            }
        )

    m = mlp_results["test_metrics"]
    rows.append(
        {
            "model": "mlp_pytorch",
            "test_pr_auc": m["pr_auc"],
            "test_roc_auc": m["roc_auc"],
            "test_f1": m["f1"],
            "test_recall": m["recall"],
            "test_precision": m["precision"],
        }
    )

    # Ordena por PR-AUC
    rows.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    return rows


if __name__ == "__main__":
    main()
