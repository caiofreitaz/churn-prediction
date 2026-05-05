"""Loop de treinamento da MLP com early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from churn_predictor.evaluation.metrics import compute_metrics
from churn_predictor.models.mlp import ChurnMLP
from churn_predictor.utils.logging import get_logger
from churn_predictor.utils.seeds import get_device

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Histórico do treinamento."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_pr_auc: float = -1.0
    stopped_early: bool = False


class EarlyStopping:
    """Para o treinamento se a métrica de validação parar de melhorar."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """Calcula pos_weight para BCEWithLogitsLoss em dados desbalanceados."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Treina por uma época. Retorna loss médio."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Avalia modelo. Retorna (loss, y_true, y_proba)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probas: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

            probas = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probas.append(probas)
            all_targets.append(y_batch.cpu().numpy().ravel())

    avg_loss = total_loss / n_samples
    y_proba = np.concatenate(all_probas)
    y_true = np.concatenate(all_targets)
    return avg_loss, y_true, y_proba


def train_mlp(
    model: ChurnMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    use_pos_weight: bool = True,
    device: torch.device | None = None,
) -> tuple[ChurnMLP, TrainingHistory]:
    """Treina a MLP com early stopping e tracking.

    Args:
        model: Instância de ChurnMLP.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        y_train: Array de labels de treino (para calcular pos_weight).
        max_epochs: Número máximo de épocas.
        learning_rate: Taxa de aprendizado inicial.
        weight_decay: L2 regularization.
        early_stopping_patience: Paciência para early stopping (em épocas).
        use_pos_weight: Se True, usa pos_weight no BCE para desbalanceamento.
        device: Device PyTorch (default: auto-detecta).

    Returns:
        (model treinado com melhores pesos, history).
    """
    device = device if device is not None else get_device()
    model = model.to(device)

    pos_weight = _compute_pos_weight(y_train).to(device) if use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode="max")

    history = TrainingHistory()
    best_state_dict = copy.deepcopy(model.state_dict())

    logger.info(
        "training_started",
        max_epochs=max_epochs,
        device=str(device),
        n_parameters=model.num_parameters(),
        pos_weight=float(pos_weight) if pos_weight is not None else None,
    )

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_val_true, y_val_proba = evaluate(model, val_loader, criterion, device)

        val_metrics = compute_metrics(y_val_true, y_val_proba)
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_pr_auc.append(val_metrics.pr_auc)
        history.val_roc_auc.append(val_metrics.roc_auc)
        history.learning_rate.append(current_lr)

        logger.info(
            "epoch_completed",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_pr_auc=round(val_metrics.pr_auc, 4),
            val_roc_auc=round(val_metrics.roc_auc, 4),
            lr=current_lr,
        )

        # Salva melhor modelo (baseado em PR-AUC)
        if val_metrics.pr_auc > history.best_val_pr_auc:
            history.best_val_pr_auc = val_metrics.pr_auc
            history.best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        # Scheduler
        scheduler.step(val_metrics.pr_auc)

        # Early stopping
        if early_stopper(val_metrics.pr_auc):
            logger.info(
                "early_stopping_triggered",
                epoch=epoch,
                best_epoch=history.best_epoch,
                best_val_pr_auc=round(history.best_val_pr_auc, 4),
            )
            history.stopped_early = True
            break

    # Restaura melhores pesos
    model.load_state_dict(best_state_dict)
    logger.info(
        "training_completed",
        best_epoch=history.best_epoch,
        best_val_pr_auc=round(history.best_val_pr_auc, 4),
        total_epochs=len(history.train_loss),
    )

    return model, history
