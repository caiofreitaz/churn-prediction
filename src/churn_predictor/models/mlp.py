"""Multi-Layer Perceptron (MLP) em PyTorch para classificação de churn.

Arquitetura configurável com BatchNorm + Dropout para regularização.
Usa BCEWithLogitsLoss (mais estável que BCELoss + sigmoid manual).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP para classificação binária de churn.

    Args:
        input_dim: Número de features de entrada (após pré-processamento).
        hidden_dims: Tupla com tamanhos das camadas ocultas.
        dropout: Taxa de dropout (entre 0 e 1).
        use_batchnorm: Se True, aplica BatchNorm1d após cada camada linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout deve estar em [0, 1), recebido {dropout}")
        if not hidden_dims:
            raise ValueError("hidden_dims não pode estar vazio")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (1 logit, sem sigmoid — usar BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização Kaiming/He para camadas com ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor (batch_size, input_dim).

        Returns:
            Logits (batch_size, 1). Aplique sigmoid externamente para
            obter probabilidades.
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades (sigmoid dos logits).

        Útil para inferência. Em treino, prefira usar logits direto
        com BCEWithLogitsLoss.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Retorna predições binárias 0/1."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()

    def num_parameters(self) -> int:
        """Total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
