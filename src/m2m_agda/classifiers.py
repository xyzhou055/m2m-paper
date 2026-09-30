"""Reusable classifier architectures for downstream or attacker-side studies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

SUPPORTED_CLASSIFIERS = ("logistic", "linear", "mlp", "ft-transformer")


@dataclass(frozen=True)
class ClassifierSpec:
    """Architecture-only specification with no training or attack semantics."""

    architecture: str
    input_dim: int
    hidden_dim: int
    num_classes: int
    ft_token_dim: int = 16
    ft_num_heads: int = 2
    ft_num_layers: int = 1
    ft_feedforward_dim: int = 32
    ft_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_CLASSIFIERS:
            choices = ", ".join(SUPPORTED_CLASSIFIERS)
            raise ValueError(
                f"unknown classifier {self.architecture!r}; choose from {choices}"
            )
        if min(self.input_dim, self.hidden_dim, self.num_classes) <= 0:
            raise ValueError("classifier dimensions and class count must be positive")
        if min(self.ft_token_dim, self.ft_num_heads, self.ft_num_layers) <= 0:
            raise ValueError(
                "FT-Transformer token, head, and layer counts must be positive"
            )
        if self.ft_feedforward_dim <= 0:
            raise ValueError("FT-Transformer feedforward dimension must be positive")
        if self.ft_token_dim % self.ft_num_heads != 0:
            raise ValueError("ft_token_dim must be divisible by ft_num_heads")
        if not 0.0 <= self.ft_dropout < 1.0:
            raise ValueError("ft_dropout must be in [0, 1)")


class LogisticRegressionClassifier(nn.Module):
    """Multinomial logistic regression implemented as one affine layer."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FTTransformerClassifier(nn.Module):
    """Compact FT-Transformer-style classifier for dense numeric inputs."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_weight = nn.Parameter(torch.empty(input_dim, token_dim))
        self.feature_bias = nn.Parameter(torch.empty(input_dim, token_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.feature_weight, std=0.02)
        nn.init.normal_(self.feature_bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = inputs.unsqueeze(-1) * self.feature_weight.unsqueeze(
            0
        ) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(len(inputs), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0])


def build_classifier(spec: ClassifierSpec) -> nn.Module:
    """Build a classifier independently of the AGDA or attack training loop."""

    if spec.architecture in {"logistic", "linear"}:
        return LogisticRegressionClassifier(spec.input_dim, spec.num_classes)
    if spec.architecture == "mlp":
        return MLPClassifier(spec.input_dim, spec.hidden_dim, spec.num_classes)
    if spec.architecture == "ft-transformer":
        return FTTransformerClassifier(
            spec.input_dim,
            spec.num_classes,
            token_dim=spec.ft_token_dim,
            num_heads=spec.ft_num_heads,
            num_layers=spec.ft_num_layers,
            feedforward_dim=spec.ft_feedforward_dim,
            dropout=spec.ft_dropout,
        )
    raise AssertionError("ClassifierSpec rejected the unknown architecture")
