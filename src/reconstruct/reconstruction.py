"""Train and evaluate a feature-conditioned reconstruction attacker.

This module is deliberately independent of AGDA training.  It consumes a
fixed train/validation/test release and reconstructs the original logical
categorical features.  Therefore importing or running the AGDA trainer never
constructs an attacker or uses attack metrics for checkpoint selection.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from m2m_agda.classifiers import ClassifierSpec, build_classifier

ArraySplits = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class AttackerConfig:
    """Architecture and optimization settings for the reconstruction attacker."""

    architecture: str = "mlp"
    hidden_dim: int = 512
    second_hidden_dim: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    seed: int = 0
    ft_token_dim: int = 16
    ft_num_heads: int = 2
    ft_num_layers: int = 1
    ft_feedforward_dim: int = 32
    ft_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.architecture not in {"logistic", "linear", "mlp", "ft-transformer"}:
            raise ValueError("unsupported attacker architecture")
        if (
            min(
                self.hidden_dim,
                self.second_hidden_dim,
                self.epochs,
                self.batch_size,
            )
            <= 0
        ):
            raise ValueError(
                "attacker dimensions, epochs, and batch size must be positive"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("attacker learning_rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("attacker weight_decay must be non-negative")
        # Reuse the architecture factory's complete FT-Transformer validation.
        ClassifierSpec(
            architecture=self.architecture,
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_classes=2,
            ft_token_dim=self.ft_token_dim,
            ft_num_heads=self.ft_num_heads,
            ft_num_layers=self.ft_num_layers,
            ft_feedforward_dim=self.ft_feedforward_dim,
            ft_dropout=self.ft_dropout,
        )


@dataclass(frozen=True)
class ReconstructionAttackResult:
    """Fitted attacker plus JSON-ready validation/test reconstruction metrics."""

    attacker: nn.Module
    standardized_releases: ArraySplits
    metrics: dict[str, Any]


class ReconstructionMLP(nn.Module):
    """The paper implementation's two-hidden-layer reconstruction network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FeatureConditionedAttacker(nn.Module):
    """Condition one shared classifier on the requested logical feature ID."""

    def __init__(self, classifier: nn.Module, num_features: int) -> None:
        super().__init__()
        self.classifier = classifier
        self.num_features = num_features

    def forward(self, release: torch.Tensor, feature_index: int) -> torch.Tensor:
        feature_id = F.one_hot(
            torch.full(
                (len(release),),
                feature_index,
                dtype=torch.long,
                device=release.device,
            ),
            num_classes=self.num_features,
        ).to(release.dtype)
        return self.classifier(torch.cat([release, feature_id], dim=1))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_splits(arrays: ArraySplits, name: str) -> None:
    dimensions = set()
    for array in arrays:
        if array.ndim != 2 or len(array) == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} splits must be non-empty matrices")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} splits must contain only finite values")
        dimensions.add(array.shape[1])
    if len(dimensions) != 1:
        raise ValueError(f"{name} splits must have one shared feature dimension")


def _validate_targets(
    releases: ArraySplits,
    targets: ArraySplits,
    category_counts: list[int],
) -> None:
    _validate_splits(releases, "release")
    if not category_counts or any(count < 2 for count in category_counts):
        raise ValueError("every reconstructed feature needs at least two categories")
    for release, target in zip(releases, targets, strict=True):
        if target.ndim != 2 or target.shape[1] != len(category_counts):
            raise ValueError("targets must have one column per category count")
        if len(target) != len(release):
            raise ValueError("release and target rows must align")
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError("reconstruction targets must use an integer dtype")
        for column, count in enumerate(category_counts):
            if target[:, column].min() < 0 or target[:, column].max() >= count:
                raise ValueError("reconstruction target is outside its category range")


def standardize_release(arrays: ArraySplits) -> ArraySplits:
    """Drop constant release columns, then fit scaling on training rows only."""

    _validate_splits(arrays, "release")
    keep = arrays[0].std(axis=0) > 1e-7
    if not np.any(keep):
        return tuple(np.zeros((len(array), 1), dtype=np.float32) for array in arrays)  # type: ignore[return-value]
    trimmed = tuple(array[:, keep] for array in arrays)
    mean = trimmed[0].mean(axis=0)
    scale = np.maximum(trimmed[0].std(axis=0), 1e-7)
    return tuple(
        np.ascontiguousarray((array - mean) / scale, dtype=np.float32)
        for array in trimmed
    )  # type: ignore[return-value]


def discretize_reconstruction_targets(
    arrays: ArraySplits, bins: int
) -> tuple[ArraySplits, list[int], list[list[float]], list[str]]:
    """Fit logical categories or quantile bins using training rows only."""

    _validate_splits(arrays, "input")
    if bins < 2:
        raise ValueError("reconstruction bins must be at least 2")
    training = arrays[0]
    edges_by_feature: list[np.ndarray] = []
    category_counts: list[int] = []
    strategies: list[str] = []
    for column in range(training.shape[1]):
        unique = np.unique(training[:, column])
        if len(unique) <= bins:
            edges = (unique[:-1] + unique[1:]) / 2.0
            strategies.append("preserve_training_categories")
        else:
            edges = np.quantile(
                training[:, column], np.linspace(0.0, 1.0, bins + 1)[1:-1]
            )
            strategies.append("training_quantile_bins")
        edges = np.unique(edges)
        edges_by_feature.append(edges)
        category_counts.append(len(edges) + 1)
    targets = tuple(
        np.ascontiguousarray(
            np.column_stack(
                [
                    np.searchsorted(edges, array[:, column], side="right")
                    for column, edges in enumerate(edges_by_feature)
                ]
            ),
            dtype=np.int64,
        )
        for array in arrays
    )
    return (
        targets,  # type: ignore[arg-type]
        category_counts,
        [edges.tolist() for edges in edges_by_feature],
        strategies,
    )


def _build_attacker(
    release_dim: int,
    category_counts: list[int],
    config: AttackerConfig,
) -> FeatureConditionedAttacker:
    input_dim = release_dim + len(category_counts)
    output_dim = max(category_counts)
    if config.architecture == "mlp":
        classifier: nn.Module = ReconstructionMLP(
            input_dim,
            output_dim,
            (config.hidden_dim, config.second_hidden_dim),
        )
    else:
        classifier = build_classifier(
            ClassifierSpec(
                architecture=config.architecture,
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                num_classes=output_dim,
                ft_token_dim=config.ft_token_dim,
                ft_num_heads=config.ft_num_heads,
                ft_num_layers=config.ft_num_layers,
                ft_feedforward_dim=config.ft_feedforward_dim,
                ft_dropout=config.ft_dropout,
            )
        )
    return FeatureConditionedAttacker(classifier, len(category_counts))


def train_attacker(
    release: np.ndarray,
    targets: np.ndarray,
    category_counts: list[int],
    config: AttackerConfig,
    *,
    device: torch.device,
) -> FeatureConditionedAttacker:
    """Fit one shared network conditioned on a logical feature identifier."""

    probe_splits = (release, release, release)
    target_splits = (targets, targets, targets)
    _validate_targets(probe_splits, target_splits, category_counts)
    _set_seed(config.seed)
    model = _build_attacker(release.shape[1], category_counts, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(release), torch.from_numpy(targets)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model.train()
    for _ in range(config.epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = torch.stack(
                [
                    F.cross_entropy(
                        model(inputs, column)[:, :category_count],
                        labels[:, column],
                    )
                    for column, category_count in enumerate(category_counts)
                ]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def reconstruction_accuracy(
    model: FeatureConditionedAttacker,
    release: np.ndarray,
    targets: np.ndarray,
    category_counts: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return macro accuracy across reconstructed logical features."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_targets(
        (release, release, release),
        (targets, targets, targets),
        category_counts,
    )
    correct = np.zeros(len(category_counts), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(release), batch_size):
            inputs = torch.from_numpy(release[start : start + batch_size]).to(device)
            labels = torch.from_numpy(targets[start : start + batch_size]).to(device)
            for column, category_count in enumerate(category_counts):
                logits = model(inputs, column)[:, :category_count]
                correct[column] += int(
                    logits.argmax(1).eq(labels[:, column]).sum().item()
                )
    return float(np.mean(correct / len(release)))


def majority_reconstruction_accuracy(
    training_targets: np.ndarray, category_counts: list[int]
) -> float:
    """Return macro accuracy of predicting each training-majority category."""

    if training_targets.ndim != 2 or training_targets.shape[1] != len(category_counts):
        raise ValueError("training targets and category counts do not align")
    scores = []
    for column, count in enumerate(category_counts):
        if count < 2:
            raise ValueError("every reconstructed feature needs two categories")
        frequencies = np.bincount(training_targets[:, column], minlength=count)
        scores.append(float(frequencies.max() / frequencies.sum()))
    return float(np.mean(scores))


def evaluate_reconstruction_attack(
    releases: ArraySplits,
    targets: ArraySplits,
    category_counts: list[int],
    config: AttackerConfig,
    *,
    device: torch.device,
) -> ReconstructionAttackResult:
    """Standardize a fixed release, train the attacker, and calculate NRR."""

    _validate_targets(releases, targets, category_counts)
    standardized = standardize_release(releases)
    started = time.perf_counter()
    attacker = train_attacker(
        standardized[0],
        targets[0],
        category_counts,
        config,
        device=device,
    )
    validation_reconstruction = reconstruction_accuracy(
        attacker,
        standardized[1],
        targets[1],
        category_counts,
        batch_size=config.batch_size,
        device=device,
    )
    test_reconstruction = reconstruction_accuracy(
        attacker,
        standardized[2],
        targets[2],
        category_counts,
        batch_size=config.batch_size,
        device=device,
    )
    majority = majority_reconstruction_accuracy(targets[0], category_counts)
    denominator = max(1e-12, 1.0 - majority)
    return ReconstructionAttackResult(
        attacker=attacker,
        standardized_releases=standardized,
        metrics={
            "attacker_architecture": config.architecture,
            "majority_reconstruction_accuracy": majority,
            "validation_reconstruction_accuracy": validation_reconstruction,
            "test_reconstruction_accuracy": test_reconstruction,
            "validation_nrr": (validation_reconstruction - majority) / denominator,
            "test_nrr": (test_reconstruction - majority) / denominator,
            "attack_runtime_seconds": time.perf_counter() - started,
        },
    )
