"""The essential M2M release model: hard-concrete gates plus projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parametrizations

from .classifiers import ClassifierSpec, MLPClassifier, build_classifier


class FeatureRemovalLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        temperature: float = 0.1,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not gamma < 0 < zeta:
            raise ValueError("hard-concrete bounds must satisfy gamma < 0 < zeta")
        self.input_dim = input_dim
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.log_alpha = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.log_alpha, mean=0.0, std=0.01)

    def expected_open_probabilities(self) -> torch.Tensor:
        threshold_logit = math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - self.temperature * threshold_logit)

    def deterministic_gates(self) -> torch.Tensor:
        locations = torch.sigmoid(self.log_alpha)
        stretched = locations * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def sample_gates(self, generator: torch.Generator | None = None) -> torch.Tensor:
        uniform = torch.rand(
            self.log_alpha.shape,
            dtype=self.log_alpha.dtype,
            device=self.log_alpha.device,
            generator=generator,
        ).clamp(1e-6, 1.0 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        locations = torch.sigmoid((self.log_alpha + logistic) / self.temperature)
        stretched = locations * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.sample_gates(generator) if sample else self.deterministic_gates()
        return inputs * gates, gates


class LinearAggregationLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, noise_std: float) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("aggregation dimensions must be positive")
        if output_dim > input_dim:
            raise ValueError("aggregated_dim cannot exceed input_dim")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = float(noise_std)
        self.linear = parametrizations.orthogonal(
            nn.Linear(input_dim, output_dim, bias=False)
        )

    @property
    def weight(self) -> torch.Tensor:
        """Projection matrix W with shape ``(input_dim, aggregated_dim)``."""

        return self.linear.weight.T

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        add_noise: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        released = self.linear(inputs)
        if add_noise and self.noise_std > 0:
            noise = torch.randn(
                released.shape,
                dtype=released.dtype,
                device=released.device,
                generator=generator,
            )
            released = released + self.noise_std * noise
        return released


class M2MMinimizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        aggregated_dim: int,
        *,
        noise_std: float,
        gate_temperature: float,
        gate_gamma: float,
        gate_zeta: float,
    ) -> None:
        super().__init__()
        self.feature_removal = FeatureRemovalLayer(
            input_dim,
            temperature=gate_temperature,
            gamma=gate_gamma,
            zeta=gate_zeta,
        )
        self.aggregation = LinearAggregationLayer(
            input_dim, aggregated_dim, noise_std=noise_std
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample_gates: bool,
        add_noise: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gated, gates = self.feature_removal(
            inputs, sample=sample_gates, generator=generator
        )
        released = self.aggregation(gated, add_noise=add_noise, generator=generator)
        return released, gates, gated


DownstreamMLP = MLPClassifier


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int
    aggregated_dim: int
    hidden_dim: int
    num_classes: int
    classifier: str
    noise_std: float
    gate_temperature: float
    gate_gamma: float
    gate_zeta: float
    ft_token_dim: int = 16
    ft_num_heads: int = 2
    ft_num_layers: int = 1
    ft_feedforward_dim: int = 32
    ft_dropout: float = 0.1


class M2MAGDAModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.minimizer = M2MMinimizer(
            spec.input_dim,
            spec.aggregated_dim,
            noise_std=spec.noise_std,
            gate_temperature=spec.gate_temperature,
            gate_gamma=spec.gate_gamma,
            gate_zeta=spec.gate_zeta,
        )
        self.classifier = build_classifier(
            ClassifierSpec(
                architecture=spec.classifier,
                input_dim=spec.aggregated_dim,
                hidden_dim=spec.hidden_dim,
                num_classes=spec.num_classes,
                ft_token_dim=spec.ft_token_dim,
                ft_num_heads=spec.ft_num_heads,
                ft_num_layers=spec.ft_num_layers,
                ft_feedforward_dim=spec.ft_feedforward_dim,
                ft_dropout=spec.ft_dropout,
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample_gates: bool | None = None,
        add_noise: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_gates is None:
            sample_gates = self.training
        if add_noise is None:
            add_noise = self.training and self.spec.noise_std > 0
        released, gates, gated = self.minimizer(
            inputs,
            sample_gates=sample_gates,
            add_noise=add_noise,
            generator=generator,
        )
        return self.classifier(released), released, gates

    def transform(
        self,
        inputs: torch.Tensor,
        *,
        deterministic_gates: bool = True,
        add_noise: bool = False,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        """Materialize the reusable minimized representation."""

        generator = None
        if noise_seed is not None:
            generator = torch.Generator(device=inputs.device).manual_seed(noise_seed)
        released, _, _ = self.minimizer(
            inputs,
            sample_gates=not deterministic_gates,
            add_noise=add_noise,
            generator=generator,
        )
        return released

    def predict(
        self,
        inputs: torch.Tensor,
        *,
        add_noise: bool = False,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        released = self.transform(
            inputs,
            deterministic_gates=True,
            add_noise=add_noise,
            noise_seed=noise_seed,
        )
        return self.classifier(released).argmax(dim=1)
