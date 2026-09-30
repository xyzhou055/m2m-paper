"""Privacy objective terms used by the AGDA primal updates and selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import FeatureRemovalLayer, M2MAGDAModel


@dataclass(frozen=True)
class TraceTerms:
    normalized: torch.Tensor
    raw: torch.Tensor
    gated_covariance_trace: torch.Tensor


@dataclass(frozen=True)
class PrivacyTerms:
    optimization_objective: torch.Tensor
    selection_objective: torch.Tensor
    trace_penalty: torch.Tensor
    normalized_trace: torch.Tensor
    raw_trace: torch.Tensor
    gated_covariance_trace: torch.Tensor
    sparsity_penalty: torch.Tensor
    binary_penalty: torch.Tensor
    gate_regularization_scale: float


def compute_training_covariance(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError("training inputs must be a two-dimensional tensor")
    if len(inputs) < 2:
        raise ValueError("at least two training rows are required for covariance")
    if not torch.is_floating_point(inputs):
        inputs = inputs.float()
    if not torch.isfinite(inputs).all():
        raise ValueError("training inputs must contain only finite values")
    centered = inputs - inputs.mean(dim=0, keepdim=True)
    # The paper defines empirical covariance with probability mass 1/N.
    return centered.T @ centered / len(inputs)


def normalized_trace_penalty(
    covariance: torch.Tensor,
    weights: torch.Tensor,
    gates: torch.Tensor,
    epsilon: float = 1e-8,
    *,
    detach_gates: bool = False,
    detach_denominator: bool = False,
) -> TraceTerms:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if weights.ndim != 2 or weights.shape[0] != covariance.shape[0]:
        raise ValueError("weights must have shape (input_dim, aggregated_dim)")
    if gates.ndim != 1 or gates.shape[0] != covariance.shape[0]:
        raise ValueError("gates must have one value per input feature")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    trace_gates = gates.detach() if detach_gates else gates
    gated_covariance = trace_gates[:, None] * covariance * trace_gates[None, :]
    raw = torch.trace(weights.T @ gated_covariance @ weights)
    denominator = torch.trace(gated_covariance).clamp_min(epsilon)
    if detach_denominator:
        denominator = denominator.detach()
    return TraceTerms(
        normalized=raw / denominator,
        raw=raw,
        gated_covariance_trace=denominator,
    )


def normalized_sparsity_penalty(
    feature_removal: FeatureRemovalLayer,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = feature_removal.expected_open_probabilities()
    if feature_weights is None:
        return probabilities.mean()
    weights = torch.as_tensor(
        feature_weights, dtype=probabilities.dtype, device=probabilities.device
    )
    if weights.shape != probabilities.shape:
        raise ValueError("feature weights must have one value per input feature")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("feature weights must be finite and non-negative")
    return torch.dot(weights, probabilities) / weights.sum().clamp_min(1e-12)


def binary_gate_penalty(feature_removal: FeatureRemovalLayer) -> torch.Tensor:
    probabilities = feature_removal.expected_open_probabilities()
    return (4.0 * probabilities * (1.0 - probabilities)).mean()


def gate_regularization_scale(
    iteration: int, warmup_iterations: int = 0, ramp_iterations: int = 0
) -> float:
    if iteration < 0 or warmup_iterations < 0 or ramp_iterations < 0:
        raise ValueError("iteration, warmup, and ramp must be non-negative")
    if iteration <= warmup_iterations:
        return 0.0
    if ramp_iterations == 0:
        return 1.0
    return min(1.0, (iteration - warmup_iterations) / ramp_iterations)


def empirical_feature_entropies(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2 or len(inputs) == 0:
        raise ValueError("inputs must be a non-empty matrix")
    weights = []
    detached = inputs.detach().cpu()
    for column in range(detached.shape[1]):
        _, counts = torch.unique(detached[:, column], return_counts=True)
        probabilities = counts.double() / len(detached)
        entropy = -(probabilities * probabilities.log()).sum()
        weights.append(entropy)
    return torch.stack(weights).to(dtype=inputs.dtype, device=inputs.device)


def empirical_feature_variances(inputs: torch.Tensor) -> torch.Tensor:
    """Return population variance weights fitted on training rows only."""

    if inputs.ndim != 2 or len(inputs) == 0:
        raise ValueError("inputs must be a non-empty matrix")
    if not torch.is_floating_point(inputs):
        inputs = inputs.float()
    return inputs.var(dim=0, correction=0)


def privacy_terms(
    model: M2MAGDAModel,
    covariance: torch.Tensor,
    gates: torch.Tensor,
    feature_weights: torch.Tensor | None,
    *,
    trace_weight: float,
    sparsity_weight: float,
    binary_weight: float,
    trace_normalization: str,
    trace_epsilon: float,
    detach_trace_gates: bool,
    detach_trace_denominator: bool,
    gate_scale: float,
) -> PrivacyTerms:
    if trace_normalization not in {"raw", "normalized"}:
        raise ValueError("trace_normalization must be 'raw' or 'normalized'")
    if not 0.0 <= gate_scale <= 1.0:
        raise ValueError("gate_scale must be in [0, 1]")
    trace = normalized_trace_penalty(
        covariance,
        model.minimizer.aggregation.weight,
        gates,
        trace_epsilon,
        detach_gates=detach_trace_gates,
        detach_denominator=detach_trace_denominator,
    )
    trace_penalty = (
        trace.normalized if trace_normalization == "normalized" else trace.raw
    )
    sparsity = normalized_sparsity_penalty(
        model.minimizer.feature_removal, feature_weights
    )
    binary = binary_gate_penalty(model.minimizer.feature_removal)
    trace_component = trace_weight * trace_penalty
    gate_component = sparsity_weight * sparsity + binary_weight * binary
    return PrivacyTerms(
        optimization_objective=trace_component + gate_scale * gate_component,
        selection_objective=trace_component + gate_component,
        trace_penalty=trace_penalty,
        normalized_trace=trace.normalized,
        raw_trace=trace.raw,
        gated_covariance_trace=trace.gated_covariance_trace,
        sparsity_penalty=sparsity,
        binary_penalty=binary,
        gate_regularization_scale=gate_scale,
    )


def privacy_terms_to_floats(terms: PrivacyTerms) -> dict[str, float]:
    return {
        "optimization_privacy_objective": float(
            terms.optimization_objective.detach().item()
        ),
        "selection_privacy_objective": float(terms.selection_objective.detach().item()),
        "trace_penalty": float(terms.trace_penalty.detach().item()),
        "normalized_trace": float(terms.normalized_trace.detach().item()),
        "raw_trace": float(terms.raw_trace.detach().item()),
        "gated_covariance_trace": float(terms.gated_covariance_trace.detach().item()),
        "sparsity_penalty": float(terms.sparsity_penalty.detach().item()),
        "binary_penalty": float(terms.binary_penalty.detach().item()),
        "gate_regularization_scale": terms.gate_regularization_scale,
    }
