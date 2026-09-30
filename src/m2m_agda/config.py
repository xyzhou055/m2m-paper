"""Validated configuration objects for M2M-AGDA experiments."""

from __future__ import annotations

import copy
import itertools
import math
import re
import types
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


def _value_matches_type(value: Any, annotation: Any) -> bool:
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return type(value) in {int, float}
    if annotation is str:
        return type(value) is str
    origin = get_origin(annotation)
    if origin is tuple:
        arguments = get_args(annotation)
        item_type = arguments[0] if arguments else Any
        return isinstance(value, tuple) and all(
            _value_matches_type(item, item_type) for item in value
        )
    if origin is types.UnionType:
        return any(_value_matches_type(value, item) for item in get_args(annotation))
    if annotation is type(None):
        return value is None
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True


def _validate_declared_types(instance: Any, context: str) -> None:
    annotations = get_type_hints(type(instance))
    for item in fields(instance):
        value = getattr(instance, item.name)
        if not _value_matches_type(value, annotations[item.name]):
            raise ValueError(
                f"{context}.{item.name} has the wrong type: {type(value).__name__}"
            )


DEFAULT_MULTIPLIER_GRID = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.5,
)


@dataclass(frozen=True)
class DataConfig:
    """Input data configuration.

    ``npz`` inputs must contain the six arrays documented in ``data/README.md``.
    Synthetic data are generated locally and require no network access.
    """

    kind: str = "synthetic"
    name: str = "synthetic"
    path: str | None = None
    num_samples: int = 10_000
    num_features: int = 100
    num_relevant_features: int = 50
    num_classes: int = 2
    class_separation: float = 1.5
    label_noise: float = 0.05
    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        _validate_declared_types(self, "data")
        if self.kind not in {"synthetic", "npz"}:
            raise ValueError("data.kind must be 'synthetic' or 'npz'")
        if not self.name:
            raise ValueError("data.name must not be empty")
        if self.kind == "npz" and not self.path:
            raise ValueError("data.path is required when data.kind='npz'")
        if self.num_samples <= 0 or self.num_features <= 0:
            raise ValueError("synthetic sample and feature counts must be positive")
        if not 1 <= self.num_relevant_features <= self.num_features:
            raise ValueError(
                "data.num_relevant_features must be between 1 and num_features"
            )
        if self.num_classes < 2:
            raise ValueError("data.num_classes must be at least 2")
        if not math.isfinite(self.class_separation) or self.class_separation <= 0:
            raise ValueError("data.class_separation must be positive")
        if not 0.0 <= self.label_noise < 1.0:
            raise ValueError("data.label_noise must be in [0, 1)")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("data.train_fraction must be in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must sum to less than 1")


@dataclass(frozen=True)
class ModelConfig:
    aggregated_dim: int = 4
    hidden_dim: int = 64
    classifier: str = "mlp"
    noise_std: float = 0.05
    gate_temperature: float = 0.1
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    ft_token_dim: int = 16
    ft_num_heads: int = 2
    ft_num_layers: int = 1
    ft_feedforward_dim: int = 32
    ft_dropout: float = 0.1

    def __post_init__(self) -> None:
        _validate_declared_types(self, "model")
        if self.aggregated_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.classifier not in {"logistic", "linear", "mlp", "ft-transformer"}:
            raise ValueError(
                "model.classifier must be logistic, linear, mlp, or ft-transformer"
            )
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("model.noise_std must be non-negative")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("model.gate_temperature must be positive")
        if not self.gate_gamma < 0 < self.gate_zeta:
            raise ValueError("hard-concrete bounds must satisfy gamma < 0 < zeta")
        if min(self.ft_token_dim, self.ft_num_heads, self.ft_num_layers) <= 0:
            raise ValueError(
                "FT-Transformer token, head, and layer counts must be positive"
            )
        if self.ft_feedforward_dim <= 0:
            raise ValueError("model.ft_feedforward_dim must be positive")
        if self.ft_token_dim % self.ft_num_heads != 0:
            raise ValueError("model.ft_token_dim must be divisible by ft_num_heads")
        if not 0.0 <= self.ft_dropout < 1.0:
            raise ValueError("model.ft_dropout must be in [0, 1)")


@dataclass(frozen=True)
class ObjectiveConfig:
    trace_weight: float = 1.0
    sparsity_weight: float = 0.2
    binary_weight: float = 0.05
    sparsity_weighting: str = "entropy"
    trace_normalization: str = "normalized"
    trace_epsilon: float = 1e-8
    detach_trace_gates: bool = False
    detach_trace_denominator: bool = False
    gate_warmup_iterations: int = 0
    gate_ramp_iterations: int = 0
    selection_gate_samples: int = 16
    selection_gate_estimator: str = "fixed_monte_carlo"

    def __post_init__(self) -> None:
        _validate_declared_types(self, "objective")
        if not all(
            math.isfinite(value)
            for value in (
                self.trace_weight,
                self.sparsity_weight,
                self.binary_weight,
                self.trace_epsilon,
            )
        ):
            raise ValueError("objective weights and epsilon must be finite")
        if min(self.trace_weight, self.sparsity_weight, self.binary_weight) < 0:
            raise ValueError("objective weights must be non-negative")
        if self.sparsity_weighting not in {"uniform", "entropy", "variance"}:
            raise ValueError(
                "objective.sparsity_weighting must be uniform, entropy, or variance"
            )
        if self.trace_normalization not in {"raw", "normalized"}:
            raise ValueError(
                "objective.trace_normalization must be 'raw' or 'normalized'"
            )
        if self.trace_epsilon <= 0:
            raise ValueError("objective.trace_epsilon must be positive")
        if self.gate_warmup_iterations < 0 or self.gate_ramp_iterations < 0:
            raise ValueError("gate warmup and ramp iterations must be non-negative")
        if self.selection_gate_samples <= 0:
            raise ValueError("objective.selection_gate_samples must be positive")
        if self.selection_gate_estimator not in {
            "fixed_monte_carlo",
            "deterministic",
        }:
            raise ValueError(
                "objective.selection_gate_estimator must be fixed_monte_carlo "
                "or deterministic"
            )


@dataclass(frozen=True)
class AGDAConfig:
    multiplier_grid: tuple[float, ...] = DEFAULT_MULTIPLIER_GRID
    initial_grid_index: int = 12
    target_accuracy: float = 0.8
    tolerance: float = 0.01
    smoothing: float = 0.8
    use_smoothed_accuracy: bool = False
    outer_iterations: int = 40
    primal_steps_per_adjustment: int = 0
    warmup_iterations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.multiplier_grid, (tuple, list)) or any(
            type(value) not in {int, float} for value in self.multiplier_grid
        ):
            raise ValueError("agda.multiplier_grid must contain only numbers")
        grid = tuple(float(value) for value in self.multiplier_grid)
        object.__setattr__(self, "multiplier_grid", grid)
        _validate_declared_types(self, "agda")
        validate_multiplier_grid(grid, self.initial_grid_index)
        if not 0.0 <= self.target_accuracy <= 1.0:
            raise ValueError("agda.target_accuracy must be in [0, 1]")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("agda.tolerance must be non-negative")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("agda.smoothing must be in [0, 1)")
        if self.outer_iterations <= 0:
            raise ValueError("agda.outer_iterations must be positive")
        if self.primal_steps_per_adjustment < 0:
            raise ValueError("agda.primal_steps_per_adjustment must be non-negative")
        if self.warmup_iterations < 0:
            raise ValueError("agda.warmup_iterations must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_declared_types(self, "training")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.optimizer not in {"adam", "adamw", "sgd", "rmsprop"}:
            raise ValueError("training.optimizer must be adam, adamw, sgd, or rmsprop")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.device != "auto" and not re.fullmatch(
            r"cpu|cuda(?::\d+)?", self.device
        ):
            raise ValueError("training.device must be 'auto', 'cpu', or a CUDA device")


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    agda: AGDAConfig = field(default_factory=AGDAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        _validate_declared_types(self, "base")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    output_dir: str
    save_checkpoints: bool
    base: RunConfig
    sweep: Mapping[str, tuple[Any, ...]]
    source: Path


def apply_accuracy_margin(config: RunConfig, margin: float = 0.0) -> RunConfig:
    """Return a run configuration with a validated CLI accuracy margin."""

    if type(margin) not in {int, float} or not math.isfinite(float(margin)):
        raise ValueError("margin must be a finite number")
    resolved_margin = float(margin)
    if resolved_margin < 0:
        raise ValueError("margin must be non-negative")
    effective_target = config.agda.target_accuracy + resolved_margin
    if effective_target > 1.0:
        raise ValueError(
            "target_accuracy + margin must be at most 1.0 "
            f"({config.agda.target_accuracy:g} + {resolved_margin:g})"
        )
    if resolved_margin == 0.0:
        return config
    return replace(
        config,
        agda=replace(config.agda, target_accuracy=effective_target),
    )


T = TypeVar("T")


def validate_multiplier_grid(
    values: tuple[float, ...] | list[float], initial_index: int
) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if len(grid) < 2:
        raise ValueError("multiplier grid must contain at least two values")
    if any(not math.isfinite(value) or value < 0 for value in grid):
        raise ValueError("multiplier grid values must be finite and non-negative")
    if any(right <= left for left, right in itertools.pairwise(grid)):
        raise ValueError("multiplier grid values must be strictly increasing")
    if not 0 <= initial_index < len(grid):
        raise ValueError("agda.initial_grid_index is outside the multiplier grid")
    if grid[-1] <= 0:
        raise ValueError("multiplier grid must include a positive value")
    return grid


def _strict_dataclass(cls: type[T], raw: Mapping[str, Any], context: str) -> T:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {', '.join(unknown)}")
    values = dict(raw)
    if cls is AGDAConfig and "multiplier_grid" in values:
        values["multiplier_grid"] = tuple(values["multiplier_grid"])
    return cls(**values)


def run_config_from_mapping(raw: Mapping[str, Any]) -> RunConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("campaign.base must be a mapping")
    allowed = {"data", "model", "objective", "agda", "training"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown base keys: {', '.join(unknown)}")
    return RunConfig(
        data=_strict_dataclass(DataConfig, raw.get("data", {}), "data"),
        model=_strict_dataclass(ModelConfig, raw.get("model", {}), "model"),
        objective=_strict_dataclass(
            ObjectiveConfig, raw.get("objective", {}), "objective"
        ),
        agda=_strict_dataclass(AGDAConfig, raw.get("agda", {}), "agda"),
        training=_strict_dataclass(TrainingConfig, raw.get("training", {}), "training"),
    )


def load_campaign(path: str | Path) -> CampaignConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("campaign file must contain a mapping")
    allowed = {"name", "output_dir", "save_checkpoints", "base", "sweep"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown campaign keys: {', '.join(unknown)}")
    if "base" not in raw:
        raise ValueError("campaign.base is required")
    name = str(raw.get("name", source.stem))
    if not name:
        raise ValueError("campaign.name must not be empty")
    output_dir = str(raw.get("output_dir", f"results/{name}"))
    if not output_dir:
        raise ValueError("campaign.output_dir must not be empty")
    save_checkpoints = raw.get("save_checkpoints", True)
    if not isinstance(save_checkpoints, bool):
        raise ValueError("campaign.save_checkpoints must be a boolean")
    sweep_raw = raw.get("sweep", {})
    if not isinstance(sweep_raw, Mapping):
        raise ValueError("campaign.sweep must be a mapping")
    sweep: dict[str, tuple[Any, ...]] = {}
    for key, values in sweep_raw.items():
        if not isinstance(key, str) or "." not in key:
            raise ValueError("sweep keys must be dotted paths such as model.noise_std")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep value for {key!r} must be a non-empty list")
        sweep[key] = tuple(values)
    if not isinstance(raw["base"], Mapping):
        raise ValueError("campaign.base must be a mapping")
    base_raw = copy.deepcopy(dict(raw["base"]))
    base = run_config_from_mapping(base_raw)
    # Validate every override eagerly rather than failing after a long campaign.
    for _ in expand_run_mappings(base.to_dict(), sweep):
        pass
    return CampaignConfig(
        name=name,
        output_dir=output_dir,
        save_checkpoints=save_checkpoints,
        base=base,
        sweep=sweep,
        source=source,
    )


def _set_dotted(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) != 2:
        raise ValueError("sweep keys must have exactly one section and one field")
    section, field_name = parts
    if section not in mapping or not isinstance(mapping[section], Mapping):
        raise ValueError(f"sweep section {section!r} is absent from campaign.base")
    if field_name not in mapping[section]:
        # A defaulted field may be omitted from YAML, so check the schema too.
        schemas = {
            "data": DataConfig,
            "model": ModelConfig,
            "objective": ObjectiveConfig,
            "agda": AGDAConfig,
            "training": TrainingConfig,
        }
        schema = schemas.get(section)
        if schema is None or field_name not in {item.name for item in fields(schema)}:
            raise ValueError(f"unknown sweep key {dotted_key!r}")
    section_values = dict(mapping[section])
    section_values[field_name] = value
    mapping[section] = section_values


def expand_run_mappings(
    base: Mapping[str, Any], sweep: Mapping[str, tuple[Any, ...]]
) -> list[tuple[dict[str, Any], dict[str, Any], RunConfig]]:
    """Return ``(resolved mapping, overrides, validated config)`` tuples."""

    keys = list(sweep)
    products = itertools.product(*(sweep[key] for key in keys)) if keys else [()]
    expanded = []
    for values in products:
        resolved = copy.deepcopy(dict(base))
        overrides = dict(zip(keys, values, strict=True))
        for key, value in overrides.items():
            _set_dotted(resolved, key, value)
        expanded.append((resolved, overrides, run_config_from_mapping(resolved)))
    return expanded


def expand_campaign(
    campaign: CampaignConfig,
) -> list[tuple[dict[str, Any], RunConfig]]:
    base = campaign.base.to_dict()
    return [
        (overrides, config)
        for _, overrides, config in expand_run_mappings(base, campaign.sweep)
    ]
