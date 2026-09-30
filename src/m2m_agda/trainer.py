"""One canonical implementation of the M2M-AGDA training protocol."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import RunConfig
from .controller import AccuracyGridController
from .data import DatasetSplits, Split
from .model import M2MAGDAModel, ModelSpec
from .objectives import (
    compute_training_covariance,
    empirical_feature_entropies,
    empirical_feature_variances,
    gate_regularization_scale,
    privacy_terms,
    privacy_terms_to_floats,
)

ProgressCallback = Callable[[dict[str, Any]], None]
ModelFactory = Callable[[RunConfig, int, int], M2MAGDAModel]


@dataclass
class FitResult:
    model: M2MAGDAModel
    history: list[dict[str, Any]]
    feasible: bool
    selected_source: str
    selected_iteration: int
    best_feasible_checkpoint: dict[str, Any] | None
    initial_privacy_terms: dict[str, Any]
    final_validation: dict[str, float]
    final_privacy_terms: dict[str, Any]
    primal_steps_per_adjustment: int
    global_steps: int
    training_runtime_seconds: float


@dataclass
class TrainingResult:
    model: M2MAGDAModel
    metrics: dict[str, Any]


class _BatchCycler:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested device {requested!r}, but CUDA is unavailable")
    return device


def build_model(config: RunConfig, input_dim: int, num_classes: int) -> M2MAGDAModel:
    """Build the standard AGDA model.

    The explicit factory boundary lets experiment-only component ablations
    replace a gate or projection without forking the training protocol.
    """

    if config.model.aggregated_dim > input_dim:
        raise ValueError(
            "model.aggregated_dim cannot exceed the encoded input dimension"
        )
    spec = ModelSpec(
        input_dim=input_dim,
        aggregated_dim=config.model.aggregated_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=num_classes,
        classifier=config.model.classifier,
        noise_std=config.model.noise_std,
        gate_temperature=config.model.gate_temperature,
        gate_gamma=config.model.gate_gamma,
        gate_zeta=config.model.gate_zeta,
        ft_token_dim=config.model.ft_token_dim,
        ft_num_heads=config.model.ft_num_heads,
        ft_num_layers=config.model.ft_num_layers,
        ft_feedforward_dim=config.model.ft_feedforward_dim,
        ft_dropout=config.model.ft_dropout,
    )
    return M2MAGDAModel(spec)


def _build_optimizer(model: torch.nn.Module, config: RunConfig):
    options = {
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
    if config.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **options)
    if config.training.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **options)
    if config.training.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **options)
    if config.training.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), momentum=0.9, **options)
    raise AssertionError("TrainingConfig rejected the unknown optimizer")


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _orthogonality_error(model: M2MAGDAModel) -> float:
    with torch.no_grad():
        weights = model.minimizer.aggregation.weight
        identity = torch.eye(
            weights.shape[1], dtype=weights.dtype, device=weights.device
        )
        return float(torch.linalg.matrix_norm(weights.T @ weights - identity).item())


def evaluate_model(
    model: M2MAGDAModel,
    split: Split,
    *,
    device: torch.device,
    batch_size: int,
    add_noise: bool,
    noise_seed: int,
) -> dict[str, float]:
    model.eval()
    generator = None
    if add_noise:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(split.x), batch_size):
            inputs = torch.from_numpy(split.x[start : start + batch_size]).to(device)
            labels = torch.from_numpy(split.y[start : start + batch_size]).to(device)
            logits, _, _ = model(
                inputs,
                sample_gates=False,
                add_noise=add_noise,
                generator=generator,
            )
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            seen += len(labels)
    return {"loss": total_loss / seen, "accuracy": correct / seen}


def _privacy_snapshot(
    model: M2MAGDAModel,
    covariance: torch.Tensor,
    feature_weights: torch.Tensor | None,
    config: RunConfig,
    *,
    gate_scale: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        if config.objective.selection_gate_estimator == "deterministic":
            gate_samples = [model.minimizer.feature_removal.deterministic_gates()]
        else:
            generator = torch.Generator(device=covariance.device).manual_seed(
                config.training.seed + 10_000_000
            )
            gate_samples = [
                model.minimizer.feature_removal.sample_gates(generator)
                for _ in range(config.objective.selection_gate_samples)
            ]
        accumulated: dict[str, float] = {}
        for sampled_gates in gate_samples:
            terms = privacy_terms(
                model,
                covariance,
                sampled_gates,
                feature_weights,
                trace_weight=config.objective.trace_weight,
                sparsity_weight=config.objective.sparsity_weight,
                binary_weight=config.objective.binary_weight,
                trace_normalization=config.objective.trace_normalization,
                trace_epsilon=config.objective.trace_epsilon,
                detach_trace_gates=config.objective.detach_trace_gates,
                detach_trace_denominator=config.objective.detach_trace_denominator,
                gate_scale=gate_scale,
            )
            for name, value in privacy_terms_to_floats(terms).items():
                accumulated[name] = accumulated.get(name, 0.0) + value
        snapshot: dict[str, Any] = {
            name: value / len(gate_samples) for name, value in accumulated.items()
        }
        gates = model.minimizer.feature_removal.deterministic_gates()
        probabilities = model.minimizer.feature_removal.expected_open_probabilities()
        snapshot.update(
            {
                "selection_gate_estimator": config.objective.selection_gate_estimator,
                "selection_gate_samples": len(gate_samples),
                "expected_open_count": float(probabilities.sum().item()),
                "expected_open_probability": float(probabilities.mean().item()),
                "deterministic_nonzero_gates": int((gates > 1e-6).sum().item()),
                "mean_deterministic_gate": float(gates.mean().item()),
                "orthogonality_error": _orthogonality_error(model),
            }
        )
        return snapshot


class AGDATrainer:
    """Fit on train/validation only; test data are deliberately not accepted."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: torch.device | None = None,
        progress: ProgressCallback | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config
        self.device = device or resolve_device(config.training.device)
        self.progress = progress
        self.model_factory = model_factory or build_model

    def fit(self, train: Split, validation: Split, *, num_classes: int) -> FitResult:
        started = time.perf_counter()
        set_seed(self.config.training.seed)
        if train.x.shape[1] != validation.x.shape[1]:
            raise ValueError("train and validation dimensions differ")
        model = self.model_factory(self.config, train.x.shape[1], num_classes).to(
            self.device
        )
        optimizer = _build_optimizer(model, self.config)

        train_inputs = torch.from_numpy(train.x)
        train_labels = torch.from_numpy(train.y)
        covariance = compute_training_covariance(train_inputs.to(self.device))
        if self.config.objective.sparsity_weighting == "entropy":
            feature_weights = empirical_feature_entropies(train_inputs.to(self.device))
        elif self.config.objective.sparsity_weighting == "variance":
            feature_weights = empirical_feature_variances(train_inputs.to(self.device))
        else:
            feature_weights = None

        loader = DataLoader(
            TensorDataset(train_inputs, train_labels),
            batch_size=self.config.training.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.training.seed),
        )
        steps_per_adjustment = self.config.agda.primal_steps_per_adjustment or len(
            loader
        )
        if steps_per_adjustment <= 0:
            raise ValueError("the training split produced no mini-batches")
        batches = _BatchCycler(loader)
        controller = AccuracyGridController(
            self.config.agda.multiplier_grid,
            self.config.agda.initial_grid_index,
            target_accuracy=self.config.agda.target_accuracy,
            tolerance=self.config.agda.tolerance,
            smoothing=self.config.agda.smoothing,
            use_smoothed_accuracy=self.config.agda.use_smoothed_accuracy,
        )

        initial_scale = gate_regularization_scale(
            0,
            self.config.objective.gate_warmup_iterations,
            self.config.objective.gate_ramp_iterations,
        )
        initial_privacy = _privacy_snapshot(
            model,
            covariance,
            feature_weights,
            self.config,
            gate_scale=initial_scale,
        )
        initial_privacy["iteration"] = 0

        history: list[dict[str, Any]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_checkpoint: dict[str, Any] | None = None
        global_steps = 0

        for iteration in range(1, self.config.agda.outer_iterations + 1):
            gate_scale = gate_regularization_scale(
                iteration,
                self.config.objective.gate_warmup_iterations,
                self.config.objective.gate_ramp_iterations,
            )
            training_multiplier = controller.multiplier
            training_grid_index = controller.state.grid_index
            model.train()
            task_total = 0.0
            optimization_privacy_total = 0.0
            primal_total = 0.0
            rows_seen = 0

            # This loop is the paper's inner loop: lambda remains fixed for
            # exactly q mini-batch updates, even when q crosses data epochs.
            for _ in range(steps_per_adjustment):
                inputs, labels = batches.next()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits, _, sampled_gates = model(
                    inputs,
                    sample_gates=True,
                    add_noise=self.config.model.noise_std > 0,
                )
                task_loss = F.cross_entropy(logits, labels)
                terms = privacy_terms(
                    model,
                    covariance,
                    sampled_gates,
                    feature_weights,
                    trace_weight=self.config.objective.trace_weight,
                    sparsity_weight=self.config.objective.sparsity_weight,
                    binary_weight=self.config.objective.binary_weight,
                    trace_normalization=self.config.objective.trace_normalization,
                    trace_epsilon=self.config.objective.trace_epsilon,
                    detach_trace_gates=self.config.objective.detach_trace_gates,
                    detach_trace_denominator=(
                        self.config.objective.detach_trace_denominator
                    ),
                    gate_scale=gate_scale,
                )
                primal_loss = (
                    terms.optimization_objective + training_multiplier * task_loss
                )
                primal_loss.backward()
                optimizer.step()

                batch_rows = len(labels)
                rows_seen += batch_rows
                global_steps += 1
                task_total += float(task_loss.detach().item()) * batch_rows
                optimization_privacy_total += (
                    float(terms.optimization_objective.detach().item()) * batch_rows
                )
                primal_total += float(primal_loss.detach().item()) * batch_rows

            validation_metrics = evaluate_model(
                model,
                validation,
                device=self.device,
                batch_size=self.config.training.batch_size,
                add_noise=self.config.model.noise_std > 0,
                noise_seed=self.config.training.seed + iteration,
            )
            snapshot = _privacy_snapshot(
                model,
                covariance,
                feature_weights,
                self.config,
                gate_scale=gate_scale,
            )
            event = controller.observe(
                validation_metrics["accuracy"],
                allow_update=iteration > self.config.agda.warmup_iterations,
            )
            # Smoothing may stabilize grid motion, but Figure 2 defines
            # feasibility with the raw validation accuracy a_t.
            feasibility_accuracy = event.raw_accuracy
            feasible = feasibility_accuracy >= self.config.agda.target_accuracy
            became_best = False
            if feasible and (
                best_checkpoint is None
                or snapshot["selection_privacy_objective"]
                < best_checkpoint["selection_privacy_objective"]
            ):
                became_best = True
                best_state = _clone_state_dict(model)
                best_checkpoint = {
                    "iteration": iteration,
                    "global_step": global_steps,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "feasibility_accuracy": feasibility_accuracy,
                    "selection_privacy_objective": snapshot[
                        "selection_privacy_objective"
                    ],
                    "optimization_privacy_objective": snapshot[
                        "optimization_privacy_objective"
                    ],
                    "training_multiplier": training_multiplier,
                    "training_grid_index": training_grid_index,
                    "next_multiplier": event.new_multiplier,
                    "next_grid_index": event.new_grid_index,
                    "gate_regularization_scale": gate_scale,
                    "selection_gate_estimator": snapshot["selection_gate_estimator"],
                    "selection_gate_samples": snapshot["selection_gate_samples"],
                }

            row: dict[str, Any] = {
                "iteration": iteration,
                "global_step": global_steps,
                "primal_steps": steps_per_adjustment,
                "training_multiplier": training_multiplier,
                "training_grid_index": training_grid_index,
                "next_multiplier": event.new_multiplier,
                "next_grid_index": event.new_grid_index,
                "controller_action": event.action,
                "controller_hit_grid_boundary": event.hit_grid_boundary,
                "raw_validation_accuracy": event.raw_accuracy,
                "smoothed_validation_accuracy": event.smoothed_accuracy,
                "controller_accuracy": event.decision_accuracy,
                "feasibility_accuracy": feasibility_accuracy,
                "validation_feasible": feasible,
                "became_best_feasible_checkpoint": became_best,
                "train_task_loss": task_total / rows_seen,
                "train_optimization_privacy_objective": (
                    optimization_privacy_total / rows_seen
                ),
                "train_primal_objective": primal_total / rows_seen,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                **snapshot,
            }
            history.append(row)
            if self.progress is not None:
                self.progress(row)

        if best_state is not None:
            model.load_state_dict(
                {name: tensor.to(self.device) for name, tensor in best_state.items()}
            )
            selected_source = "best_feasible"
            assert best_checkpoint is not None
            selected_iteration = int(best_checkpoint["iteration"])
        else:
            selected_source = "final_infeasible"
            selected_iteration = int(history[-1]["iteration"])

        selected_gate_scale = gate_regularization_scale(
            selected_iteration,
            self.config.objective.gate_warmup_iterations,
            self.config.objective.gate_ramp_iterations,
        )
        final_validation = evaluate_model(
            model,
            validation,
            device=self.device,
            batch_size=self.config.training.batch_size,
            add_noise=self.config.model.noise_std > 0,
            noise_seed=self.config.training.seed + 100_000,
        )
        final_privacy = _privacy_snapshot(
            model,
            covariance,
            feature_weights,
            self.config,
            gate_scale=selected_gate_scale,
        )
        return FitResult(
            model=model,
            history=history,
            feasible=best_state is not None,
            selected_source=selected_source,
            selected_iteration=selected_iteration,
            best_feasible_checkpoint=best_checkpoint,
            initial_privacy_terms=initial_privacy,
            final_validation=final_validation,
            final_privacy_terms=final_privacy,
            primal_steps_per_adjustment=steps_per_adjustment,
            global_steps=global_steps,
            training_runtime_seconds=time.perf_counter() - started,
        )


def train_and_evaluate(
    dataset: DatasetSplits,
    config: RunConfig,
    *,
    device: torch.device | None = None,
    progress: ProgressCallback | None = None,
    model_factory: ModelFactory | None = None,
) -> TrainingResult:
    """Fit without test access, select a checkpoint, then evaluate test once."""

    trainer = AGDATrainer(
        config,
        device=device,
        progress=progress,
        model_factory=model_factory,
    )
    fit = trainer.fit(
        dataset.train, dataset.validation, num_classes=dataset.num_classes
    )
    final_test = evaluate_model(
        fit.model,
        dataset.test,
        device=trainer.device,
        batch_size=config.training.batch_size,
        add_noise=config.model.noise_std > 0,
        noise_seed=config.training.seed + 200_000,
    )
    metrics: dict[str, Any] = {
        "settings": config.to_dict(),
        "dataset": {
            "name": dataset.name,
            "input_dim": dataset.input_dim,
            "num_classes": dataset.num_classes,
            "rows": {
                "train": len(dataset.train.x),
                "validation": len(dataset.validation.x),
                "test": len(dataset.test.x),
            },
        },
        "feasible": fit.feasible,
        "selected_checkpoint_source": fit.selected_source,
        "selected_iteration": fit.selected_iteration,
        "best_feasible_checkpoint": fit.best_feasible_checkpoint,
        "resolved_primal_steps_per_adjustment": (fit.primal_steps_per_adjustment),
        "global_steps": fit.global_steps,
        "initial_privacy_terms": fit.initial_privacy_terms,
        "final_validation": fit.final_validation,
        "final_test": final_test,
        "final_privacy_terms": fit.final_privacy_terms,
        "history": fit.history,
        "training_runtime_seconds": fit.training_runtime_seconds,
    }
    return TrainingResult(model=fit.model, metrics=metrics)


def save_checkpoint(
    path: str | Path,
    result: TrainingResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_spec": asdict(result.model.spec),
        "state_dict": _clone_state_dict(result.model),
        "selection": {
            "feasible": result.metrics["feasible"],
            "selected_checkpoint_source": result.metrics["selected_checkpoint_source"],
            "selected_iteration": result.metrics["selected_iteration"],
            "best_feasible_checkpoint": result.metrics["best_feasible_checkpoint"],
        },
        "metadata": metadata or {},
    }
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[M2MAGDAModel, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = M2MAGDAModel(ModelSpec(**payload["model_spec"]))
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    metadata = {
        "selection": payload.get("selection", {}),
        "metadata": payload.get("metadata", {}),
    }
    return model, metadata


def transform_numpy(
    model: M2MAGDAModel,
    inputs: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    add_noise: bool = False,
    noise_seed: int | None = None,
) -> np.ndarray:
    target = torch.device(device)
    model = model.to(target)
    tensor = torch.as_tensor(inputs, dtype=torch.float32, device=target)
    with torch.no_grad():
        released = model.transform(
            tensor,
            deterministic_gates=True,
            add_noise=add_noise,
            noise_seed=noise_seed,
        )
    return released.cpu().numpy()
