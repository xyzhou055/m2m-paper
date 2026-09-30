"""Matched M2M component ablation with reconstruction NRR."""


from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from m2m_agda.classifiers import ClassifierSpec, build_classifier  # noqa: E402
from m2m_agda.config import (  # noqa: E402
    CampaignConfig,
    RunConfig,
    apply_accuracy_margin,
    expand_campaign,
    load_campaign,
)
from m2m_agda.data import (  # noqa: E402
    DatasetSplits,
    dataset_fingerprint,
    load_dataset,
)
from m2m_agda.model import M2MAGDAModel  # noqa: E402
from m2m_agda.trainer import (  # noqa: E402
    AGDATrainer,
    build_model,
    resolve_device,
    set_seed,
    transform_numpy,
)
from reconstruct import (  # noqa: E402
    AttackerConfig,
    discretize_reconstruction_targets,
    evaluate_reconstruction_attack,
    standardize_release,
)

DEFAULT_CONFIG = ROOT / "experiments/configs/model_ablation.yaml"
VARIANTS = ("m2m", "m2m-sel", "m2m-agg", "m2m-no-trace", "m2m-pca")
VARIANT_LABELS = {
    "m2m": "M2M",
    "m2m-sel": "M2M_sel",
    "m2m-agg": "M2M_agg",
    "m2m-no-trace": "M2M_no_trace",
    "m2m-pca": "M2M_PCA",
}
VARIANT_DESCRIPTIONS = {
    "m2m": "learned hard-concrete gates and trace-minimizing aggregation",
    "m2m-sel": "learned gates with a fixed identity release and no trace term",
    "m2m-agg": "fixed-open gates with learned trace-minimizing aggregation",
    "m2m-no-trace": "full architecture with the trace term removed",
    "m2m-pca": "selection-only training followed by train-fitted PCA",
}


@dataclass(frozen=True)
class ReconstructionData:
    targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    category_counts: list[int]
    metadata: dict[str, Any]


def load_reconstruction_data(
    config: RunConfig,
    dataset: DatasetSplits,
    *,
    config_directory: Path,
) -> ReconstructionData | None:
    """Load logical reconstruction targets emitted by data_preparation.py."""

    if config.data.kind != "npz" or config.data.path is None:
        return None
    model_path = Path(config.data.path).expanduser()
    if not model_path.is_absolute():
        model_path = config_directory / model_path
    model_path = model_path.resolve()
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_fingerprint = metadata.get("dataset_fingerprint")
    actual_fingerprint = dataset_fingerprint(dataset)
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(
            f"reconstruction metadata fingerprint mismatch for {model_path}"
        )
    reconstruction_name = metadata.get("reconstruction_archive")
    if not isinstance(reconstruction_name, str) or not reconstruction_name:
        raise ValueError("metadata is missing reconstruction_archive")
    reconstruction_path = metadata_path.parent / reconstruction_name
    required = {"x_train_raw", "x_val_raw", "x_test_raw", "category_counts"}
    with np.load(reconstruction_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("reconstruction archive has an invalid schema")
        targets = tuple(
            np.ascontiguousarray(archive[name], dtype=np.int64)
            for name in ("x_train_raw", "x_val_raw", "x_test_raw")
        )
        category_counts = archive["category_counts"].astype(np.int64).tolist()
    expected_rows = (
        len(dataset.train.x),
        len(dataset.validation.x),
        len(dataset.test.x),
    )
    if any(
        target.ndim != 2 or len(target) != rows
        for target, rows in zip(targets, expected_rows, strict=True)
    ):
        raise ValueError("reconstruction targets do not align with dataset splits")
    if not category_counts or targets[0].shape[1] != len(category_counts):
        raise ValueError("category counts do not align with reconstruction targets")
    for target in targets:
        for column, count in enumerate(category_counts):
            if count < 2 or target[:, column].min() < 0:
                raise ValueError("reconstruction categories must be non-negative")
            if target[:, column].max() >= count:
                raise ValueError("reconstruction category exceeds its declared count")
    return ReconstructionData(
        targets=targets,  # type: ignore[arg-type]
        category_counts=category_counts,
        metadata={
            "metadata_path": str(metadata_path),
            "reconstruction_path": str(reconstruction_path),
            "feature_names": metadata.get("feature_names"),
            "feature_types": metadata.get("feature_types"),
        },
    )


class FixedOpenFeatureRemoval(nn.Module):
    """Feature-removal interface with every gate fixed to one."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.register_buffer("gates", torch.ones(input_dim))

    def expected_open_probabilities(self) -> torch.Tensor:
        return self.gates

    def deterministic_gates(self) -> torch.Tensor:
        return self.gates

    def sample_gates(self, generator: torch.Generator | None = None) -> torch.Tensor:
        del generator
        return self.gates

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sample: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sample, generator
        return inputs, self.gates.to(dtype=inputs.dtype, device=inputs.device)


class FixedIdentityAggregation(nn.Module):
    """Aggregation interface for a raw selected-feature release."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.noise_std = 0.0
        self.register_buffer("_weight", torch.eye(input_dim))

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        add_noise: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del add_noise, generator
        return inputs


def variant_config(config: RunConfig, variant: str, input_dim: int) -> RunConfig:
    """Return a matched immutable configuration for one ablation variant."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation variant {variant!r}")
    model = config.model
    objective = config.objective
    if variant in {"m2m-sel", "m2m-pca"}:
        model = replace(model, aggregated_dim=input_dim, noise_std=0.0)
        objective = replace(objective, trace_weight=0.0)
    elif variant == "m2m-agg":
        objective = replace(objective, sparsity_weight=0.0, binary_weight=0.0)
    elif variant == "m2m-no-trace":
        objective = replace(objective, trace_weight=0.0)
    return replace(config, model=model, objective=objective)


def model_factory(variant: str):
    """Build a model whose changed component is explicit and frozen."""

    def factory(config: RunConfig, input_dim: int, num_classes: int) -> M2MAGDAModel:
        model = build_model(config, input_dim, num_classes)
        if variant == "m2m-agg":
            model.minimizer.feature_removal = FixedOpenFeatureRemoval(input_dim)
        elif variant in {"m2m-sel", "m2m-pca"}:
            model.minimizer.aggregation = FixedIdentityAggregation(input_dim)
        return model

    return factory


def fit_pca(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], components: int
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Fit PCA with NumPy on training rows and transform every split."""

    train = np.asarray(arrays[0], dtype=np.float64)
    components = min(components, train.shape[0], train.shape[1])
    if components <= 0:
        raise ValueError("PCA needs a positive component count")
    mean = train.mean(axis=0)
    centered = train - mean
    if not np.any(np.std(centered, axis=0) > 1e-8):
        transformed = tuple(
            np.zeros((len(array), 1), dtype=np.float32) for array in arrays
        )
        return transformed, {
            "components": 1,
            "constant_input_fallback": True,
            "fit_split": "training",
        }
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:components].T
    transformed = tuple(
        np.asarray((array - mean) @ basis, dtype=np.float32) for array in arrays
    )
    total = float(np.square(singular_values).sum())
    explained = float(np.square(singular_values[:components]).sum() / max(total, 1e-12))
    return transformed, {
        "components": components,
        "constant_input_fallback": False,
        "fit_split": "training",
        "explained_variance_ratio_sum": explained,
    }


def _classifier_spec(
    config: RunConfig,
    *,
    architecture: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
) -> ClassifierSpec:
    return ClassifierSpec(
        architecture=architecture,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        ft_token_dim=config.model.ft_token_dim,
        ft_num_heads=config.model.ft_num_heads,
        ft_num_layers=config.model.ft_num_layers,
        ft_feedforward_dim=config.model.ft_feedforward_dim,
        ft_dropout=config.model.ft_dropout,
    )


def train_utility_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: RunConfig,
    *,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    set_seed(config.training.seed)
    model = build_classifier(
        _classifier_spec(
            config,
            architecture=config.model.classifier,
            input_dim=train_x.shape[1],
            num_classes=int(train_y.max()) + 1,
            hidden_dim=config.model.hidden_dim,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    model.train()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(inputs), labels).backward()
            optimizer.step()
    return model.eval()


def classifier_accuracy(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    correct = 0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            inputs = torch.from_numpy(features[start : start + batch_size]).to(device)
            target = torch.from_numpy(labels[start : start + batch_size]).to(device)
            correct += int(model(inputs).argmax(1).eq(target).sum().item())
    return correct / len(labels)


def _materialize_releases(
    model: M2MAGDAModel,
    dataset: DatasetSplits,
    config: RunConfig,
    variant: str,
    device: torch.device,
    selected_iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    add_noise = (
        variant in {"m2m", "m2m-agg", "m2m-no-trace"} and config.model.noise_std > 0
    )
    noise_seeds = (
        config.training.seed + 100_000,
        config.training.seed + selected_iteration,
        config.training.seed + 200_000,
    )
    return tuple(
        transform_numpy(
            model,
            split.x,
            device=device,
            add_noise=add_noise,
            noise_seed=noise_seeds[index],
        )
        for index, split in enumerate((dataset.train, dataset.validation, dataset.test))
    )  # type: ignore[return-value]


def run_variant(
    dataset: DatasetSplits,
    config: RunConfig,
    variant: str,
    *,
    device: torch.device,
    reconstruction_bins: int,
    attacker_architecture: str,
    attacker_hidden_dim: int,
    attacker_second_hidden_dim: int,
    attacker_epochs: int,
    attacker_learning_rate: float,
    attacker_batch_size: int,
    pca_utility_epochs: int,
    reconstruction_data: ReconstructionData | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = variant_config(config, variant, dataset.input_dim)
    trainer = AGDATrainer(
        resolved,
        device=device,
        model_factory=model_factory(variant),
    )
    fit = trainer.fit(
        dataset.train, dataset.validation, num_classes=dataset.num_classes
    )
    row: dict[str, Any] = {
        "variant": VARIANT_LABELS[variant],
        "variant_key": variant,
        "description": VARIANT_DESCRIPTIONS[variant],
        "status": "infeasible",
        "selected_checkpoint_source": fit.selected_source,
        "selected_iteration": fit.selected_iteration,
        "best_feasible_checkpoint": fit.best_feasible_checkpoint,
        "final_privacy_terms": fit.final_privacy_terms,
        "training_runtime_seconds": fit.training_runtime_seconds,
        "pca": None,
        "reconstruction": None,
    }
    if not fit.feasible:
        row["runtime_seconds"] = time.perf_counter() - started
        return row

    releases = _materialize_releases(
        fit.model,
        dataset,
        resolved,
        variant,
        device,
        fit.selected_iteration,
    )
    if variant == "m2m-pca":
        releases, row["pca"] = fit_pca(releases, config.model.aggregated_dim)
        standardized = standardize_release(releases)
        utility_model = train_utility_classifier(
            standardized[0],
            dataset.train.y,
            config,
            device=device,
            epochs=pca_utility_epochs,
        )
        utility_arrays = standardized
        classifier_source = "fresh classifier on train-fitted PCA release"
    else:
        utility_model = fit.model.classifier
        utility_arrays = releases
        classifier_source = "selected AGDA classifier"
    validation_accuracy = classifier_accuracy(
        utility_model,
        utility_arrays[1],
        dataset.validation.y,
        device=device,
        batch_size=config.training.batch_size,
    )
    test_accuracy = classifier_accuracy(
        utility_model,
        utility_arrays[2],
        dataset.test.y,
        device=device,
        batch_size=config.training.batch_size,
    )
    release_feasible = validation_accuracy >= config.agda.target_accuracy
    row["materialized_release_utility"] = {
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "classifier_source": classifier_source,
        "feasible": release_feasible,
        "threshold": config.agda.target_accuracy,
    }
    if not release_feasible:
        row["status"] = "infeasible_materialized_release"
        row["runtime_seconds"] = time.perf_counter() - started
        return row

    if reconstruction_data is None:
        raw_arrays = (dataset.train.x, dataset.validation.x, dataset.test.x)
        targets, category_counts, bin_edges, target_strategies = (
            discretize_reconstruction_targets(raw_arrays, reconstruction_bins)
        )
        target_preprocessing = (
            "preserve low-cardinality training categories; otherwise use "
            "training-fitted quantile bins"
        )
        target_source: dict[str, Any] = {"kind": "derived_from_model_input"}
    else:
        targets = reconstruction_data.targets
        category_counts = reconstruction_data.category_counts
        bin_edges = []
        target_strategies = ["prepared_logical_categories"] * len(category_counts)
        target_preprocessing = "logical categorical IDs from data_preparation.py"
        target_source = {
            "kind": "prepared_reconstruction_sidecar",
            **reconstruction_data.metadata,
        }
    attack = evaluate_reconstruction_attack(
        releases,
        targets,
        category_counts,
        AttackerConfig(
            architecture=attacker_architecture,
            hidden_dim=attacker_hidden_dim,
            second_hidden_dim=attacker_second_hidden_dim,
            epochs=attacker_epochs,
            learning_rate=attacker_learning_rate,
            batch_size=attacker_batch_size,
            seed=config.training.seed,
            ft_token_dim=config.model.ft_token_dim,
            ft_num_heads=config.model.ft_num_heads,
            ft_num_layers=config.model.ft_num_layers,
            ft_feedforward_dim=config.model.ft_feedforward_dim,
            ft_dropout=config.model.ft_dropout,
        ),
        device=device,
    )
    row["reconstruction"] = {
        "target_preprocessing": target_preprocessing,
        "target_source": target_source,
        "target_strategies": target_strategies,
        "reconstruction_bins_requested": reconstruction_bins,
        "category_counts": category_counts,
        "training_bin_edges": bin_edges,
        **attack.metrics,
    }
    row["status"] = "completed"
    row["runtime_seconds"] = time.perf_counter() - started
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["variant_key"] == variant
                and row["status"] == "completed"
            ]
            if not selected:
                continue
            nrr = np.asarray([row["reconstruction"]["test_nrr"] for row in selected])
            accuracy = np.asarray(
                [
                    row["materialized_release_utility"]["test_accuracy"]
                    for row in selected
                ]
            )
            summary.append(
                {
                    "dataset": dataset,
                    "variant": VARIANT_LABELS[variant],
                    "variant_key": variant,
                    "completed_runs": len(selected),
                    "test_nrr_mean": float(nrr.mean()),
                    "test_nrr_std": float(nrr.std(ddof=1)) if len(nrr) > 1 else 0.0,
                    "test_accuracy_mean": float(accuracy.mean()),
                    "test_accuracy_std": (
                        float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0
                    ),
                }
            )
    return summary


def _write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _write_figure(path: Path, summary: list[dict[str, Any]]) -> bool:
    if not summary:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; JSON and CSV were written, "
            "but the plot was skipped",
            file=sys.stderr,
        )
        return False
    datasets = sorted({row["dataset"] for row in summary})
    variants = [
        variant
        for variant in VARIANTS
        if any(row["variant_key"] == variant for row in summary)
    ]
    width = 0.8 / len(variants)
    x = np.arange(len(datasets), dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(4.8, 1.3 * len(datasets)), 3.0),
        constrained_layout=True,
    )
    colors = ["#1F77FF", "#74C0FC", "#00B4D8", "#6C757D", "#4C6EF5"]
    for index, variant in enumerate(variants):
        values = []
        errors = []
        for dataset in datasets:
            match = next(
                (
                    row
                    for row in summary
                    if row["dataset"] == dataset and row["variant_key"] == variant
                ),
                None,
            )
            values.append(np.nan if match is None else match["test_nrr_mean"])
            errors.append(0.0 if match is None else match["test_nrr_std"])
        offset = (index - (len(variants) - 1) / 2) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            label=VARIANT_LABELS[variant],
            color=colors[index],
            edgecolor="black",
            linewidth=0.7,
        )
    axis.set_ylabel("normalized reconstruction risk")
    axis.set_xticks(x, datasets)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def run_experiment(
    campaign: CampaignConfig,
    *,
    variants: tuple[str, ...],
    reconstruction_bins: int,
    attacker_architecture: str,
    attacker_hidden_dim: int,
    attacker_second_hidden_dim: int,
    attacker_epochs: int,
    attacker_learning_rate: float,
    attacker_batch_size: int,
    pca_utility_epochs: int,
    output_dir: Path | None = None,
    device_override: str | None = None,
    margin: float = 0.0,
) -> dict[str, Any]:
    expanded = [
        (overrides, base_config, apply_accuracy_margin(base_config, margin))
        for overrides, base_config in expand_campaign(campaign)
    ]
    destination = (output_dir or Path(campaign.output_dir)).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for run_index, (overrides, base_config, config) in enumerate(expanded, start=1):
        if (
            config.data.kind == "synthetic"
            and config.model.aggregated_dim > config.data.num_features
        ):
            raise ValueError(
                "model.aggregated_dim exceeds the synthetic input dimension"
            )
        device = resolve_device(device_override or config.training.device)
        dataset = load_dataset(
            config.data,
            seed=config.training.seed,
            config_directory=campaign.source.parent,
        )
        reconstruction_data = load_reconstruction_data(
            config,
            dataset,
            config_directory=campaign.source.parent,
        )
        for variant_index, variant in enumerate(variants, start=1):
            print(
                f"[{run_index}/{len(expanded)}; {variant_index}/{len(variants)}] "
                f"dataset={dataset.name} seed={config.training.seed} "
                f"variant={VARIANT_LABELS[variant]}",
                flush=True,
            )
            row = run_variant(
                dataset,
                config,
                variant,
                device=device,
                reconstruction_bins=reconstruction_bins,
                attacker_architecture=attacker_architecture,
                attacker_hidden_dim=attacker_hidden_dim,
                attacker_second_hidden_dim=attacker_second_hidden_dim,
                attacker_epochs=attacker_epochs,
                attacker_learning_rate=attacker_learning_rate,
                attacker_batch_size=attacker_batch_size,
                pca_utility_epochs=pca_utility_epochs,
                reconstruction_data=reconstruction_data,
            )
            row.update(
                {
                    "dataset": dataset.name,
                    "seed": config.training.seed,
                    "overrides": overrides,
                    "margin": float(margin),
                    "configured_target_accuracy": (
                        base_config.agda.target_accuracy
                    ),
                    "effective_target_accuracy": config.agda.target_accuracy,
                }
            )
            rows.append(row)
            if row["reconstruction"] is not None:
                print(
                    "  test_acc="
                    f"{row['materialized_release_utility']['test_accuracy']:.4f} "
                    f"test_nrr={row['reconstruction']['test_nrr']:.4f}",
                    flush=True,
                )
    aggregate = summarize(rows)
    payload = {
        "experiment": "Figure 3(p): model component ablation",
        "source_config": str(campaign.source),
        "settings": {
            "variants": [VARIANT_LABELS[variant] for variant in variants],
            "reconstruction_bins": reconstruction_bins,
            "attacker_architecture": attacker_architecture,
            "attacker_hidden_dim": attacker_hidden_dim,
            "attacker_second_hidden_dim": attacker_second_hidden_dim,
            "attacker_epochs": attacker_epochs,
            "attacker_learning_rate": attacker_learning_rate,
            "attacker_batch_size": attacker_batch_size,
            "pca_utility_epochs": pca_utility_epochs,
            "margin": float(margin),
            "selection": (
                "validation feasibility and internal privacy objective only; "
                "attacker is trained post-selection"
            ),
        },
        "runs": rows,
        "summary": aggregate,
        "runtime_seconds": time.perf_counter() - started,
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(destination / "summary.csv", aggregate)
    _write_figure(destination / "figure_p_model_ablation.png", aggregate)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--reconstruction-bins", type=int, default=4)
    parser.add_argument(
        "--attacker-architecture",
        choices=("logistic", "mlp", "ft-transformer"),
        default="mlp",
    )
    parser.add_argument("--attacker-hidden-dim", type=int, default=512)
    parser.add_argument("--attacker-second-hidden-dim", type=int, default=256)
    parser.add_argument("--attacker-epochs", type=int, default=20)
    parser.add_argument("--attacker-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--attacker-batch-size", type=int, default=128)
    parser.add_argument("--pca-utility-epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="non-negative offset added to each configured target accuracy",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (
        min(
            args.attacker_hidden_dim,
            args.attacker_second_hidden_dim,
            args.attacker_epochs,
            args.attacker_batch_size,
            args.pca_utility_epochs,
        )
        <= 0
    ):
        raise ValueError("attacker and PCA utility dimensions/epochs must be positive")
    if args.attacker_learning_rate <= 0:
        raise ValueError("attacker_learning_rate must be positive")
    campaign = load_campaign(args.config)
    variants = tuple(args.variants)
    if args.dry_run:
        manifest = [
            {
                "overrides": overrides,
                "seed": config.training.seed,
                "dataset": config.data.name,
                "variants": list(variants),
                "margin": args.margin,
                "configured_target_accuracy": config.agda.target_accuracy,
                "effective_target_accuracy": apply_accuracy_margin(
                    config, args.margin
                ).agda.target_accuracy,
            }
            for overrides, config in expand_campaign(campaign)
        ]
        print(json.dumps(manifest, indent=2))
        return
    run_experiment(
        campaign,
        variants=variants,
        reconstruction_bins=args.reconstruction_bins,
        attacker_architecture=args.attacker_architecture,
        attacker_hidden_dim=args.attacker_hidden_dim,
        attacker_second_hidden_dim=args.attacker_second_hidden_dim,
        attacker_epochs=args.attacker_epochs,
        attacker_learning_rate=args.attacker_learning_rate,
        attacker_batch_size=args.attacker_batch_size,
        pca_utility_epochs=args.pca_utility_epochs,
        output_dir=args.output_dir,
        device_override=args.device,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
