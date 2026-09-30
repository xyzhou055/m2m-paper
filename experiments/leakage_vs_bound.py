"""Gaussian leakage versus the Theorem 4 trace bound.

Train AGDA on a Gaussian reference channel with a known population covariance.
Leakage is evaluated only for validation-feasible selected checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from m2m_agda.config import (  # noqa: E402
    CampaignConfig,
    DataConfig,
    apply_accuracy_margin,
    expand_campaign,
    load_campaign,
)
from m2m_agda.data import DatasetSplits, Split  # noqa: E402
from m2m_agda.trainer import resolve_device, train_and_evaluate  # noqa: E402

DEFAULT_CONFIG = ROOT / "experiments/configs/leakage_vs_bound.yaml"


def make_covariance(
    num_features: int, condition_number: float, seed: int
) -> np.ndarray:
    """Construct a deterministic positive-definite covariance matrix."""

    if num_features < 2:
        raise ValueError("the leakage experiment needs at least two features")
    if condition_number < 1:
        raise ValueError("condition_number must be at least 1")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(num_features, num_features)))
    eigenvalues = np.geomspace(1.0, condition_number, num_features)
    covariance = (basis * eigenvalues) @ basis.T
    return 0.5 * (covariance + covariance.T)


def _stratified_split(
    labels: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: tuple[list[int], list[int], list[int]] = ([], [], [])
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        if len(indices) < 3:
            raise ValueError("each class needs at least three examples")
        train_end = min(
            max(1, int(round(len(indices) * train_fraction))), len(indices) - 2
        )
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_end = min(train_end + validation_count, len(indices) - 1)
        groups[0].extend(indices[:train_end])
        groups[1].extend(indices[train_end:validation_end])
        groups[2].extend(indices[validation_end:])
    resolved = []
    for group in groups:
        indices = np.asarray(group, dtype=np.int64)
        rng.shuffle(indices)
        resolved.append(indices)
    return tuple(resolved)  # type: ignore[return-value]


def generate_gaussian_dataset(
    config: DataConfig,
    *,
    seed: int,
    condition_number: float,
    label_noise_std: float,
) -> tuple[DatasetSplits, np.ndarray, dict[str, Any]]:
    """Generate the reference distribution used by the bound experiment."""

    if config.kind != "synthetic":
        raise ValueError("Figure F requires data.kind='synthetic'")
    if config.num_classes != 2:
        raise ValueError("Figure F currently requires data.num_classes=2")
    if label_noise_std < 0:
        raise ValueError("label_noise_std must be non-negative")
    covariance = make_covariance(config.num_features, condition_number, seed + 10_001)
    rng = np.random.default_rng(seed)
    features = rng.multivariate_normal(
        np.zeros(config.num_features), covariance, size=config.num_samples
    ).astype(np.float32)
    relevant = np.sort(
        rng.choice(
            config.num_features,
            size=config.num_relevant_features,
            replace=False,
        )
    )
    coefficients = rng.normal(size=config.num_relevant_features)
    coefficients /= np.linalg.norm(coefficients)
    score = features[:, relevant] @ coefficients
    if label_noise_std > 0:
        score += rng.normal(scale=label_noise_std, size=config.num_samples)
    threshold = float(np.median(score))
    labels = (score >= threshold).astype(np.int64)
    indices = _stratified_split(
        labels,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=seed,
    )
    dataset = DatasetSplits(
        name=config.name,
        train=Split(features[indices[0]], labels[indices[0]]),
        validation=Split(features[indices[1]], labels[indices[1]]),
        test=Split(features[indices[2]], labels[indices[2]]),
        class_names=("0", "1"),
    )
    metadata = {
        "distribution": "zero-mean multivariate Gaussian",
        "population_covariance": "known; no feature standardization",
        "condition_number": condition_number,
        "population_covariance_eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
        "relevant_feature_indices": relevant.tolist(),
        "teacher_coefficients": coefficients.tolist(),
        "label_noise_std": label_noise_std,
        "label_threshold": threshold,
    }
    return dataset, covariance, metadata


def theorem_metrics(
    model: torch.nn.Module,
    population_covariance: np.ndarray,
    noise_std: float,
) -> dict[str, Any]:
    """Compute exact Gaussian MI and the trace upper bound in nats."""

    if noise_std <= 0:
        raise ValueError("finite Gaussian leakage requires noise_std > 0")
    with torch.no_grad():
        gates = (
            model.minimizer.feature_removal.deterministic_gates()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        weights = (
            model.minimizer.aggregation.weight.detach().cpu().numpy().astype(np.float64)
        )
    masked_covariance = gates[:, None] * population_covariance * gates[None, :]
    masked_covariance = 0.5 * (masked_covariance + masked_covariance.T)
    projected_covariance = weights.T @ masked_covariance @ weights
    projected_covariance = 0.5 * (projected_covariance + projected_covariance.T)
    projected_trace = float(np.trace(projected_covariance))
    matrix = np.eye(weights.shape[1]) + projected_covariance / noise_std**2
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise FloatingPointError("Gaussian channel covariance is not positive definite")
    actual = float(0.5 * log_determinant)
    bound = float(
        0.5
        * weights.shape[1]
        * math.log1p(projected_trace / (weights.shape[1] * noise_std**2))
    )
    eigenvalues, eigenvectors = np.linalg.eigh(masked_covariance)
    alignments = np.sum((eigenvectors.T @ weights) ** 2, axis=1)
    spectral_trace = float(np.dot(eigenvalues, alignments))
    slack = bound - actual
    return {
        "actual_leakage_nats": actual,
        "theorem_bound_nats": bound,
        "absolute_slack_nats": slack,
        "relative_slack": None if actual <= 1e-12 else slack / actual,
        "bound_to_actual_ratio": None if actual <= 1e-12 else bound / actual,
        "bound_holds_within_numerical_tolerance": actual <= bound + 1e-9,
        "projected_trace": projected_trace,
        "spectral_trace": spectral_trace,
        "spectral_trace_residual": spectral_trace - projected_trace,
        "orthogonality_error_frobenius": float(
            np.linalg.norm(weights.T @ weights - np.eye(weights.shape[1]), ord="fro")
        ),
        "masked_covariance_trace": float(np.trace(masked_covariance)),
        "deterministic_gates": gates.tolist(),
        "deterministic_nonzero_gates": int(np.sum(gates > 1e-6)),
        "mean_deterministic_gate": float(gates.mean()),
        "assumptions": {
            "input": "Gaussian with known population covariance",
            "gates": "fixed deterministic hard-concrete gates after selection",
            "release": "O = W^T Z + epsilon; epsilon ~ N(0, sigma^2 I)",
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["status"] == "feasible"]
    if not feasible:
        return {"feasible_runs": 0, "total_runs": len(rows)}
    actual = np.asarray([row["theorem"]["actual_leakage_nats"] for row in feasible])
    bound = np.asarray([row["theorem"]["theorem_bound_nats"] for row in feasible])
    relative = np.asarray(
        [
            row["theorem"]["relative_slack"]
            for row in feasible
            if row["theorem"]["relative_slack"] is not None
        ]
    )
    return {
        "feasible_runs": len(feasible),
        "total_runs": len(rows),
        "bound_violations": int(np.sum(actual > bound + 1e-9)),
        "max_actual_minus_bound": float(np.max(actual - bound)),
        "mean_actual_leakage_nats": float(actual.mean()),
        "mean_theorem_bound_nats": float(bound.mean()),
        "mean_absolute_slack_nats": float((bound - actual).mean()),
        "mean_relative_slack": (None if len(relative) == 0 else float(relative.mean())),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "seed",
        "aggregated_dim",
        "noise_std",
        "sparsity_weight",
        "margin",
        "configured_target_accuracy",
        "target_accuracy",
        "status",
        "validation_accuracy",
        "test_accuracy",
        "actual_leakage_nats",
        "theorem_bound_nats",
        "absolute_slack_nats",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            theorem = row.get("theorem") or {}
            writer.writerow(
                {
                    **row["configuration"],
                    "status": row["status"],
                    "validation_accuracy": row["validation"]["accuracy"],
                    "test_accuracy": row["test"]["accuracy"],
                    "actual_leakage_nats": theorem.get("actual_leakage_nats"),
                    "theorem_bound_nats": theorem.get("theorem_bound_nats"),
                    "absolute_slack_nats": theorem.get("absolute_slack_nats"),
                }
            )


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> bool:
    feasible = [row for row in rows if row["status"] == "feasible"]
    if not feasible:
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
    actual = np.asarray([row["theorem"]["actual_leakage_nats"] for row in feasible])
    bound = np.asarray([row["theorem"]["theorem_bound_nats"] for row in feasible])
    limit = 1.05 * float(max(actual.max(), bound.max(), 1e-6))
    figure, axis = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    axis.plot([0, limit], [0, limit], "--", color="#555555", linewidth=1.6)
    axis.scatter(
        actual,
        bound,
        s=52,
        color="#1F77FF",
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    axis.set(xlabel=r"$I(X;O)$ (nats)", ylabel=r"$B_{th}$ (nats)")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.8)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def run_experiment(
    campaign: CampaignConfig,
    *,
    condition_number: float,
    label_noise_std: float,
    output_dir: Path | None = None,
    device_override: str | None = None,
    margin: float = 0.0,
) -> dict[str, Any]:
    """Run all expanded configurations and write paper-ready artifacts."""

    expanded = [
        (overrides, base_config, apply_accuracy_margin(base_config, margin))
        for overrides, base_config in expand_campaign(campaign)
    ]
    destination = (output_dir or Path(campaign.output_dir)).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device_override or campaign.base.training.device)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], tuple[DatasetSplits, np.ndarray, dict[str, Any]]] = {}
    started = time.perf_counter()
    for index, (overrides, base_config, config) in enumerate(expanded, start=1):
        if config.model.noise_std <= 0:
            raise ValueError("every Figure F configuration needs model.noise_std > 0")
        key = (
            config.training.seed,
            config.data.num_samples,
            config.data.num_features,
            config.data.num_relevant_features,
            config.data.train_fraction,
            config.data.validation_fraction,
        )
        if key not in cache:
            cache[key] = generate_gaussian_dataset(
                config.data,
                seed=config.training.seed,
                condition_number=condition_number,
                label_noise_std=label_noise_std,
            )
        dataset, covariance, data_metadata = cache[key]
        print(
            f"[{index}/{len(expanded)}] seed={config.training.seed} "
            f"k={config.model.aggregated_dim} sigma={config.model.noise_std:g} "
            f"eta={config.objective.sparsity_weight:g} "
            f"target_accuracy={config.agda.target_accuracy:g}",
            flush=True,
        )
        result = train_and_evaluate(dataset, config, device=resolved_device)
        feasible = bool(result.metrics["feasible"])
        theorem = (
            theorem_metrics(result.model, covariance, config.model.noise_std)
            if feasible
            else None
        )
        row = {
            "overrides": overrides,
            "configuration": {
                "seed": config.training.seed,
                "aggregated_dim": config.model.aggregated_dim,
                "noise_std": config.model.noise_std,
                "sparsity_weight": config.objective.sparsity_weight,
                "margin": float(margin),
                "configured_target_accuracy": (
                    base_config.agda.target_accuracy
                ),
                "target_accuracy": config.agda.target_accuracy,
            },
            "dataset": data_metadata,
            "status": "feasible" if feasible else "infeasible",
            "selection": {
                "selected_checkpoint_source": result.metrics[
                    "selected_checkpoint_source"
                ],
                "selected_iteration": result.metrics["selected_iteration"],
                "best_feasible_checkpoint": result.metrics["best_feasible_checkpoint"],
            },
            "validation": result.metrics["final_validation"],
            "test": result.metrics["final_test"],
            "final_privacy_terms": result.metrics["final_privacy_terms"],
            "runtime_seconds": result.metrics["training_runtime_seconds"],
            "theorem": theorem,
        }
        rows.append(row)
        if theorem is not None:
            print(
                f"  I={theorem['actual_leakage_nats']:.6f} "
                f"B_th={theorem['theorem_bound_nats']:.6f} "
                f"holds={theorem['bound_holds_within_numerical_tolerance']}",
                flush=True,
            )
    payload = {
        "experiment": "Figure 3(f): synthetic leakage versus bound",
        "source_config": str(campaign.source),
        "settings": {
            "condition_number": condition_number,
            "label_noise_std": label_noise_std,
            "margin": float(margin),
            "device": str(resolved_device),
            "theorem_bound": "k/2 * log(1 + Tr(W^T Sigma_Z W)/(k sigma^2))",
            "selection": (
                "lowest AGDA privacy objective among validation-feasible checkpoints"
            ),
        },
        "runs": rows,
        "summary": summarize(rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_table(destination / "points.csv", rows)
    _write_figure(destination / "figure_f_leakage_vs_bound.png", rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--condition-number", type=float, default=25.0)
    parser.add_argument("--label-noise-std", type=float, default=1.0)
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
    campaign = load_campaign(args.config)
    if args.dry_run:
        manifest = [
            {
                "overrides": overrides,
                "margin": args.margin,
                "configured_target_accuracy": config.agda.target_accuracy,
                "config": apply_accuracy_margin(config, args.margin).to_dict(),
            }
            for overrides, config in expand_campaign(campaign)
        ]
        print(json.dumps(manifest, indent=2))
        return
    run_experiment(
        campaign,
        condition_number=args.condition_number,
        label_noise_std=args.label_noise_std,
        output_dir=args.output_dir,
        device_override=args.device,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
