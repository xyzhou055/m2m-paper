"""Config-driven campaign runner, inspired by LiteDBX's thin experiment CLI."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (
    CampaignConfig,
    RunConfig,
    apply_accuracy_margin,
    expand_campaign,
)
from .data import dataset_fingerprint, load_dataset
from .trainer import resolve_device, save_checkpoint, train_and_evaluate


def _config_hash(config: RunConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _progress(run_id: str):
    def report(row: dict[str, Any]) -> None:
        print(
            f"[{run_id}] iteration={row['iteration']:03d} "
            f"step={row['global_step']} "
            f"val_acc={row['validation_accuracy']:.4f} "
            f"lambda={row['training_multiplier']:.4g}"
            f"->{row['next_multiplier']:.4g} "
            f"action={row['controller_action']} "
            f"selection_privacy={row['selection_privacy_objective']:.6g} "
            f"feasible={row['validation_feasible']}",
            flush=True,
        )

    return report


def _summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    best = metrics.get("best_feasible_checkpoint") or {}
    final_privacy = metrics["final_privacy_terms"]
    return {
        "run_id": metrics["run_id"],
        "overrides": json.dumps(metrics["overrides"], sort_keys=True),
        "margin": metrics["margin"],
        "configured_target_accuracy": metrics["configured_target_accuracy"],
        "effective_target_accuracy": metrics["effective_target_accuracy"],
        "feasible": metrics["feasible"],
        "selected_checkpoint_source": metrics["selected_checkpoint_source"],
        "selected_iteration": metrics["selected_iteration"],
        "selected_validation_accuracy": best.get(
            "validation_accuracy", metrics["final_validation"]["accuracy"]
        ),
        "final_test_accuracy": metrics["final_test"]["accuracy"],
        "selection_privacy_objective": final_privacy["selection_privacy_objective"],
        "expected_open_count": final_privacy["expected_open_count"],
        "training_runtime_seconds": metrics["training_runtime_seconds"],
        "checkpoint": metrics.get("checkpoint"),
    }


def _write_summary(path: Path, metrics: list[dict[str, Any]]) -> None:
    rows = [_summary_row(item) for item in metrics]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def campaign_manifest(
    campaign: CampaignConfig, *, margin: float = 0.0
) -> list[dict[str, Any]]:
    manifest = []
    for index, (overrides, base_config) in enumerate(expand_campaign(campaign)):
        config = apply_accuracy_margin(base_config, margin)
        run_id = f"run-{index:04d}-{_config_hash(config)}"
        manifest.append(
            {
                "run_id": run_id,
                "overrides": overrides,
                "margin": float(margin),
                "configured_target_accuracy": base_config.agda.target_accuracy,
                "effective_target_accuracy": config.agda.target_accuracy,
                "config": config.to_dict(),
            }
        )
    return manifest


def run_campaign(
    campaign: CampaignConfig,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    margin: float = 0.0,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    expanded = []
    manifest = []
    for index, (overrides, base_config) in enumerate(expand_campaign(campaign)):
        config = apply_accuracy_margin(base_config, margin)
        if device is not None:
            config = replace(config, training=replace(config.training, device=device))
        run_id = f"run-{index:04d}-{_config_hash(config)}"
        expanded.append(
            (overrides, config, run_id, base_config.agda.target_accuracy)
        )
        manifest.append(
            {
                "run_id": run_id,
                "overrides": overrides,
                "margin": float(margin),
                "configured_target_accuracy": base_config.agda.target_accuracy,
                "effective_target_accuracy": config.agda.target_accuracy,
                "config": config.to_dict(),
            }
        )
    if dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest

    destination = Path(output_dir or campaign.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for overrides, config, run_id, configured_target_accuracy in expanded:
        dataset = load_dataset(
            config.data,
            seed=config.training.seed,
            config_directory=campaign.source.parent,
        )
        fingerprint = dataset_fingerprint(dataset)
        run_directory = destination / run_id
        metrics_path = run_directory / "metrics.json"
        if metrics_path.exists():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            previous_fingerprint = previous.get("dataset", {}).get("sha256")
            if previous_fingerprint != fingerprint:
                raise ValueError(
                    f"refusing to overwrite {run_id}: its dataset fingerprint changed"
                )
        run_directory.mkdir(parents=True, exist_ok=True)
        effective_device = resolve_device(config.training.device)
        print(
            f"[{run_id}] dataset={dataset.name} "
            f"rows={len(dataset.train.x)}/{len(dataset.validation.x)}/"
            f"{len(dataset.test.x)} device={effective_device}",
            flush=True,
        )
        result = train_and_evaluate(
            dataset,
            config,
            device=effective_device,
            progress=_progress(run_id),
        )
        metrics = result.metrics
        metrics.update(
            {
                "campaign": campaign.name,
                "run_id": run_id,
                "overrides": overrides,
                "margin": float(margin),
                "configured_target_accuracy": configured_target_accuracy,
                "effective_target_accuracy": config.agda.target_accuracy,
            }
        )
        metrics["dataset"]["sha256"] = fingerprint
        if campaign.save_checkpoints:
            checkpoint = run_directory / "checkpoint.pt"
            save_checkpoint(
                checkpoint,
                result,
                metadata={"campaign": campaign.name, "run_id": run_id},
            )
            metrics["checkpoint"] = str(checkpoint)
        else:
            metrics["checkpoint"] = None
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(metrics)
        _write_summary(destination / "summary.csv", results)
        print(f"[{run_id}] saved {metrics_path}", flush=True)
    return results
