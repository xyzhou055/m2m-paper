#!/usr/bin/env python3
"""Generate a model-ready synthetic NPZ archive for M2M-AGDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import numpy as np  # noqa: E402

from m2m_agda.config import DataConfig  # noqa: E402
from m2m_agda.data import dataset_fingerprint, make_synthetic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stratified, train-standardized synthetic classification "
            "data in the six-array M2M-AGDA NPZ format."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "synthetic_ready.npz",
    )
    parser.add_argument("--name", default="synthetic-ready")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--num-features", type=int, default=100)
    parser.add_argument("--num-relevant-features", type=int, default=50)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-separation", type=float, default=1.5)
    parser.add_argument("--label-noise", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output archive",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        raise FileExistsError(
            f"{destination} already exists; pass --force to replace it"
        )
    config = DataConfig(
        kind="synthetic",
        name=args.name,
        num_samples=args.num_samples,
        num_features=args.num_features,
        num_relevant_features=args.num_relevant_features,
        num_classes=args.num_classes,
        class_separation=args.class_separation,
        label_noise=args.label_noise,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    dataset = make_synthetic(config, seed=args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=dataset.train.x,
        y_train=dataset.train.y,
        x_val=dataset.validation.x,
        y_val=dataset.validation.y,
        x_test=dataset.test.x,
        y_test=dataset.test.y,
    )
    print(f"Saved {destination}")
    print(
        "Rows "
        f"train={len(dataset.train.x)} "
        f"validation={len(dataset.validation.x)} "
        f"test={len(dataset.test.x)}"
    )
    print(
        f"Features={dataset.input_dim} classes={dataset.num_classes} "
        f"sha256={dataset_fingerprint(dataset)}"
    )


if __name__ == "__main__":
    main()
