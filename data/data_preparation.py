#!/usr/bin/env python3
"""Prepare supported tabular datasets for AGDA experiments.

The shared pipeline splits before fitting preprocessing, learns every feature
schema from training rows, and writes two aligned artifacts:

* a six-array model NPZ consumed by :mod:`m2m_agda.data`; and
* a reconstruction NPZ containing logical categorical IDs for Figure 3(p).

Adult, CoverType, Mushroom, and Marketing use train-fitted categorical one-hot
encoding. Adult and CoverType numeric columns are first converted to
train-fitted quantile bins. NATICUSdroid's native binary indicators are kept
as-is after constant columns are removed. Every recipe keeps all source rows;
there is no rebalancing or subsampling.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

try:
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
except ImportError as error:  # pragma: no cover - depends on optional installation
    raise ImportError(
        'dataset preparation requires: python -m pip install -e ".[data]"'
    ) from error

from m2m_agda.data import dataset_fingerprint, load_npz  # noqa: E402

DATASETS = ("adult", "covertype", "marketing", "mushroom", "naticusdroid")
DATASET_ALIASES = {
    "covtype": "covertype",
    "coverypte": "covertype",
}
DATASET_CHOICES = (*DATASETS, *DATASET_ALIASES)
ADULT_OPENML_ID = 1590
COVERTYPE_OPENML_ID = 1596
MUSHROOM_OPENML_ID = 24
NATICUSDROID_URL = (
    "https://archive.ics.uci.edu/static/public/722/"
    "naticusdroid%2Bandroid%2Bpermissions%2Bdataset.zip"
)
CLEANML_DATASETS_URL = (
    "https://www.dropbox.com/s/nerfrhbrseev928/CleanML-datasets-2020.zip?dl=1"
)
ADULT_CONTINUOUS = {
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
}
TARGET_CANDIDATES = {
    "adult": ("__adult_target__", "class", "income", "target"),
    "covertype": (
        "__covertype_target__",
        "Cover_Type",
        "cover_type",
        "class",
        "target",
    ),
    "marketing": ("Income", "income", "target"),
    "mushroom": ("class", "poisonous", "target"),
    "naticusdroid": ("Result", "result", "target"),
}


@dataclass(frozen=True)
class PreparationConfig:
    seed: int = 0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    continuous_bins: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than 1")
        if self.continuous_bins < 2:
            raise ValueError("continuous_bins must be at least 2")


@dataclass(frozen=True)
class PreparedArtifacts:
    name: str
    encoded_splits: tuple[np.ndarray, np.ndarray, np.ndarray]
    raw_splits: tuple[np.ndarray, np.ndarray, np.ndarray]
    label_splits: tuple[np.ndarray, np.ndarray, np.ndarray]
    feature_names: list[str]
    feature_types: list[str]
    feature_mapping: list[tuple[int, int]]
    category_counts: list[int]
    encoded_column_names: list[str]
    class_names: list[str]
    preprocessing: dict[str, Any]
    source: dict[str, Any]


def clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, whitespace, and common missing markers."""

    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    for column in cleaned.columns:
        values = cleaned[column]
        if (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
            or isinstance(values.dtype, pd.CategoricalDtype)
        ):
            normalized = values.astype("object").where(values.notna(), "__MISSING__")
            cleaned[column] = (
                normalized.astype(str)
                .str.strip()
                .replace({"?": "__MISSING__", "nan": "__MISSING__"})
            )
    return cleaned


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    values = labels.astype(str).str.strip().to_numpy()
    classes = sorted(np.unique(values).tolist())
    if len(classes) < 2:
        raise ValueError("the target must contain at least two classes")
    lookup = {value: index for index, value in enumerate(classes)}
    encoded = np.asarray([lookup[value] for value in values], dtype=np.int64)
    return encoded, classes


def split_stratified(
    frame: pd.DataFrame,
    labels: np.ndarray,
    config: PreparationConfig,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    train_validation, test, y_train_validation, y_test = train_test_split(
        frame,
        labels,
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=labels,
    )
    relative_validation = config.validation_fraction / (1 - config.test_fraction)
    train, validation, y_train, y_validation = train_test_split(
        train_validation,
        y_train_validation,
        test_size=relative_validation,
        random_state=config.seed,
        stratify=y_train_validation,
    )
    frames = tuple(part.reset_index(drop=True) for part in (train, validation, test))
    label_splits = tuple(
        np.ascontiguousarray(values, dtype=np.int64)
        for values in (y_train, y_validation, y_test)
    )
    return frames, label_splits  # type: ignore[return-value]


def encode_categorical_features(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    *,
    numerical_columns: set[str],
    continuous_bins: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    list[str],
    list[tuple[int, int]],
    list[int],
    list[str],
    dict[str, Any],
]:
    """Fit categorical schemas on training and transform all three splits."""

    train, validation, test = frames
    encoded_parts: list[list[np.ndarray]] = [[], [], []]
    raw_parts: list[list[np.ndarray]] = [[], [], []]
    feature_names: list[str] = []
    feature_mapping: list[tuple[int, int]] = []
    category_counts: list[int] = []
    encoded_names: list[str] = []
    details: dict[str, Any] = {
        "schema_source": "training_split",
        "continuous_binning": {},
        "categorical": {},
        "removed_features": [],
    }
    offset = 0
    for raw_name in train.columns:
        name = str(raw_name)
        if name in numerical_columns:
            train_numeric = pd.to_numeric(train[raw_name], errors="coerce")
            median = float(train_numeric.median())
            train_values = train_numeric.fillna(median).to_numpy(dtype=np.float32)
            cuts = np.unique(
                np.quantile(
                    train_values,
                    np.linspace(0, 1, continuous_bins + 1)[1:-1],
                )
            )
            train_bins = np.searchsorted(cuts, train_values, side="right")
            observed = sorted(np.unique(train_bins).tolist())
            lookup = {value: index for index, value in enumerate(observed)}
            category_labels = [f"bin_{value}" for value in observed]
            split_values = [
                pd.to_numeric(part[raw_name], errors="coerce")
                .fillna(median)
                .to_numpy(dtype=np.float32)
                for part in frames
            ]
            split_ids = [
                np.asarray(
                    [
                        lookup.get(value, 0)
                        for value in np.searchsorted(cuts, values, side="right")
                    ],
                    dtype=np.int64,
                )
                for values in split_values
            ]
            details["continuous_binning"][name] = {
                "impute_median": median,
                "requested_bins": continuous_bins,
                "cut_points": cuts.tolist(),
                "training_observed_bin_ids": observed,
            }
        else:
            training_values = (
                train[raw_name]
                .astype("object")
                .where(train[raw_name].notna(), "__MISSING__")
                .astype(str)
            )
            category_labels = sorted(pd.unique(training_values).tolist())
            lookup = {value: index for index, value in enumerate(category_labels)}
            string_splits = [
                part[raw_name]
                .astype("object")
                .where(part[raw_name].notna(), "__MISSING__")
                .astype(str)
                for part in frames
            ]
            split_ids = [
                np.asarray([lookup.get(value, 0) for value in values], dtype=np.int64)
                for values in string_splits
            ]
            details["categorical"][name] = {
                "categories": category_labels,
                "unknown_validation_mapped_to_first_category": int(
                    sum(value not in lookup for value in string_splits[1])
                ),
                "unknown_test_mapped_to_first_category": int(
                    sum(value not in lookup for value in string_splits[2])
                ),
            }
        width = len(category_labels)
        if width < 2:
            details["removed_features"].append(
                {
                    "feature": name,
                    "reason": "fewer than two training categories",
                }
            )
            continue
        for destination, identifiers in zip(encoded_parts, split_ids, strict=True):
            one_hot = np.zeros((len(identifiers), width), dtype=np.float32)
            one_hot[np.arange(len(identifiers)), identifiers] = 1.0
            destination.append(one_hot)
        for destination, identifiers in zip(raw_parts, split_ids, strict=True):
            destination.append(identifiers)
        feature_names.append(name)
        feature_mapping.append((offset, offset + width))
        category_counts.append(width)
        encoded_names.extend(f"{name}:{value}" for value in category_labels)
        offset += width
    if not feature_names:
        raise ValueError("no nonconstant features remain after preprocessing")
    encoded = tuple(
        np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
        for parts in encoded_parts
    )
    raw = tuple(
        np.ascontiguousarray(np.column_stack(parts), dtype=np.int64)
        for parts in raw_parts
    )
    return (
        encoded,  # type: ignore[return-value]
        raw,  # type: ignore[return-value]
        feature_names,
        feature_mapping,
        category_counts,
        encoded_names,
        details,
    )


def canonical_dataset_name(dataset: str) -> str:
    """Resolve supported aliases to the artifact's canonical dataset name."""

    resolved = DATASET_ALIASES.get(dataset, dataset)
    if resolved not in DATASETS:
        choices = ", ".join(DATASET_CHOICES)
        raise ValueError(f"unknown dataset {dataset!r}; choose from {choices}")
    return resolved


def prepare_from_frame(
    dataset: str,
    frame: pd.DataFrame,
    target: pd.Series,
    config: PreparationConfig,
    *,
    source: dict[str, Any] | None = None,
) -> PreparedArtifacts:
    """Apply a supported dataset recipe to an already-loaded frame."""

    dataset = canonical_dataset_name(dataset)
    frame = clean_strings(frame).reset_index(drop=True)
    target = target.reset_index(drop=True).astype(str).str.strip()
    if len(frame) != len(target):
        raise ValueError("features and target have different row counts")
    sampling: dict[str, Any] = {
        "mode": "full_dataset_no_subsampling",
        "rows": len(frame),
        "class_counts_before_split": {
            str(key): int(value)
            for key, value in target.value_counts().sort_index().items()
        },
    }
    if dataset == "adult":
        target = target.str.replace(".", "", regex=False)
        unexpected = sorted(set(target) - {"<=50K", ">50K"})
        if unexpected:
            raise ValueError(f"Adult contains unexpected target values: {unexpected}")
        sampling["class_counts_before_split"] = {
            str(key): int(value)
            for key, value in target.value_counts().sort_index().items()
        }
    labels, class_names = encode_labels(target)
    frames, label_splits = split_stratified(frame, labels, config)

    if dataset == "naticusdroid":
        numeric_frames = tuple(
            part.apply(pd.to_numeric, errors="raise") for part in frames
        )
        invalid = [
            str(column)
            for column in numeric_frames[0].columns
            if not set(frame[column].dropna().astype(int).unique()).issubset({0, 1})
        ]
        if invalid:
            raise ValueError(
                f"NATICUSdroid contains non-binary feature columns: {invalid}"
            )
        constant = [
            column
            for column in numeric_frames[0].columns
            if numeric_frames[0][column].nunique() < 2
        ]
        numeric_frames = tuple(part.drop(columns=constant) for part in numeric_frames)
        feature_names = [str(column) for column in numeric_frames[0].columns]
        encoded = tuple(
            np.ascontiguousarray(part.to_numpy(), dtype=np.float32)
            for part in numeric_frames
        )
        raw = tuple(
            np.ascontiguousarray(part.to_numpy(), dtype=np.int64)
            for part in numeric_frames
        )
        feature_mapping = [(index, index + 1) for index in range(len(feature_names))]
        category_counts = [2] * len(feature_names)
        encoded_names = list(feature_names)
        preprocessing = {
            "mode": "native_binary",
            "schema_source": "training_split",
            "removed_training_constant_features": [str(item) for item in constant],
            "sampling": sampling,
        }
        feature_types = ["binary"] * len(feature_names)
    else:
        if dataset == "adult":
            numerical = {
                str(column)
                for column in frames[0].columns
                if str(column) in ADULT_CONTINUOUS
                or pd.api.types.is_numeric_dtype(frames[0][column])
            }
        elif dataset == "covertype":
            numerical = {
                str(column)
                for column in frames[0].columns
                if pd.api.types.is_numeric_dtype(frames[0][column])
                and frames[0][column].nunique(dropna=True) > 10
            }
        else:
            numerical = set()
        (
            encoded,
            raw,
            feature_names,
            feature_mapping,
            category_counts,
            encoded_names,
            preprocessing,
        ) = encode_categorical_features(
            frames,
            numerical_columns=numerical,
            continuous_bins=config.continuous_bins,
        )
        preprocessing.update(
            {
                "mode": (
                    "quantile_binned_and_one_hot"
                    if numerical
                    else "all_categorical"
                ),
                "continuous_bins": config.continuous_bins,
                "numerical_columns_binned": sorted(numerical),
                "sampling": sampling,
            }
        )
        feature_types = [
            "binned_continuous" if name in numerical else "categorical"
            for name in feature_names
        ]
    if encoded[0].shape[1] == 0:
        raise ValueError("no encoded features remain")
    preprocessing["split_seed"] = config.seed
    preprocessing["validation_fraction"] = config.validation_fraction
    preprocessing["test_fraction"] = config.test_fraction
    return PreparedArtifacts(
        name=dataset,
        encoded_splits=encoded,
        raw_splits=raw,
        label_splits=label_splits,
        feature_names=feature_names,
        feature_types=feature_types,
        feature_mapping=feature_mapping,
        category_counts=category_counts,
        encoded_column_names=encoded_names,
        class_names=class_names,
        preprocessing=preprocessing,
        source=source or {"kind": "in_memory"},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_column(frame: pd.DataFrame, dataset: str, requested: str | None) -> str:
    dataset = canonical_dataset_name(dataset)
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"target column {requested!r} is absent from the source")
        return requested
    for candidate in TARGET_CANDIDATES[dataset]:
        if candidate in frame.columns:
            return candidate
    choices = ", ".join(TARGET_CANDIDATES[dataset])
    raise ValueError(f"could not infer target column; pass --target-column ({choices})")


def load_local_source(
    path: Path, dataset: str, target_column: str | None
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset = canonical_dataset_name(dataset)
    source = path.expanduser().resolve()
    frame = pd.read_csv(source)
    target_name = _target_column(frame, dataset, target_column)
    target = frame.pop(target_name)
    return (
        frame,
        target,
        {
            "kind": "local_csv",
            "path": str(source),
            "sha256": _sha256(source),
            "target_column": target_name,
        },
    )


def download_source(
    dataset: str, cache_dir: Path
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset = canonical_dataset_name(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset in {"adult", "covertype", "mushroom"}:
        data_ids = {
            "adult": ADULT_OPENML_ID,
            "covertype": COVERTYPE_OPENML_ID,
            "mushroom": MUSHROOM_OPENML_ID,
        }
        data_id = data_ids[dataset]
        fetched = fetch_openml(
            data_id=data_id,
            as_frame=True,
            parser="auto",
            data_home=str(cache_dir / "openml"),
        )
        return (
            fetched.data,
            pd.Series(fetched.target),
            {
                "kind": "openml",
                "data_id": data_id,
                "cache_directory": str(cache_dir / "openml"),
            },
        )
    if dataset == "marketing":
        csv_path = cache_dir / "marketing.csv"
        archive_path = cache_dir / "CleanML-datasets-2020.zip"
        if not csv_path.exists():
            if not archive_path.exists():
                urllib.request.urlretrieve(CLEANML_DATASETS_URL, archive_path)
            with zipfile.ZipFile(archive_path) as zipped:
                member = next(
                    (
                        name
                        for name in zipped.namelist()
                        if name.endswith("Marketing/raw/raw.csv")
                    ),
                    None,
                )
                if member is None:
                    raise ValueError(
                        "CleanML archive does not contain Marketing/raw/raw.csv"
                    )
                with zipped.open(member) as source, csv_path.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
        frame = pd.read_csv(csv_path)
        target_name = _target_column(frame, dataset, None)
        target = frame.pop(target_name)
        return (
            frame,
            target,
            {
                "kind": "cleanml_download_cache",
                "url": CLEANML_DATASETS_URL,
                "archive_path": str(archive_path),
                "path": str(csv_path),
                "sha256": _sha256(csv_path),
                "target_column": target_name,
            },
        )

    csv_path = cache_dir / "naticusdroid.csv"
    if not csv_path.exists():
        with urllib.request.urlopen(NATICUSDROID_URL) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            member = next(
                name
                for name in zipped.namelist()
                if Path(name).name.lower() == "data.csv"
            )
            frame = pd.read_csv(zipped.open(member))
        frame.to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path)
    target_name = _target_column(frame, dataset, None)
    target = frame.pop(target_name)
    return (
        frame,
        target,
        {
            "kind": "uci_download_cache",
            "url": NATICUSDROID_URL,
            "path": str(csv_path),
            "sha256": _sha256(csv_path),
            "target_column": target_name,
        },
    )


def write_artifacts(
    prepared: PreparedArtifacts, output: Path, *, force: bool
) -> tuple[Path, Path, Path]:
    destination = output.expanduser().resolve()
    reconstruction_path = destination.with_suffix(".reconstruction.npz")
    metadata_path = destination.with_suffix(".metadata.json")
    existing = [
        path
        for path in (destination, reconstruction_path, metadata_path)
        if path.exists()
    ]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output exists: {names}; pass --force to replace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=prepared.encoded_splits[0],
        y_train=prepared.label_splits[0],
        x_val=prepared.encoded_splits[1],
        y_val=prepared.label_splits[1],
        x_test=prepared.encoded_splits[2],
        y_test=prepared.label_splits[2],
    )
    np.savez_compressed(
        reconstruction_path,
        x_train_raw=prepared.raw_splits[0],
        x_val_raw=prepared.raw_splits[1],
        x_test_raw=prepared.raw_splits[2],
        category_counts=np.asarray(prepared.category_counts, dtype=np.int64),
    )
    dataset = load_npz(destination, name=prepared.name)
    metadata = {
        "format_version": 1,
        "dataset": prepared.name,
        "model_archive": destination.name,
        "reconstruction_archive": reconstruction_path.name,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "source": prepared.source,
        "splits": {
            "seed": prepared.preprocessing.get("split_seed"),
            "rows": {
                "train": len(prepared.label_splits[0]),
                "validation": len(prepared.label_splits[1]),
                "test": len(prepared.label_splits[2]),
            },
        },
        "class_names": prepared.class_names,
        "feature_names": prepared.feature_names,
        "feature_types": prepared.feature_types,
        "feature_mapping": prepared.feature_mapping,
        "category_counts": prepared.category_counts,
        "encoded_column_names": prepared.encoded_column_names,
        "preprocessing": prepared.preprocessing,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination, reconstruction_path, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="local raw CSV")
    source.add_argument(
        "--download",
        action="store_true",
        help="download from the dataset's OpenML, UCI, or CleanML source and cache it",
    )
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--continuous-bins", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = canonical_dataset_name(args.dataset)
    config = PreparationConfig(
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        continuous_bins=args.continuous_bins,
    )
    if args.source is not None:
        frame, target, source = load_local_source(
            args.source, dataset, args.target_column
        )
    else:
        frame, target, source = download_source(dataset, args.cache_dir)
    prepared = prepare_from_frame(
        dataset,
        frame,
        target,
        config,
        source=source,
    )
    output = args.output or ROOT / f"data/prepared/{dataset}.npz"
    model_path, reconstruction_path, metadata_path = write_artifacts(
        prepared, output, force=args.force
    )
    print(f"Saved model data: {model_path}")
    print(f"Saved reconstruction targets: {reconstruction_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"Rows={tuple(len(values) for values in prepared.label_splits)} "
        f"logical_features={len(prepared.feature_names)} "
        f"encoded_features={prepared.encoded_splits[0].shape[1]}"
    )


if __name__ == "__main__":
    main()
