"""Small, network-free dataset boundary for M2M-AGDA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        raw_y = np.asarray(self.y)
        if raw_y.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if not np.issubdtype(raw_y.dtype, np.integer):
            raise ValueError("labels must use an integer dtype")
        x = np.asarray(self.x, dtype=np.float32)
        y = raw_y.astype(np.int64, copy=False)
        if x.ndim != 2:
            raise ValueError("feature arrays must be two-dimensional")
        if x.shape[1] == 0:
            raise ValueError("feature arrays must have at least one column")
        if len(x) != len(y):
            raise ValueError("labels must align with feature rows")
        if len(x) == 0:
            raise ValueError("dataset splits must not be empty")
        if not np.isfinite(x).all():
            raise ValueError("features must contain only finite values")
        object.__setattr__(self, "x", np.ascontiguousarray(x))
        object.__setattr__(self, "y", np.ascontiguousarray(y))


@dataclass(frozen=True)
class DatasetSplits:
    name: str
    train: Split
    validation: Split
    test: Split
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        dimensions = {
            self.train.x.shape[1],
            self.validation.x.shape[1],
            self.test.x.shape[1],
        }
        if len(dimensions) != 1:
            raise ValueError("all splits must have the same feature dimension")
        if len(self.class_names) < 2:
            raise ValueError("at least two training classes are required")
        for split in (self.train, self.validation, self.test):
            if split.y.min() < 0 or split.y.max() >= len(self.class_names):
                raise ValueError("labels must index class_names")

    @property
    def input_dim(self) -> int:
        return int(self.train.x.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _encode_labels_from_training(
    y_train: np.ndarray, y_validation: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    train_values = np.unique(y_train)
    if len(train_values) < 2:
        raise ValueError("training labels must contain at least two classes")
    mapping = {
        value.item() if hasattr(value, "item") else value: i
        for i, value in enumerate(train_values)
    }

    def encode(values: np.ndarray, split_name: str) -> np.ndarray:
        encoded = []
        for value in np.asarray(values).reshape(-1):
            key = value.item() if hasattr(value, "item") else value
            if key not in mapping:
                raise ValueError(
                    f"{split_name} contains label {key!r} absent from training"
                )
            encoded.append(mapping[key])
        return np.asarray(encoded, dtype=np.int64)

    class_names = tuple(str(value) for value in train_values.tolist())
    return (
        encode(y_train, "training"),
        encode(y_validation, "validation"),
        encode(y_test, "test"),
        class_names,
    )


def load_npz(path: str | Path, *, name: str | None = None) -> DatasetSplits:
    source = Path(path).expanduser().resolve()
    required = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}
    with np.load(source, allow_pickle=False) as archive:
        present = set(archive.files)
        if present != required:
            missing = sorted(required - present)
            extra = sorted(present - required)
            raise ValueError(
                "NPZ schema mismatch; "
                f"missing={missing or 'none'}, extra={extra or 'none'}"
            )
        arrays = {key: archive[key] for key in required}
    for split_name, x_key, y_key in (
        ("training", "x_train", "y_train"),
        ("validation", "x_val", "y_val"),
        ("test", "x_test", "y_test"),
    ):
        if arrays[x_key].ndim != 2:
            raise ValueError(f"{split_name} features must be two-dimensional")
        if arrays[y_key].ndim != 1:
            raise ValueError(f"{split_name} labels must be one-dimensional")
        if not np.issubdtype(arrays[y_key].dtype, np.integer):
            raise ValueError(f"{split_name} labels must use an integer dtype")
        if len(arrays[x_key]) != len(arrays[y_key]):
            raise ValueError(f"{split_name} feature and label rows differ")
    y_train, y_val, y_test, class_names = _encode_labels_from_training(
        arrays["y_train"], arrays["y_val"], arrays["y_test"]
    )
    return DatasetSplits(
        name=name or source.stem,
        train=Split(arrays["x_train"], y_train),
        validation=Split(arrays["x_val"], y_val),
        test=Split(arrays["x_test"], y_test),
        class_names=class_names,
    )


def _stratified_indices(
    labels: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = ([], [], [])
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        if len(indices) < 3:
            raise ValueError("each class needs at least three rows for three splits")
        train_end = min(
            max(1, int(round(len(indices) * train_fraction))), len(indices) - 2
        )
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_end = min(train_end + validation_count, len(indices) - 1)
        groups[0].extend(indices[:train_end])
        groups[1].extend(indices[train_end:validation_end])
        groups[2].extend(indices[validation_end:])
    result = []
    for group in groups:
        values = np.asarray(group, dtype=np.int64)
        rng.shuffle(values)
        result.append(values)
    return tuple(result)  # type: ignore[return-value]


def make_synthetic(config: DataConfig, *, seed: int) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, config.num_classes, size=config.num_samples)
    features = rng.normal(size=(config.num_samples, config.num_features))
    means = rng.normal(size=(config.num_classes, config.num_relevant_features))
    means -= means.mean(axis=0, keepdims=True)
    features[:, : config.num_relevant_features] += (
        config.class_separation * means[labels]
    )
    if config.label_noise > 0:
        noisy = rng.random(config.num_samples) < config.label_noise
        offsets = rng.integers(1, config.num_classes, size=int(noisy.sum()))
        labels[noisy] = (labels[noisy] + offsets) % config.num_classes

    train_idx, val_idx, test_idx = _stratified_indices(
        labels, config.train_fraction, config.validation_fraction, rng
    )
    # Fit preprocessing statistics on training rows only.
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = ((features - mean) / scale).astype(np.float32)
    labels = labels.astype(np.int64)
    class_names = tuple(str(index) for index in range(config.num_classes))
    return DatasetSplits(
        name=config.name,
        train=Split(features[train_idx], labels[train_idx]),
        validation=Split(features[val_idx], labels[val_idx]),
        test=Split(features[test_idx], labels[test_idx]),
        class_names=class_names,
    )


def load_dataset(
    config: DataConfig, *, seed: int, config_directory: Path | None = None
) -> DatasetSplits:
    if config.kind == "synthetic":
        return make_synthetic(config, seed=seed)
    assert config.path is not None
    path = Path(config.path).expanduser()
    if not path.is_absolute() and config_directory is not None:
        path = config_directory / path
    return load_npz(path, name=config.name)


def dataset_fingerprint(dataset: DatasetSplits) -> str:
    """Hash the exact arrays and label vocabulary consumed by one run."""

    digest = hashlib.sha256()
    digest.update(json.dumps(dataset.class_names).encode("utf-8"))
    for split_name, split in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        digest.update(split_name.encode("ascii"))
        for array in (split.x, split.y):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
