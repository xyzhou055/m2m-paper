"""Focused, standalone implementation of M2M-AGDA."""

from .classifiers import (
    SUPPORTED_CLASSIFIERS,
    ClassifierSpec,
    FTTransformerClassifier,
    LogisticRegressionClassifier,
    MLPClassifier,
    build_classifier,
)
from .config import AGDAConfig, RunConfig, apply_accuracy_margin
from .data import (
    DatasetSplits,
    Split,
    dataset_fingerprint,
    load_dataset,
    load_npz,
    make_synthetic,
)
from .model import M2MAGDAModel, ModelSpec
from .trainer import (
    AGDATrainer,
    ModelFactory,
    TrainingResult,
    build_model,
    load_checkpoint,
    save_checkpoint,
    train_and_evaluate,
    transform_numpy,
)

__all__ = [
    "AGDAConfig",
    "AGDATrainer",
    "ClassifierSpec",
    "DatasetSplits",
    "FTTransformerClassifier",
    "LogisticRegressionClassifier",
    "MLPClassifier",
    "M2MAGDAModel",
    "ModelSpec",
    "ModelFactory",
    "RunConfig",
    "Split",
    "SUPPORTED_CLASSIFIERS",
    "TrainingResult",
    "apply_accuracy_margin",
    "dataset_fingerprint",
    "build_classifier",
    "build_model",
    "load_checkpoint",
    "load_dataset",
    "load_npz",
    "make_synthetic",
    "save_checkpoint",
    "train_and_evaluate",
    "transform_numpy",
]

__version__ = "0.1.0"
