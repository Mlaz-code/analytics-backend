"""
Pikkit ML Pipeline - Enhanced Training System

This package provides a production-ready ML training pipeline with:
- Experiment tracking (MLflow)
- Hyperparameter optimization (Optuna)
- Model registry with version control
- Time-series cross-validation
- Configurable sample weighting
"""

from .config import PipelineConfig
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .trainer import ModelTrainer
from .optimizer import HyperparameterOptimizer
from .registry import ModelRegistry
from .experiment_tracker import ExperimentTracker

__version__ = "2.0.0"

__all__ = [
    "PipelineConfig",
    "DataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "HyperparameterOptimizer",
    "ModelRegistry",
    "ExperimentTracker",
]
