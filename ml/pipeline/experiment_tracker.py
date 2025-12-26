#!/usr/bin/env python3
"""
MLflow Experiment Tracking for Pikkit ML Pipeline

Provides comprehensive experiment tracking including:
- Metric logging (accuracy, AUC, MAE, calibration)
- Artifact management (models, plots, feature importance)
- Experiment comparison and analysis
"""

import os
import json
import pickle
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .config import PipelineConfig, ExperimentTrackingConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracking for ML training runs.

    Features:
    - Automatic metric logging
    - Model artifact storage
    - Feature importance visualization
    - Calibration plot generation
    - Run comparison and analysis
    """

    def __init__(self, config: Optional[Union[PipelineConfig, ExperimentTrackingConfig]] = None):
        """
        Initialize experiment tracker.

        Args:
            config: Pipeline or experiment tracking configuration
        """
        if isinstance(config, PipelineConfig):
            self.config = config.experiment_tracking
            self.paths = config.paths
        elif isinstance(config, ExperimentTrackingConfig):
            self.config = config
            self.paths = None
        else:
            self.config = ExperimentTrackingConfig()
            self.paths = None

        self.enabled = self.config.enabled and MLFLOW_AVAILABLE
        self._active_run = None
        self._client = None

        if self.enabled:
            self._setup_mlflow()
        elif not MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed. Experiment tracking disabled.")

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking"""
        # Set tracking URI
        tracking_uri = self.config.tracking_uri
        if tracking_uri.startswith('/'):
            # Local file path - ensure directory exists
            Path(tracking_uri).mkdir(parents=True, exist_ok=True)
            tracking_uri = f"file://{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        # Create or get experiment
        experiment_name = self.config.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"{self.config.tracking_uri}/artifacts/{experiment_name}"
            )
            logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient()

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new experiment run.

        Args:
            run_name: Name for the run (auto-generated if not provided)
            tags: Additional tags for the run

        Returns:
            Run ID
        """
        if not self.enabled:
            return "disabled"

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"training_run_{timestamp}"

        # Merge tags
        all_tags = {**self.config.tags, **(tags or {})}
        all_tags['run_name'] = run_name
        all_tags['timestamp'] = datetime.now().isoformat()

        # Start MLflow run
        self._active_run = mlflow.start_run(run_name=run_name, tags=all_tags)
        run_id = self._active_run.info.run_id

        logger.info(f"Started experiment run: {run_name} (ID: {run_id})")
        return run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current experiment run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.enabled or self._active_run is None:
            return

        mlflow.end_run(status=status)
        logger.info(f"Ended experiment run with status: {status}")
        self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled:
            return

        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)

        # MLflow has limits on param value length
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 250:
                str_value = str_value[:247] + "..."
            mlflow.log_param(key, str_value)

        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metric
        """
        if not self.enabled:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(name, value, step=step)

        logger.debug(f"Logged {len(metrics)} metrics at step {step}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a trained model as an artifact.

        Args:
            model: Trained model object
            model_name: Name for the model artifact
            model_type: Model type (sklearn, xgboost, etc.)
            metadata: Optional metadata dictionary

        Returns:
            Artifact URI
        """
        if not self.enabled:
            return ""

        # Log model based on type
        if model_type == "xgboost":
            artifact_path = mlflow.xgboost.log_model(
                model,
                model_name,
                registered_model_name=None  # Don't auto-register
            )
        else:
            artifact_path = mlflow.sklearn.log_model(
                model,
                model_name
            )

        # Log metadata as JSON artifact
        if metadata:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                json.dump(metadata, f, indent=2, default=str)
                temp_path = f.name

            mlflow.log_artifact(temp_path, f"{model_name}_metadata")
            os.unlink(temp_path)

        logger.info(f"Logged model: {model_name}")
        return artifact_path.model_uri if hasattr(artifact_path, 'model_uri') else ""

    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        model_name: str = "model",
        top_k: int = 20
    ) -> None:
        """
        Log feature importance as table and visualization.

        Args:
            feature_names: List of feature names
            importance_values: Array of importance values
            model_name: Name prefix for artifacts
            top_k: Number of top features to visualize
        """
        if not self.enabled:
            return

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        # Log as CSV
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            importance_df.to_csv(f, index=False)
            temp_csv = f.name
        mlflow.log_artifact(temp_csv, f"{model_name}_feature_importance")
        os.unlink(temp_csv)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(top_k)

        ax.barh(
            range(len(top_features)),
            top_features['importance'].values,
            color='steelblue'
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Feature Importance - {model_name}')
        plt.tight_layout()

        # Save and log plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_png = f.name
        fig.savefig(temp_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(temp_png, f"{model_name}_plots")
        os.unlink(temp_png)

        logger.info(f"Logged feature importance for {model_name}")

    def log_calibration_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
        n_bins: int = 10
    ) -> None:
        """
        Log calibration plot for probability predictions.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name prefix for artifacts
            n_bins: Number of calibration bins
        """
        if not self.enabled:
            return

        from sklearn.calibration import calibration_curve

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

        # Model calibration
        ax.plot(prob_pred, prob_true, 's-', label=model_name)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Plot - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save and log
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_png = f.name
        fig.savefig(temp_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(temp_png, f"{model_name}_plots")
        os.unlink(temp_png)

        # Log calibration metrics
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        mlflow.log_metric(f"{model_name}_calibration_error", calibration_error)

        logger.info(f"Logged calibration plot for {model_name}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Log confusion matrix as visualization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name prefix for artifacts
            labels: Class labels
        """
        if not self.enabled:
            return

        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        if labels is None:
            labels = ['Loss', 'Win']

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            ylabel='True label',
            xlabel='Predicted label',
            title=f'Confusion Matrix - {model_name}'
        )

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        # Save and log
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_png = f.name
        fig.savefig(temp_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        mlflow.log_artifact(temp_png, f"{model_name}_plots")
        os.unlink(temp_png)

        logger.info(f"Logged confusion matrix for {model_name}")

    def log_predictions(
        self,
        predictions_df: pd.DataFrame,
        artifact_name: str = "predictions"
    ) -> None:
        """
        Log predictions DataFrame as artifact.

        Args:
            predictions_df: DataFrame with predictions
            artifact_name: Name for the artifact
        """
        if not self.enabled:
            return

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            predictions_df.to_json(f, orient='records', indent=2)
            temp_path = f.name

        mlflow.log_artifact(temp_path, artifact_name)
        os.unlink(temp_path)

        logger.info(f"Logged predictions: {len(predictions_df)} records")

    def log_cv_results(
        self,
        cv_results: List[Dict[str, float]],
        model_name: str = "model"
    ) -> None:
        """
        Log cross-validation results.

        Args:
            cv_results: List of dictionaries with metrics per fold
            model_name: Name prefix for metrics
        """
        if not self.enabled:
            return

        # Calculate aggregate statistics
        metrics_df = pd.DataFrame(cv_results)

        for metric_name in metrics_df.columns:
            if metric_name.startswith('fold'):
                continue

            values = metrics_df[metric_name].values
            mlflow.log_metric(f"{model_name}_cv_{metric_name}_mean", np.mean(values))
            mlflow.log_metric(f"{model_name}_cv_{metric_name}_std", np.std(values))
            mlflow.log_metric(f"{model_name}_cv_{metric_name}_min", np.min(values))
            mlflow.log_metric(f"{model_name}_cv_{metric_name}_max", np.max(values))

        # Log detailed results
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(cv_results, f, indent=2)
            temp_path = f.name

        mlflow.log_artifact(temp_path, f"{model_name}_cv_results")
        os.unlink(temp_path)

        logger.info(f"Logged CV results for {model_name}")

    def get_best_run(self, metric: str = "val_auc", ascending: bool = False) -> Optional[Dict]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric to optimize
            ascending: If True, lower is better

        Returns:
            Dictionary with best run info
        """
        if not self.enabled:
            return None

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if runs.empty:
            return None

        best_run = runs.iloc[0].to_dict()
        return {
            'run_id': best_run['run_id'],
            'metric_value': best_run.get(f'metrics.{metric}'),
            'params': {k.replace('params.', ''): v
                       for k, v in best_run.items() if k.startswith('params.')},
            'metrics': {k.replace('metrics.', ''): v
                        for k, v in best_run.items() if k.startswith('metrics.')},
        }

    def compare_runs(
        self,
        run_ids: Optional[List[str]] = None,
        n_runs: int = 5,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiment runs.

        Args:
            run_ids: Specific run IDs to compare
            n_runs: Number of recent runs if run_ids not provided
            metrics: Specific metrics to compare

        Returns:
            DataFrame with run comparison
        """
        if not self.enabled:
            return pd.DataFrame()

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            return pd.DataFrame()

        if run_ids:
            filter_string = " OR ".join([f"run_id = '{rid}'" for rid in run_ids])
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string
            )
        else:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n_runs
            )

        if runs.empty:
            return pd.DataFrame()

        # Select columns to display
        display_cols = ['run_id', 'start_time', 'status']

        if metrics:
            metric_cols = [f'metrics.{m}' for m in metrics
                          if f'metrics.{m}' in runs.columns]
            display_cols.extend(metric_cols)
        else:
            # Include all metric columns
            metric_cols = [c for c in runs.columns if c.startswith('metrics.')]
            display_cols.extend(metric_cols)

        return runs[display_cols]

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __enter__(self):
        """Context manager entry"""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
        return False
