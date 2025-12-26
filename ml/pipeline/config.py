#!/usr/bin/env python3
"""
Configuration management for ML pipeline
Handles YAML configuration loading and validation
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data source configuration"""
    supabase_url: str = ""
    supabase_key: str = ""
    fetch_limit: int = 50000
    batch_size: int = 1000
    min_bets_per_market: int = 30
    filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Load from environment if not set
        if not self.supabase_url:
            self.supabase_url = os.environ.get(
                'SUPABASE_URL',
                'https://mnnjjvbaxzumfcgibtme.supabase.co'
            )
        if not self.supabase_key:
            self.supabase_key = os.environ.get('SUPABASE_SERVICE_KEY', '')


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    categorical: List[str] = field(default_factory=lambda: [
        'sport', 'league', 'market', 'institution_name', 'bet_type'
    ])
    numerical: List[str] = field(default_factory=lambda: [
        'implied_prob', 'clv_percentage', 'clv_ev', 'is_live'
    ])
    historical_windows: List[Dict] = field(default_factory=list)
    recent_window_size: int = 10
    selection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration"""
    type: str = "xgboost_classifier"
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 20

    def get_xgb_params(self) -> Dict[str, Any]:
        """Get parameters in XGBoost format"""
        params = {
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'random_state': 42,
            'tree_method': 'hist',
            **self.hyperparameters
        }
        return params


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration"""
    method: str = "time_series"
    n_splits: int = 5
    embargo_days: int = 7
    min_train_size: float = 0.5
    test_size: float = 0.2


@dataclass
class SampleWeightingConfig:
    """Sample weighting configuration"""
    strategy: str = "recency"
    recency: Dict[str, float] = field(default_factory=lambda: {
        'decay_factor': 1.0,
        'min_weight': 0.1
    })
    outcome_balanced: Dict[str, float] = field(default_factory=lambda: {
        'minority_weight': 1.5
    })
    clv_based: Dict[str, float] = field(default_factory=lambda: {
        'with_clv_weight': 1.5,
        'without_clv_weight': 1.0
    })


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration"""
    enabled: bool = True
    n_trials: int = 50
    timeout_per_trial: int = 300
    pruning: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'patience': 10
    })
    win_classifier_search_space: Dict[str, List] = field(default_factory=dict)
    roi_regressor_search_space: Dict[str, List] = field(default_factory=dict)
    optimize_metric: str = "val_auc"


@dataclass
class ExperimentTrackingConfig:
    """MLflow experiment tracking configuration"""
    enabled: bool = True
    tracking_uri: str = "/root/pikkit/ml/experiments/mlruns"
    experiment_name: str = "pikkit-betting-models"
    log_artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelRegistryConfig:
    """Model registry configuration"""
    path: str = "/root/pikkit/ml/registry"
    stages: List[str] = field(default_factory=lambda: [
        'development', 'staging', 'production', 'archived'
    ])
    promotion_thresholds: Dict[str, Dict] = field(default_factory=dict)
    retention: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathsConfig:
    """Output paths configuration"""
    models: str = "/root/pikkit/ml/models"
    predictions: str = "/root/pikkit/ml/predictions"
    data: str = "/root/pikkit/ml/data"
    experiments: str = "/root/pikkit/ml/experiments"
    registry: str = "/root/pikkit/ml/registry"
    logs: str = "/var/log/pikkit-ml"


class PipelineConfig:
    """
    Main configuration class for the ML pipeline.
    Loads and validates YAML configuration files.
    """

    DEFAULT_CONFIG_PATH = "/root/pikkit/ml/config/training_config.yaml"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._raw_config = {}

        # Initialize sub-configs with defaults
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.win_classifier = ModelConfig()
        self.roi_regressor = ModelConfig(
            type="xgboost_regressor",
            objective="reg:squarederror",
            eval_metric="mae"
        )
        self.cross_validation = CrossValidationConfig()
        self.sample_weighting = SampleWeightingConfig()
        self.optimization = OptimizationConfig()
        self.experiment_tracking = ExperimentTrackingConfig()
        self.model_registry = ModelRegistryConfig()
        self.paths = PathsConfig()

        # Load from file if exists
        if Path(self.config_path).exists():
            self.load(self.config_path)
        else:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")

    def load(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)

        # Parse sections
        self._parse_data_config()
        self._parse_feature_config()
        self._parse_model_configs()
        self._parse_cv_config()
        self._parse_sample_weighting_config()
        self._parse_optimization_config()
        self._parse_experiment_tracking_config()
        self._parse_registry_config()
        self._parse_paths_config()

        # Ensure directories exist
        self._create_directories()

        logger.info("Configuration loaded successfully")

    def _parse_data_config(self) -> None:
        """Parse data configuration section"""
        data_cfg = self._raw_config.get('data', {})
        self.data = DataConfig(
            supabase_url=data_cfg.get('supabase_url') or "",
            supabase_key=data_cfg.get('supabase_key') or "",
            fetch_limit=data_cfg.get('fetch_limit', 50000),
            batch_size=data_cfg.get('batch_size', 1000),
            min_bets_per_market=data_cfg.get('min_bets_per_market', 30),
            filters=data_cfg.get('filters', {})
        )

    def _parse_feature_config(self) -> None:
        """Parse feature configuration section"""
        feat_cfg = self._raw_config.get('features', {})
        self.features = FeatureConfig(
            categorical=feat_cfg.get('categorical', self.features.categorical),
            numerical=feat_cfg.get('numerical', self.features.numerical),
            historical_windows=feat_cfg.get('historical_windows', []),
            recent_window_size=feat_cfg.get('recent_window_size', 10),
            selection=feat_cfg.get('selection', {})
        )

    def _parse_model_configs(self) -> None:
        """Parse model configuration sections"""
        models_cfg = self._raw_config.get('models', {})

        # Win classifier
        win_cfg = models_cfg.get('win_classifier', {})
        self.win_classifier = ModelConfig(
            type=win_cfg.get('type', 'xgboost_classifier'),
            objective=win_cfg.get('objective', 'binary:logistic'),
            eval_metric=win_cfg.get('eval_metric', 'logloss'),
            hyperparameters=win_cfg.get('hyperparameters', {}),
            early_stopping_rounds=win_cfg.get('early_stopping_rounds', 20)
        )

        # ROI regressor
        roi_cfg = models_cfg.get('roi_regressor', {})
        self.roi_regressor = ModelConfig(
            type=roi_cfg.get('type', 'xgboost_regressor'),
            objective=roi_cfg.get('objective', 'reg:squarederror'),
            eval_metric=roi_cfg.get('eval_metric', 'mae'),
            hyperparameters=roi_cfg.get('hyperparameters', {}),
            early_stopping_rounds=roi_cfg.get('early_stopping_rounds', 20)
        )

    def _parse_cv_config(self) -> None:
        """Parse cross-validation configuration"""
        cv_cfg = self._raw_config.get('cross_validation', {})
        self.cross_validation = CrossValidationConfig(
            method=cv_cfg.get('method', 'time_series'),
            n_splits=cv_cfg.get('n_splits', 5),
            embargo_days=cv_cfg.get('embargo_days', 7),
            min_train_size=cv_cfg.get('min_train_size', 0.5),
            test_size=cv_cfg.get('test_size', 0.2)
        )

    def _parse_sample_weighting_config(self) -> None:
        """Parse sample weighting configuration"""
        sw_cfg = self._raw_config.get('sample_weighting', {})
        self.sample_weighting = SampleWeightingConfig(
            strategy=sw_cfg.get('strategy', 'recency'),
            recency=sw_cfg.get('recency', self.sample_weighting.recency),
            outcome_balanced=sw_cfg.get('outcome_balanced',
                                        self.sample_weighting.outcome_balanced),
            clv_based=sw_cfg.get('clv_based', self.sample_weighting.clv_based)
        )

    def _parse_optimization_config(self) -> None:
        """Parse optimization configuration"""
        opt_cfg = self._raw_config.get('optimization', {})
        self.optimization = OptimizationConfig(
            enabled=opt_cfg.get('enabled', True),
            n_trials=opt_cfg.get('n_trials', 50),
            timeout_per_trial=opt_cfg.get('timeout_per_trial', 300),
            pruning=opt_cfg.get('pruning', {'enabled': True, 'patience': 10}),
            win_classifier_search_space=opt_cfg.get(
                'win_classifier_search_space', {}),
            roi_regressor_search_space=opt_cfg.get(
                'roi_regressor_search_space', {}),
            optimize_metric=opt_cfg.get('optimize_metric', 'val_auc')
        )

    def _parse_experiment_tracking_config(self) -> None:
        """Parse experiment tracking configuration"""
        et_cfg = self._raw_config.get('experiment_tracking', {})
        self.experiment_tracking = ExperimentTrackingConfig(
            enabled=et_cfg.get('enabled', True),
            tracking_uri=et_cfg.get('tracking_uri',
                                    "/root/pikkit/ml/experiments/mlruns"),
            experiment_name=et_cfg.get('experiment_name',
                                       'pikkit-betting-models'),
            log_artifacts=et_cfg.get('log_artifacts', []),
            tags=et_cfg.get('tags', {})
        )

    def _parse_registry_config(self) -> None:
        """Parse model registry configuration"""
        reg_cfg = self._raw_config.get('model_registry', {})
        self.model_registry = ModelRegistryConfig(
            path=reg_cfg.get('path', '/root/pikkit/ml/registry'),
            stages=reg_cfg.get('stages', self.model_registry.stages),
            promotion_thresholds=reg_cfg.get('promotion_thresholds', {}),
            retention=reg_cfg.get('retention', {})
        )

    def _parse_paths_config(self) -> None:
        """Parse paths configuration"""
        paths_cfg = self._raw_config.get('paths', {})
        self.paths = PathsConfig(
            models=paths_cfg.get('models', self.paths.models),
            predictions=paths_cfg.get('predictions', self.paths.predictions),
            data=paths_cfg.get('data', self.paths.data),
            experiments=paths_cfg.get('experiments', self.paths.experiments),
            registry=paths_cfg.get('registry', self.paths.registry),
            logs=paths_cfg.get('logs', self.paths.logs)
        )

    def _create_directories(self) -> None:
        """Create output directories if they don't exist"""
        for path in [
            self.paths.models,
            self.paths.predictions,
            self.paths.data,
            self.paths.experiments,
            self.paths.registry,
            self.paths.logs
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def save(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file"""
        path = config_path or self.config_path

        config_dict = {
            'data': {
                'fetch_limit': self.data.fetch_limit,
                'batch_size': self.data.batch_size,
                'min_bets_per_market': self.data.min_bets_per_market,
                'filters': self.data.filters,
            },
            'features': {
                'categorical': self.features.categorical,
                'numerical': self.features.numerical,
                'historical_windows': self.features.historical_windows,
                'recent_window_size': self.features.recent_window_size,
                'selection': self.features.selection,
            },
            'models': {
                'win_classifier': {
                    'type': self.win_classifier.type,
                    'objective': self.win_classifier.objective,
                    'eval_metric': self.win_classifier.eval_metric,
                    'hyperparameters': self.win_classifier.hyperparameters,
                    'early_stopping_rounds': self.win_classifier.early_stopping_rounds,
                },
                'roi_regressor': {
                    'type': self.roi_regressor.type,
                    'objective': self.roi_regressor.objective,
                    'eval_metric': self.roi_regressor.eval_metric,
                    'hyperparameters': self.roi_regressor.hyperparameters,
                    'early_stopping_rounds': self.roi_regressor.early_stopping_rounds,
                },
            },
            'cross_validation': {
                'method': self.cross_validation.method,
                'n_splits': self.cross_validation.n_splits,
                'embargo_days': self.cross_validation.embargo_days,
                'min_train_size': self.cross_validation.min_train_size,
                'test_size': self.cross_validation.test_size,
            },
            'sample_weighting': {
                'strategy': self.sample_weighting.strategy,
                'recency': self.sample_weighting.recency,
                'outcome_balanced': self.sample_weighting.outcome_balanced,
                'clv_based': self.sample_weighting.clv_based,
            },
            'optimization': {
                'enabled': self.optimization.enabled,
                'n_trials': self.optimization.n_trials,
                'timeout_per_trial': self.optimization.timeout_per_trial,
                'pruning': self.optimization.pruning,
                'win_classifier_search_space': self.optimization.win_classifier_search_space,
                'roi_regressor_search_space': self.optimization.roi_regressor_search_space,
                'optimize_metric': self.optimization.optimize_metric,
            },
            'experiment_tracking': {
                'enabled': self.experiment_tracking.enabled,
                'tracking_uri': self.experiment_tracking.tracking_uri,
                'experiment_name': self.experiment_tracking.experiment_name,
                'log_artifacts': self.experiment_tracking.log_artifacts,
                'tags': self.experiment_tracking.tags,
            },
            'model_registry': {
                'path': self.model_registry.path,
                'stages': self.model_registry.stages,
                'promotion_thresholds': self.model_registry.promotion_thresholds,
                'retention': self.model_registry.retention,
            },
            'paths': {
                'models': self.paths.models,
                'predictions': self.paths.predictions,
                'data': self.paths.data,
                'experiments': self.paths.experiments,
                'registry': self.paths.registry,
                'logs': self.paths.logs,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  config_path='{self.config_path}',\n"
            f"  data_limit={self.data.fetch_limit},\n"
            f"  cv_method='{self.cross_validation.method}',\n"
            f"  optimization_enabled={self.optimization.enabled},\n"
            f"  experiment_tracking={self.experiment_tracking.enabled}\n"
            f")"
        )
