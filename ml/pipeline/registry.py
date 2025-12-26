#!/usr/bin/env python3
"""
Model Registry for Pikkit ML Pipeline

Provides model version control with:
- Version management (dev -> staging -> production)
- Model metadata tracking
- Promotion workflow
- Rollback capability
"""

import os
import json
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import hashlib

from .config import PipelineConfig, ModelRegistryConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a single model version"""
    version_id: str
    model_name: str
    stage: str
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    features: List[str]
    data_info: Dict[str, Any]
    artifact_path: str
    description: str = ""
    promoted_from: Optional[str] = None
    promoted_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        return cls(**data)


class ModelRegistry:
    """
    Model registry with version control and promotion workflow.

    Features:
    - Stage-based model management (dev/staging/production)
    - Model metadata and lineage tracking
    - Auto-promotion based on metrics
    - Rollback support
    - Version retention policy
    """

    STAGES = ['development', 'staging', 'production', 'archived']

    def __init__(self, config: Optional[Union[PipelineConfig, ModelRegistryConfig]] = None):
        """
        Initialize model registry.

        Args:
            config: Pipeline or registry configuration
        """
        if isinstance(config, PipelineConfig):
            self.config = config.model_registry
            self.paths = config.paths
        elif isinstance(config, ModelRegistryConfig):
            self.config = config
            self.paths = None
        else:
            self.config = ModelRegistryConfig()
            self.paths = None

        self.registry_path = Path(self.config.path)
        self._ensure_structure()

        # Load registry index
        self._index: Dict[str, List[ModelVersion]] = {}
        self._load_index()

    def _ensure_structure(self) -> None:
        """Ensure registry directory structure exists"""
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Create stage directories
        for stage in self.STAGES:
            (self.registry_path / stage).mkdir(exist_ok=True)

        # Create index file if not exists
        index_path = self.registry_path / 'registry_index.json'
        if not index_path.exists():
            with open(index_path, 'w') as f:
                json.dump({}, f)

    def _load_index(self) -> None:
        """Load registry index from disk"""
        index_path = self.registry_path / 'registry_index.json'

        if index_path.exists():
            with open(index_path, 'r') as f:
                raw_index = json.load(f)

            self._index = {}
            for model_name, versions in raw_index.items():
                self._index[model_name] = [
                    ModelVersion.from_dict(v) for v in versions
                ]
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save registry index to disk"""
        index_path = self.registry_path / 'registry_index.json'

        raw_index = {}
        for model_name, versions in self._index.items():
            raw_index[model_name] = [v.to_dict() for v in versions]

        with open(index_path, 'w') as f:
            json.dump(raw_index, f, indent=2, default=str)

    def _generate_version_id(
        self,
        model_name: str,
        timestamp: str,
        metrics: Dict[str, float]
    ) -> str:
        """Generate unique version ID"""
        hash_input = f"{model_name}_{timestamp}_{json.dumps(metrics, sort_keys=True)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        features: List[str],
        data_info: Optional[Dict[str, Any]] = None,
        description: str = "",
        stage: str = "development"
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'win_classifier', 'roi_regressor')
            metrics: Dictionary of evaluation metrics
            hyperparameters: Model hyperparameters
            features: List of feature names used
            data_info: Information about training data
            description: Optional description
            stage: Initial stage (default: development)

        Returns:
            ModelVersion object for the registered model
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.STAGES}")

        timestamp = datetime.now().isoformat()
        version_id = self._generate_version_id(model_name, timestamp, metrics)

        # Create artifact directory
        artifact_dir = self.registry_path / stage / f"{model_name}_{version_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = artifact_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            stage=stage,
            created_at=timestamp,
            metrics=metrics,
            hyperparameters=hyperparameters,
            features=features,
            data_info=data_info or {},
            artifact_path=str(artifact_dir),
            description=description
        )

        # Save metadata
        metadata_path = artifact_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        # Update index
        if model_name not in self._index:
            self._index[model_name] = []
        self._index[model_name].append(version)
        self._save_index()

        # Apply retention policy
        self._apply_retention_policy(model_name, stage)

        logger.info(f"Registered model {model_name} version {version_id} in {stage}")
        return version

    def get_model(
        self,
        model_name: str,
        version_id: Optional[str] = None,
        stage: str = "production"
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model
            version_id: Specific version ID (uses latest in stage if not provided)
            stage: Stage to load from

        Returns:
            Loaded model object
        """
        version = self.get_version(model_name, version_id, stage)
        if version is None:
            raise ValueError(f"No model found: {model_name} in {stage}")

        model_path = Path(version.artifact_path) / 'model.pkl'
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def get_version(
        self,
        model_name: str,
        version_id: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get version metadata.

        Args:
            model_name: Name of the model
            version_id: Specific version ID
            stage: Filter by stage

        Returns:
            ModelVersion object or None
        """
        if model_name not in self._index:
            return None

        versions = self._index[model_name]

        if version_id:
            for v in versions:
                if v.version_id == version_id:
                    return v
            return None

        if stage:
            stage_versions = [v for v in versions if v.stage == stage]
            if not stage_versions:
                return None
            # Return most recent
            return max(stage_versions, key=lambda v: v.created_at)

        # Return most recent overall
        return max(versions, key=lambda v: v.created_at)

    def list_versions(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        List model versions.

        Args:
            model_name: Filter by model name
            stage: Filter by stage

        Returns:
            List of ModelVersion objects
        """
        versions = []

        if model_name:
            if model_name in self._index:
                versions = self._index[model_name]
        else:
            for model_versions in self._index.values():
                versions.extend(model_versions)

        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def promote_model(
        self,
        model_name: str,
        version_id: str,
        to_stage: str,
        auto_check: bool = True
    ) -> bool:
        """
        Promote a model to a new stage.

        Args:
            model_name: Name of the model
            version_id: Version ID to promote
            to_stage: Target stage
            auto_check: Check promotion thresholds

        Returns:
            True if promotion successful
        """
        if to_stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {to_stage}")

        version = self.get_version(model_name, version_id)
        if version is None:
            raise ValueError(f"Version not found: {version_id}")

        # Check promotion thresholds
        if auto_check and not self._check_promotion_thresholds(version, to_stage):
            logger.warning(f"Model {version_id} does not meet thresholds for {to_stage}")
            return False

        from_stage = version.stage
        old_artifact_path = Path(version.artifact_path)
        new_artifact_dir = self.registry_path / to_stage / f"{model_name}_{version_id}"

        # Move artifacts
        if old_artifact_path != new_artifact_dir:
            shutil.move(str(old_artifact_path), str(new_artifact_dir))

        # Update version record
        version.stage = to_stage
        version.promoted_from = from_stage
        version.promoted_at = datetime.now().isoformat()
        version.artifact_path = str(new_artifact_dir)

        # Update metadata file
        metadata_path = new_artifact_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        self._save_index()

        logger.info(f"Promoted {model_name} {version_id}: {from_stage} -> {to_stage}")
        return True

    def rollback(
        self,
        model_name: str,
        stage: str = "production",
        to_version_id: Optional[str] = None
    ) -> bool:
        """
        Rollback to a previous model version.

        Args:
            model_name: Name of the model
            stage: Stage to rollback
            to_version_id: Specific version to rollback to (uses previous if not provided)

        Returns:
            True if rollback successful
        """
        current = self.get_version(model_name, stage=stage)
        if current is None:
            logger.warning(f"No current {stage} model for {model_name}")
            return False

        if to_version_id:
            # Rollback to specific version
            target = self.get_version(model_name, version_id=to_version_id)
            if target is None:
                raise ValueError(f"Version not found: {to_version_id}")
        else:
            # Find previous version in same stage
            stage_versions = [
                v for v in self._index.get(model_name, [])
                if v.stage == stage and v.version_id != current.version_id
            ]
            if not stage_versions:
                logger.warning(f"No previous {stage} version for {model_name}")
                return False

            target = max(stage_versions, key=lambda v: v.created_at)

        # Archive current version
        self.promote_model(model_name, current.version_id, "archived", auto_check=False)

        # Promote target to stage
        return self.promote_model(model_name, target.version_id, stage, auto_check=False)

    def _check_promotion_thresholds(
        self,
        version: ModelVersion,
        to_stage: str
    ) -> bool:
        """Check if model meets promotion thresholds"""
        thresholds = self.config.promotion_thresholds.get(to_stage, {})

        if not thresholds:
            return True

        # Check AUC threshold
        min_auc = thresholds.get('min_auc')
        if min_auc:
            model_auc = version.metrics.get('val_auc', version.metrics.get('auc', 0))
            if model_auc < min_auc:
                logger.debug(f"AUC {model_auc} below threshold {min_auc}")
                return False

        # Check sample size
        min_samples = thresholds.get('min_samples')
        if min_samples:
            n_samples = version.data_info.get('n_train_samples', 0)
            if n_samples < min_samples:
                logger.debug(f"Samples {n_samples} below threshold {min_samples}")
                return False

        return True

    def _apply_retention_policy(self, model_name: str, stage: str) -> None:
        """Apply retention policy to limit versions per stage"""
        max_versions = self.config.retention.get('max_versions_per_stage', 5)

        stage_versions = [
            v for v in self._index.get(model_name, [])
            if v.stage == stage
        ]

        if len(stage_versions) <= max_versions:
            return

        # Sort by creation time and archive oldest
        sorted_versions = sorted(stage_versions, key=lambda v: v.created_at)
        to_archive = sorted_versions[:-max_versions]

        for version in to_archive:
            self.promote_model(
                model_name,
                version.version_id,
                "archived",
                auto_check=False
            )

    def get_production_models(self) -> Dict[str, ModelVersion]:
        """
        Get all production models.

        Returns:
            Dictionary mapping model names to their production versions
        """
        production_models = {}

        for model_name in self._index:
            version = self.get_version(model_name, stage="production")
            if version:
                production_models[model_name] = version

        return production_models

    def compare_versions(
        self,
        model_name: str,
        version_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple model versions.

        Args:
            model_name: Name of the model
            version_ids: List of version IDs to compare

        Returns:
            Comparison dictionary
        """
        versions = []
        for vid in version_ids:
            v = self.get_version(model_name, version_id=vid)
            if v:
                versions.append(v)

        if not versions:
            return {}

        comparison = {
            'versions': [v.version_id for v in versions],
            'metrics': {},
            'hyperparameters': {}
        }

        # Compare metrics
        all_metrics = set()
        for v in versions:
            all_metrics.update(v.metrics.keys())

        for metric in all_metrics:
            comparison['metrics'][metric] = {
                v.version_id: v.metrics.get(metric)
                for v in versions
            }

        # Compare hyperparameters
        all_params = set()
        for v in versions:
            all_params.update(v.hyperparameters.keys())

        for param in all_params:
            comparison['hyperparameters'][param] = {
                v.version_id: v.hyperparameters.get(param)
                for v in versions
            }

        return comparison

    def export_lineage(self, model_name: str, version_id: str) -> Dict[str, Any]:
        """
        Export model lineage (promotion history).

        Args:
            model_name: Name of the model
            version_id: Version ID

        Returns:
            Lineage dictionary
        """
        version = self.get_version(model_name, version_id)
        if version is None:
            return {}

        lineage = {
            'model_name': model_name,
            'version_id': version_id,
            'current_stage': version.stage,
            'created_at': version.created_at,
            'history': []
        }

        # Build promotion history
        if version.promoted_from:
            lineage['history'].append({
                'from_stage': version.promoted_from,
                'to_stage': version.stage,
                'promoted_at': version.promoted_at
            })

        return lineage

    def delete_version(
        self,
        model_name: str,
        version_id: str,
        force: bool = False
    ) -> bool:
        """
        Delete a model version.

        Args:
            model_name: Name of the model
            version_id: Version ID to delete
            force: Force deletion even if in production

        Returns:
            True if deletion successful
        """
        version = self.get_version(model_name, version_id)
        if version is None:
            return False

        if version.stage == "production" and not force:
            logger.error("Cannot delete production model without force=True")
            return False

        # Remove artifact directory
        artifact_path = Path(version.artifact_path)
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

        # Remove from index
        self._index[model_name] = [
            v for v in self._index[model_name]
            if v.version_id != version_id
        ]
        self._save_index()

        logger.info(f"Deleted model {model_name} version {version_id}")
        return True

    def update_latest_symlinks(self) -> None:
        """
        Update 'latest' symlinks for production models.
        Creates symlinks at /root/pikkit/ml/models/ pointing to production versions.
        """
        if self.paths is None:
            return

        models_dir = Path(self.paths.models)

        for model_name in self._index:
            prod_version = self.get_version(model_name, stage="production")
            if prod_version is None:
                continue

            # Source model path
            src_model = Path(prod_version.artifact_path) / 'model.pkl'
            if not src_model.exists():
                continue

            # Create symlink
            link_name = models_dir / f'{model_name}_latest.pkl'

            if link_name.is_symlink():
                link_name.unlink()

            # Copy instead of symlink for compatibility
            shutil.copy2(src_model, link_name)

            # Also save metadata
            src_metadata = Path(prod_version.artifact_path) / 'metadata.json'
            if src_metadata.exists():
                shutil.copy2(
                    src_metadata,
                    models_dir / f'model_metadata_latest.json'
                )

        logger.info("Updated latest model symlinks")
