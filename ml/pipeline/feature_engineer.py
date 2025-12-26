#!/usr/bin/env python3
"""
Feature Engineering for Pikkit ML Pipeline

Handles:
- Categorical encoding
- Historical performance features
- Temporal features
- CLV features
- Sample weighting
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig, FeatureConfig, SampleWeightingConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for betting data.

    Features:
    - Categorical encoding with unknown handling
    - Historical performance aggregations
    - Time-series safe feature computation
    - Multiple sample weighting strategies
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Pipeline configuration
        """
        if config is None:
            config = PipelineConfig()

        self.feature_config = config.features
        self.weighting_config = config.sample_weighting

        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Fit encoders and transform data.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data

        Returns:
            Tuple of (transformed DataFrame, encoders dict)
        """
        logger.info("Starting feature engineering...")
        df = df.copy()

        # Convert dates
        df['created_at'] = pd.to_datetime(df['created_at'])

        # Sort by date for time-based features
        df = df.sort_values('created_at').reset_index(drop=True)

        # Create target variables
        df = self._create_targets(df)

        # Basic features
        df = self._create_odds_features(df)

        # CLV features
        df = self._create_clv_features(df)

        # Categorical encoding
        if is_training:
            df = self._fit_encode_categoricals(df)
        else:
            df = self._transform_categoricals(df)

        # Historical performance features (time-series safe)
        df = self._create_historical_features(df)

        # Temporal features
        df = self._create_temporal_features(df)

        # Select final features
        self.feature_names = self._select_features(df)
        self._fitted = True

        logger.info(f"Feature engineering complete. {len(self.feature_names)} features created.")
        return df, self.encoders

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        return self.fit_transform(df, is_training=False)[0]

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables"""
        # Win/loss indicator
        if 'is_win' in df.columns:
            df['won'] = df['is_win'].fillna(False).astype(int)
        elif 'result' in df.columns:
            df['won'] = (df['result'] == 'Won').astype(int)
        else:
            raise ValueError("No result field found")

        # ROI
        if 'profit' in df.columns and 'roi' in df.columns:
            df['profit'] = df['profit'].fillna(0)
            df['roi'] = df['roi'].fillna(0)
        else:
            wager_col = 'amount' if 'amount' in df.columns else 'wager'
            df['profit'] = df.apply(
                lambda row: row.get('to_win', 0) if row['won'] == 1
                else -row.get(wager_col, 0),
                axis=1
            )
            df['roi'] = (df['profit'] / df[wager_col]) * 100

        return df

    def _create_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create odds-based features"""
        odds_col = 'american_odds' if 'american_odds' in df.columns else 'odds'

        if 'implied_prob' not in df.columns:
            df['implied_prob'] = df[odds_col].apply(
                lambda x: abs(x) / (abs(x) + 100) if x < 0
                else 100 / (x + 100) if pd.notna(x) else 0.5
            )

        # Is live bet
        if 'is_live' in df.columns:
            df['is_live'] = df['is_live'].fillna(False).astype(int)
        else:
            df['is_live'] = 0

        return df

    def _create_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create CLV-based features"""
        df['has_clv'] = df['clv_percentage'].notna().astype(int)
        df['clv_percentage'] = df['clv_percentage'].fillna(0)
        df['clv_ev'] = df['clv_percentage'] * df['implied_prob']

        return df

    def _fit_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and encode categorical features"""
        for col in self.feature_config.categorical:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le

        return df

    def _transform_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders"""
        for col, le in self.encoders.items():
            if col in df.columns:
                # Handle unknown categories
                known_classes = set(le.classes_)
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known_classes else 0
                )

        return df

    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical performance features.

        Uses expanding window to avoid lookahead bias.
        """
        windows = self.feature_config.historical_windows

        if not windows:
            # Default windows
            windows = [
                {'name': 'sport', 'columns': ['sport']},
                {'name': 'sport_market', 'columns': ['sport', 'market']},
                {'name': 'sport_league', 'columns': ['sport', 'league']},
                {'name': 'sport_league_market', 'columns': ['sport', 'league', 'market']},
                {'name': 'institution_name', 'columns': ['institution_name']},
            ]

        for window in windows:
            group_name = window['name']
            group_cols = window['columns']

            # Check columns exist
            if not all(c in df.columns for c in group_cols):
                continue

            # Expanding window win rate (shifted to avoid lookahead)
            df[f'{group_name}_win_rate'] = (
                df.groupby(group_cols)['won']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0.5)  # Prior: 50% win rate
            )

            # Expanding window ROI
            df[f'{group_name}_roi'] = (
                df.groupby(group_cols)['roi']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0)  # Prior: 0% ROI
            )

            # Sample count
            df[f'{group_name}_count'] = df.groupby(group_cols).cumcount()

        # Recent performance (rolling window)
        recent_window = self.feature_config.recent_window_size
        df['recent_win_rate'] = (
            df.groupby(['sport', 'market'])['won']
            .transform(lambda x: x.shift(1).rolling(recent_window, min_periods=1).mean())
            .fillna(0.5)
        )

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['hour_of_day'] = df['created_at'].dt.hour
        df['month'] = df['created_at'].dt.month
        df['days_since_first_bet'] = (
            df['created_at'] - df['created_at'].min()
        ).dt.days

        return df

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features for modeling"""
        features = [
            # Categorical encodings
            'sport_encoded', 'league_encoded', 'market_encoded',
            'institution_name_encoded', 'bet_type_encoded',

            # Bet characteristics
            'implied_prob', 'is_live',

            # CLV features
            'clv_percentage', 'clv_ev', 'has_clv',

            # Historical performance
            'sport_win_rate', 'sport_roi',
            'sport_market_win_rate', 'sport_market_roi',
            'sport_league_win_rate', 'sport_league_roi',
            'sport_league_market_win_rate', 'sport_league_market_roi',
            'institution_name_win_rate', 'institution_name_roi',

            # Sample sizes
            'sport_market_count', 'institution_name_count',

            # Recent trends
            'recent_win_rate',

            # Temporal
            'day_of_week', 'hour_of_day', 'days_since_first_bet',
        ]

        # Filter to features that exist
        features = [f for f in features if f in df.columns]

        return features

    def get_sample_weights(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate sample weights based on configured strategy.

        Args:
            df: DataFrame with samples
            strategy: Weight strategy (uses config if not provided)

        Returns:
            Array of sample weights
        """
        strategy = strategy or self.weighting_config.strategy
        n_samples = len(df)

        if strategy == 'none':
            return np.ones(n_samples)

        elif strategy == 'recency':
            # More weight to recent samples
            decay = self.weighting_config.recency.get('decay_factor', 1.0)
            min_weight = self.weighting_config.recency.get('min_weight', 0.1)

            # Exponential decay from oldest to newest
            weights = np.exp(np.linspace(-decay, 0, n_samples))
            weights = np.maximum(weights, min_weight)
            return weights

        elif strategy == 'outcome_balanced':
            # Balance win/loss classes
            minority_weight = self.weighting_config.outcome_balanced.get(
                'minority_weight', 1.5
            )

            if 'won' not in df.columns:
                return np.ones(n_samples)

            win_rate = df['won'].mean()
            weights = np.ones(n_samples)

            if win_rate < 0.5:
                # Wins are minority
                weights[df['won'] == 1] = minority_weight
            else:
                # Losses are minority
                weights[df['won'] == 0] = minority_weight

            return weights

        elif strategy == 'clv_based':
            # More weight to bets with CLV data
            with_clv_weight = self.weighting_config.clv_based.get(
                'with_clv_weight', 1.5
            )
            without_clv_weight = self.weighting_config.clv_based.get(
                'without_clv_weight', 1.0
            )

            if 'has_clv' not in df.columns:
                return np.ones(n_samples)

            weights = np.where(
                df['has_clv'] == 1,
                with_clv_weight,
                without_clv_weight
            )
            return weights

        elif strategy == 'combined':
            # Combine multiple strategies
            recency_weights = self.get_sample_weights(df, 'recency')
            outcome_weights = self.get_sample_weights(df, 'outcome_balanced')
            clv_weights = self.get_sample_weights(df, 'clv_based')

            # Geometric mean of all weights
            weights = (recency_weights * outcome_weights * clv_weights) ** (1/3)
            return weights

        else:
            logger.warning(f"Unknown weighting strategy: {strategy}")
            return np.ones(n_samples)

    def get_encoder_info(self) -> Dict[str, List[str]]:
        """Get encoder class information for serialization"""
        return {
            col: list(le.classes_)
            for col, le in self.encoders.items()
        }

    def set_encoders(self, encoder_info: Dict[str, List[str]]) -> None:
        """Set encoders from serialized info"""
        for col, classes in encoder_info.items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            self.encoders[col] = le
        self._fitted = True
