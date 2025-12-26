#!/usr/bin/env python3
"""
Pikkit Feature Definitions
Define all features with metadata for the feature store
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class FeatureType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    EMBEDDING = "embedding"


class ComputationMode(str, Enum):
    BATCH = "batch"       # Computed offline during training
    STREAMING = "streaming"  # Computed in real-time
    HYBRID = "hybrid"     # Precomputed base, updated incrementally


@dataclass
class FeatureDefinition:
    """Define a feature with its metadata and computation logic"""
    name: str
    feature_type: FeatureType
    computation_mode: ComputationMode
    description: str
    entity_key: str  # 'bet_id', 'sport_market', 'user', etc.
    dependencies: List[str] = None
    ttl_seconds: Optional[int] = None  # For cached features
    version: str = "1.0.0"

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


# Feature Registry - Single source of truth for all features
FEATURE_REGISTRY = {
    # === BET-LEVEL FEATURES (Streaming) ===
    'implied_prob': FeatureDefinition(
        name='implied_prob',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Implied probability from American odds',
        entity_key='bet_id',
    ),
    'clv_percentage': FeatureDefinition(
        name='clv_percentage',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Closing Line Value percentage',
        entity_key='bet_id',
    ),
    'clv_ev': FeatureDefinition(
        name='clv_ev',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='CLV * implied_prob (expected value contribution)',
        entity_key='bet_id',
        dependencies=['clv_percentage', 'implied_prob'],
    ),
    'has_clv': FeatureDefinition(
        name='has_clv',
        feature_type=FeatureType.BOOLEAN,
        computation_mode=ComputationMode.STREAMING,
        description='Whether CLV data is available',
        entity_key='bet_id',
    ),
    'is_live': FeatureDefinition(
        name='is_live',
        feature_type=FeatureType.BOOLEAN,
        computation_mode=ComputationMode.STREAMING,
        description='Whether bet was placed live',
        entity_key='bet_id',
    ),

    # === AGGREGATED FEATURES (Batch - Offline) ===
    'sport_win_rate': FeatureDefinition(
        name='sport_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for this sport',
        entity_key='sport',
        ttl_seconds=86400,  # 24 hours
    ),
    'sport_roi': FeatureDefinition(
        name='sport_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for this sport',
        entity_key='sport',
        ttl_seconds=86400,
    ),
    'sport_market_win_rate': FeatureDefinition(
        name='sport_market_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for sport+market combination',
        entity_key='sport_market',
        ttl_seconds=86400,
    ),
    'sport_market_roi': FeatureDefinition(
        name='sport_market_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for sport+market combination',
        entity_key='sport_market',
        ttl_seconds=86400,
    ),
    'sport_league_win_rate': FeatureDefinition(
        name='sport_league_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for sport+league combination',
        entity_key='sport_league',
        ttl_seconds=86400,
    ),
    'sport_league_roi': FeatureDefinition(
        name='sport_league_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for sport+league combination',
        entity_key='sport_league',
        ttl_seconds=86400,
    ),
    'sport_league_market_win_rate': FeatureDefinition(
        name='sport_league_market_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for sport+league+market',
        entity_key='sport_league_market',
        ttl_seconds=86400,
    ),
    'sport_league_market_roi': FeatureDefinition(
        name='sport_league_market_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for sport+league+market',
        entity_key='sport_league_market',
        ttl_seconds=86400,
    ),
    'institution_name_win_rate': FeatureDefinition(
        name='institution_name_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical win rate for this sportsbook',
        entity_key='institution_name',
        ttl_seconds=86400,
    ),
    'institution_name_roi': FeatureDefinition(
        name='institution_name_roi',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.BATCH,
        description='Historical ROI for this sportsbook',
        entity_key='institution_name',
        ttl_seconds=86400,
    ),

    # === HYBRID FEATURES (Precomputed + Updated) ===
    'recent_win_rate': FeatureDefinition(
        name='recent_win_rate',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.HYBRID,
        description='Win rate over last 10 bets in same sport+market',
        entity_key='sport_market',
        ttl_seconds=3600,  # 1 hour
    ),
    'sport_market_count': FeatureDefinition(
        name='sport_market_count',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.HYBRID,
        description='Number of historical bets in sport+market',
        entity_key='sport_market',
        ttl_seconds=3600,
    ),
    'institution_name_count': FeatureDefinition(
        name='institution_name_count',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.HYBRID,
        description='Number of historical bets with this sportsbook',
        entity_key='institution_name',
        ttl_seconds=3600,
    ),

    # === TEMPORAL FEATURES (Streaming) ===
    'day_of_week': FeatureDefinition(
        name='day_of_week',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Day of week (0=Monday, 6=Sunday)',
        entity_key='bet_id',
    ),
    'hour_of_day': FeatureDefinition(
        name='hour_of_day',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Hour of day (0-23)',
        entity_key='bet_id',
    ),
    'days_since_first_bet': FeatureDefinition(
        name='days_since_first_bet',
        feature_type=FeatureType.NUMERICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Days since first bet in dataset',
        entity_key='bet_id',
    ),

    # === ENCODED FEATURES (Streaming) ===
    'sport_encoded': FeatureDefinition(
        name='sport_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded sport',
        entity_key='bet_id',
    ),
    'league_encoded': FeatureDefinition(
        name='league_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded league',
        entity_key='bet_id',
    ),
    'market_encoded': FeatureDefinition(
        name='market_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded market type',
        entity_key='bet_id',
    ),
    'institution_name_encoded': FeatureDefinition(
        name='institution_name_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded sportsbook',
        entity_key='bet_id',
    ),
    'bet_type_encoded': FeatureDefinition(
        name='bet_type_encoded',
        feature_type=FeatureType.CATEGORICAL,
        computation_mode=ComputationMode.STREAMING,
        description='Label-encoded bet type',
        entity_key='bet_id',
    ),
}


# Feature groups for different use cases
TRAINING_FEATURES = [
    # Categorical (encoded)
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

INFERENCE_FEATURES = TRAINING_FEATURES  # Same for now

OFFLINE_FEATURES = [
    name for name, defn in FEATURE_REGISTRY.items()
    if defn.computation_mode == ComputationMode.BATCH
]

ONLINE_FEATURES = [
    name for name, defn in FEATURE_REGISTRY.items()
    if defn.computation_mode == ComputationMode.STREAMING
]


def get_feature_info(feature_name: str) -> Optional[FeatureDefinition]:
    """Get feature definition by name"""
    return FEATURE_REGISTRY.get(feature_name)


def list_features_by_mode(mode: ComputationMode) -> List[str]:
    """List features by computation mode"""
    return [
        name for name, defn in FEATURE_REGISTRY.items()
        if defn.computation_mode == mode
    ]


def list_features_by_entity(entity_key: str) -> List[str]:
    """List features by entity key"""
    return [
        name for name, defn in FEATURE_REGISTRY.items()
        if defn.entity_key == entity_key
    ]


if __name__ == '__main__':
    print("Feature Registry Summary")
    print("=" * 60)

    print(f"\nTotal features: {len(FEATURE_REGISTRY)}")

    print("\nBy Computation Mode:")
    for mode in ComputationMode:
        features = list_features_by_mode(mode)
        print(f"  {mode.value}: {len(features)} features")

    print("\nTraining Features:")
    for f in TRAINING_FEATURES:
        defn = get_feature_info(f)
        if defn:
            print(f"  {f}: {defn.feature_type.value} ({defn.computation_mode.value})")

    print("\nOffline Features (need pre-computation):")
    for f in OFFLINE_FEATURES:
        print(f"  - {f}")
