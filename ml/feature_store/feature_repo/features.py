#!/usr/bin/env python3
"""
Pikkit Feature Store Definitions (Feast)
Define entities and feature views for sports betting analytics
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64, String


# =============================================================================
# Entities
# =============================================================================

sport_market = Entity(
    name="sport_market_key",
    description="Combination of sport + market (e.g., 'Basketball_Spread')",
    value_type=ValueType.STRING,
)

institution = Entity(
    name="institution_name",
    description="Sportsbook institution name (e.g., 'DraftKings', 'FanDuel')",
    value_type=ValueType.STRING,
)

sport_league = Entity(
    name="sport_league_key",
    description="Combination of sport + league (e.g., 'Basketball_NBA')",
    value_type=ValueType.STRING,
)


# =============================================================================
# Feature Views
# =============================================================================

# Historical Performance Features (Sport + Market level)
historical_performance_source = FileSource(
    name="historical_performance_source",
    path="/root/pikkit/ml/feature_store/data/historical_performance.parquet",
    timestamp_field="event_timestamp",
)

historical_performance = FeatureView(
    name="historical_performance",
    entities=[sport_market],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sport_win_rate", dtype=Float32),
        Field(name="sport_roi", dtype=Float32),
        Field(name="sport_market_win_rate", dtype=Float32),
        Field(name="sport_market_roi", dtype=Float32),
        Field(name="sport_market_count", dtype=Int64),
        Field(name="avg_clv", dtype=Float32),
        Field(name="recent_win_rate_10", dtype=Float32),
        Field(name="recent_roi_10", dtype=Float32),
    ],
    online=True,
    source=historical_performance_source,
    description="Historical betting performance aggregated by sport and market",
    tags={"team": "ml", "category": "performance"},
)


# Institution-Level Features
institution_features_source = FileSource(
    name="institution_features_source",
    path="/root/pikkit/ml/feature_store/data/institution_features.parquet",
    timestamp_field="event_timestamp",
)

institution_features = FeatureView(
    name="institution_features",
    entities=[institution],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="institution_win_rate", dtype=Float32),
        Field(name="institution_roi", dtype=Float32),
        Field(name="institution_count", dtype=Int64),
        Field(name="institution_avg_odds", dtype=Float32),
        Field(name="institution_avg_clv", dtype=Float32),
        Field(name="institution_sharp_pct", dtype=Float32),
    ],
    online=True,
    source=institution_features_source,
    description="Sportsbook-specific performance metrics",
    tags={"team": "ml", "category": "institution"},
)


# League-Level Features
league_features_source = FileSource(
    name="league_features_source",
    path="/root/pikkit/ml/feature_store/data/league_features.parquet",
    timestamp_field="event_timestamp",
)

league_features = FeatureView(
    name="league_features",
    entities=[sport_league],
    ttl=timedelta(hours=12),
    schema=[
        Field(name="league_win_rate", dtype=Float32),
        Field(name="league_roi", dtype=Float32),
        Field(name="league_avg_edge", dtype=Float32),
        Field(name="league_total_bets", dtype=Int64),
        Field(name="league_recent_performance", dtype=Float32),
    ],
    online=True,
    source=league_features_source,
    description="League-specific betting patterns and performance",
    tags={"team": "ml", "category": "league"},
)
