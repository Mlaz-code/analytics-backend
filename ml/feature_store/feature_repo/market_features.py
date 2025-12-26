#!/usr/bin/env python3
"""
Market Performance Feature Views for Feast
==========================================
Feature definitions for market-level predictions.

These features support real-time take/skip decisions in the Chrome extension
and historical analysis in the dashboard.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String


# =============================================================================
# Market Entity
# =============================================================================

market_entity = Entity(
    name="market_key",
    description="Unique identifier for sport+league+market combination (e.g., 'Basketball|NBA|Spread')",
    value_type=String,
    join_keys=["market_key"]
)


# =============================================================================
# Market Performance Feature View
# =============================================================================

market_performance_source = FileSource(
    name="market_performance_source",
    path="/root/pikkit/ml/feature_store/data/market_performance.parquet",
    timestamp_field="event_timestamp",
)

market_performance_features = FeatureView(
    name="market_performance",
    entities=[market_entity],
    ttl=timedelta(hours=12),  # Refresh every 12 hours
    schema=[
        # Expanding (all-time) metrics
        Field(name="expanding_winrate", dtype=Float32),
        Field(name="expanding_roi", dtype=Float32),
        Field(name="expanding_bet_count", dtype=Int64),

        # Rolling window metrics (10 bets)
        Field(name="rolling_10_winrate", dtype=Float32),
        Field(name="rolling_10_roi", dtype=Float32),
        Field(name="rolling_10_volume", dtype=Int64),
        Field(name="rolling_10_clv", dtype=Float32),
        Field(name="rolling_10_winrate_std", dtype=Float32),
        Field(name="rolling_10_roi_std", dtype=Float32),

        # Rolling window metrics (30 bets)
        Field(name="rolling_30_winrate", dtype=Float32),
        Field(name="rolling_30_roi", dtype=Float32),
        Field(name="rolling_30_volume", dtype=Int64),
        Field(name="rolling_30_clv", dtype=Float32),
        Field(name="rolling_30_winrate_std", dtype=Float32),
        Field(name="rolling_30_roi_std", dtype=Float32),

        # Rolling window metrics (50 bets)
        Field(name="rolling_50_winrate", dtype=Float32),
        Field(name="rolling_50_roi", dtype=Float32),
        Field(name="rolling_50_volume", dtype=Int64),
        Field(name="rolling_50_clv", dtype=Float32),

        # Rolling window metrics (100 bets)
        Field(name="rolling_100_winrate", dtype=Float32),
        Field(name="rolling_100_roi", dtype=Float32),
        Field(name="rolling_100_volume", dtype=Int64),

        # Trend indicators
        Field(name="winrate_momentum_10_100", dtype=Float32),
        Field(name="winrate_momentum_30_exp", dtype=Float32),
        Field(name="roi_momentum_10_50", dtype=Float32),
        Field(name="consistency_score", dtype=Float32),

        # Statistical validation
        Field(name="sample_adequacy", dtype=Float32),
        Field(name="ci_lower", dtype=Float32),
        Field(name="ci_upper", dtype=Float32),
        Field(name="ci_width", dtype=Float32),
        Field(name="reliability_score", dtype=Float32),

        # Context indicators
        Field(name="is_hot", dtype=Int64),
        Field(name="is_cold", dtype=Int64),
        Field(name="roi_improving", dtype=Int64),
        Field(name="roi_declining", dtype=Int64),
        Field(name="significantly_profitable", dtype=Int64),
        Field(name="significantly_unprofitable", dtype=Int64),

        # Metadata
        Field(name="market_popularity_norm", dtype=Float32),
        Field(name="is_player_prop", dtype=Int64),
        Field(name="is_main_market", dtype=Int64),
        Field(name="betting_frequency", dtype=Float32),
    ],
    online=True,
    source=market_performance_source,
    description="Market-level historical performance and trend features",
    tags={"team": "ml", "category": "market_performance", "version": "v1"}
)


# =============================================================================
# Market Predictions Feature View
# =============================================================================

market_predictions_source = FileSource(
    name="market_predictions_source",
    path="/root/pikkit/ml/feature_store/data/market_predictions.parquet",
    timestamp_field="event_timestamp",
)

market_predictions = FeatureView(
    name="market_predictions",
    entities=[market_entity],
    ttl=timedelta(hours=6),  # Predictions refresh every 6 hours
    schema=[
        # Model predictions
        Field(name="predicted_winrate", dtype=Float32),
        Field(name="predicted_roi", dtype=Float32),
        Field(name="prediction_confidence", dtype=Float32),

        # Recommendation
        Field(name="should_take", dtype=Int64),
        Field(name="recommendation_score", dtype=Float32),
        Field(name="grade", dtype=String),

        # Model metadata
        Field(name="model_version", dtype=String),
        Field(name="prediction_timestamp", dtype=String),
    ],
    online=True,
    source=market_predictions_source,
    description="ML model predictions for market performance",
    tags={"team": "ml", "category": "predictions", "version": "v1"}
)


# =============================================================================
# Sport Context Features
# =============================================================================

sport_context_source = FileSource(
    name="sport_context_source",
    path="/root/pikkit/ml/feature_store/data/sport_context.parquet",
    timestamp_field="event_timestamp",
)

sport_context_features = FeatureView(
    name="sport_context",
    entities=[market_entity],
    ttl=timedelta(days=1),
    schema=[
        # Sport/League metadata
        Field(name="is_ball_sport", dtype=Int64),
        Field(name="is_major_league", dtype=Int64),

        # Seasonal indicators
        Field(name="in_nfl_season", dtype=Int64),
        Field(name="in_nba_season", dtype=Int64),
        Field(name="in_mlb_season", dtype=Int64),
        Field(name="in_nhl_season", dtype=Int64),

        # Temporal patterns
        Field(name="day_of_week_sin", dtype=Float32),
        Field(name="day_of_week_cos", dtype=Float32),
        Field(name="month_sin", dtype=Float32),
        Field(name="month_cos", dtype=Float32),
        Field(name="is_weekend", dtype=Int64),
    ],
    online=True,
    source=sport_context_source,
    description="Sport and temporal context features",
    tags={"team": "ml", "category": "context", "version": "v1"}
)


# =============================================================================
# Feature Service for Chrome Extension
# =============================================================================

from feast import FeatureService

market_recommendation_service = FeatureService(
    name="market_recommendation",
    features=[
        market_performance_features,
        market_predictions,
        sport_context_features,
    ],
    description="Feature service for real-time market recommendations in Chrome extension",
    tags={"application": "chrome_extension", "version": "v1"}
)


# =============================================================================
# Feature Service for Dashboard
# =============================================================================

market_analytics_service = FeatureService(
    name="market_analytics",
    features=[
        market_performance_features,
        market_predictions,
        sport_context_features,
    ],
    description="Feature service for historical market analytics in dashboard",
    tags={"application": "dashboard", "version": "v1"}
)
