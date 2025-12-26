#!/usr/bin/env python3
"""
Market Performance Feature Engineering
=======================================
Advanced feature engineering for predicting market-level winrate and ROI.

This module creates time-series features for predicting future performance of
specific sport/league/market combinations to guide bet-taking decisions.

Target Variables:
- Future winrate (next 30 bets or 30 days)
- Future ROI (next 30 bets or 30 days)
- Confidence score (based on sample size and variance)
- Take/Skip recommendation

Feature Categories:
1. Historical Aggregates: Rolling winrate, ROI, volume over multiple windows
2. Trend Features: Momentum, trajectory, consistency metrics
3. Statistical Features: Confidence intervals, variance, sample adequacy
4. Context Features: Sport, league, institution patterns
5. Seasonal Features: Time-based patterns and sport schedules
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MarketPerformanceFeatureEngineer:
    """
    Feature engineering for market-level performance prediction.

    Aggregates historical bet data at sport/league/market level and creates
    features that predict future performance metrics.
    """

    def __init__(
        self,
        lookback_windows: List[int] = None,
        min_sample_size: int = 20,
        prediction_horizon_bets: int = 30,
        prediction_horizon_days: int = 30
    ):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: Rolling window sizes in bets (default: [10, 30, 50, 100])
            min_sample_size: Minimum bets required for reliable predictions
            prediction_horizon_bets: Number of future bets to predict over
            prediction_horizon_days: Number of future days to predict over
        """
        self.lookback_windows = lookback_windows or [10, 30, 50, 100]
        self.min_sample_size = min_sample_size
        self.prediction_horizon_bets = prediction_horizon_bets
        self.prediction_horizon_days = prediction_horizon_days
        self.scalers: Dict[str, StandardScaler] = {}
        self.fitted = False

    def prepare_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate bet-level data to market-level timeseries.

        Args:
            df: DataFrame with bet-level data (sport, league, market, is_win, roi, created_at)

        Returns:
            DataFrame with market-level aggregated data
        """
        # Ensure datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at').reset_index(drop=True)

        # Create market key
        df['market_key'] = (
            df['sport'].astype(str) + '|' +
            df['league'].fillna('Unknown').astype(str) + '|' +
            df['market'].astype(str)
        )

        # Create time-based grouping (daily)
        df['date'] = df['created_at'].dt.date

        # Aggregate to market-date level
        market_daily = df.groupby(['market_key', 'date']).agg({
            'is_win': ['sum', 'count', 'mean'],
            'roi': ['mean', 'std', 'sum'],
            'clv_percentage': ['mean', 'std', 'count'],
            'american_odds': ['mean', 'min', 'max'],
            'created_at': 'min',
            'sport': 'first',
            'league': 'first',
            'market': 'first',
            'institution_name': lambda x: x.mode()[0] if len(x) > 0 else None
        }).reset_index()

        # Flatten column names
        market_daily.columns = [
            'market_key', 'date',
            'wins', 'bet_count', 'winrate',
            'roi_mean', 'roi_std', 'roi_total',
            'clv_mean', 'clv_std', 'clv_count',
            'odds_mean', 'odds_min', 'odds_max',
            'timestamp', 'sport', 'league', 'market', 'primary_institution'
        ]

        # Fill NaN standard deviations
        market_daily['roi_std'] = market_daily['roi_std'].fillna(0)
        market_daily['clv_std'] = market_daily['clv_std'].fillna(0)

        # Sort by market and date
        market_daily = market_daily.sort_values(['market_key', 'date']).reset_index(drop=True)

        return market_daily

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables: future winrate and ROI.

        Targets are forward-looking:
        - Future winrate: win rate over next N bets
        - Future ROI: average ROI over next N bets

        Args:
            df: Market-level daily DataFrame

        Returns:
            DataFrame with target variables added
        """
        targets = []

        for market_key in df['market_key'].unique():
            market_df = df[df['market_key'] == market_key].copy()
            market_df = market_df.reset_index(drop=True)

            future_winrates = []
            future_rois = []
            future_bet_counts = []
            future_confidence = []

            for i in range(len(market_df)):
                # Look ahead from current row
                future_mask = market_df.index > i
                future_data = market_df[future_mask]

                if len(future_data) == 0:
                    # No future data available
                    future_winrates.append(np.nan)
                    future_rois.append(np.nan)
                    future_bet_counts.append(0)
                    future_confidence.append(0.0)
                    continue

                # Calculate cumulative future bets
                future_data['cumsum_bets'] = future_data['bet_count'].cumsum()

                # Get data within prediction horizon (bets or days)
                horizon_date = market_df.loc[i, 'date'] + timedelta(days=self.prediction_horizon_days)

                # Filter by bet count OR date (whichever comes first)
                horizon_data = future_data[
                    (future_data['cumsum_bets'] <= self.prediction_horizon_bets) |
                    (future_data['date'] <= horizon_date)
                ]

                if len(horizon_data) == 0:
                    future_winrates.append(np.nan)
                    future_rois.append(np.nan)
                    future_bet_counts.append(0)
                    future_confidence.append(0.0)
                    continue

                # Calculate future metrics
                total_bets = horizon_data['bet_count'].sum()
                total_wins = horizon_data['wins'].sum()

                future_wr = total_wins / total_bets if total_bets > 0 else np.nan

                # Weighted average ROI by bet count
                weighted_roi = (
                    (horizon_data['roi_mean'] * horizon_data['bet_count']).sum() /
                    total_bets if total_bets > 0 else np.nan
                )

                # Confidence based on sample size (sqrt of n adjustment)
                confidence = min(1.0, np.sqrt(total_bets / self.min_sample_size))

                future_winrates.append(future_wr)
                future_rois.append(weighted_roi)
                future_bet_counts.append(int(total_bets))
                future_confidence.append(confidence)

            market_df['target_winrate'] = future_winrates
            market_df['target_roi'] = future_rois
            market_df['target_bet_count'] = future_bet_counts
            market_df['target_confidence'] = future_confidence

            targets.append(market_df)

        result = pd.concat(targets, ignore_index=True)

        return result

    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical aggregate features with multiple time windows.

        Features include:
        - Rolling winrate/ROI over different windows
        - Expanding (all-time) winrate/ROI
        - Recent vs long-term performance deltas

        Args:
            df: Market-level daily DataFrame

        Returns:
            DataFrame with historical features added
        """
        result = []

        for market_key in df['market_key'].unique():
            market_df = df[df['market_key'] == market_key].copy()
            market_df = market_df.sort_values('date').reset_index(drop=True)

            # Cumulative bet count
            market_df['cumsum_bets'] = market_df['bet_count'].cumsum()

            # Expanding (all-time) metrics - shifted to avoid lookahead
            market_df['expanding_winrate'] = (
                market_df['wins'].shift(1).expanding().sum() /
                market_df['bet_count'].shift(1).expanding().sum()
            ).fillna(0.5)

            market_df['expanding_roi'] = (
                (market_df['roi_mean'] * market_df['bet_count']).shift(1).expanding().sum() /
                market_df['bet_count'].shift(1).expanding().sum()
            ).fillna(0.0)

            market_df['expanding_bet_count'] = (
                market_df['bet_count'].shift(1).expanding().sum()
            ).fillna(0)

            # Rolling window features (in terms of ROWS/DAYS, not bet count)
            for window in self.lookback_windows:
                # Rolling winrate
                market_df[f'rolling_{window}_winrate'] = (
                    market_df['wins'].shift(1).rolling(window, min_periods=1).sum() /
                    market_df['bet_count'].shift(1).rolling(window, min_periods=1).sum()
                ).fillna(0.5)

                # Rolling ROI
                market_df[f'rolling_{window}_roi'] = (
                    (market_df['roi_mean'] * market_df['bet_count']).shift(1).rolling(window, min_periods=1).sum() /
                    market_df['bet_count'].shift(1).rolling(window, min_periods=1).sum()
                ).fillna(0.0)

                # Rolling bet volume
                market_df[f'rolling_{window}_volume'] = (
                    market_df['bet_count'].shift(1).rolling(window, min_periods=1).sum()
                ).fillna(0)

                # Rolling CLV
                market_df[f'rolling_{window}_clv'] = (
                    market_df['clv_mean'].shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0.0)

                # Rolling variance (consistency)
                market_df[f'rolling_{window}_winrate_std'] = (
                    market_df['winrate'].shift(1).rolling(window, min_periods=2).std()
                ).fillna(0.0)

                market_df[f'rolling_{window}_roi_std'] = (
                    market_df['roi_mean'].shift(1).rolling(window, min_periods=2).std()
                ).fillna(0.0)

            result.append(market_df)

        result = pd.concat(result, ignore_index=True)

        return result

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend/momentum features.

        Features include:
        - Winrate momentum (recent vs long-term)
        - ROI trajectory (improving/declining)
        - Consistency scores (low variance = consistent)
        - Hot/cold streak indicators

        Args:
            df: DataFrame with historical features

        Returns:
            DataFrame with trend features added
        """
        # Short-term vs long-term momentum
        if 'rolling_10_winrate' in df.columns and 'rolling_100_winrate' in df.columns:
            df['winrate_momentum_10_100'] = (
                df['rolling_10_winrate'] - df['rolling_100_winrate']
            )
        else:
            df['winrate_momentum_10_100'] = 0.0

        if 'rolling_30_winrate' in df.columns and 'expanding_winrate' in df.columns:
            df['winrate_momentum_30_exp'] = (
                df['rolling_30_winrate'] - df['expanding_winrate']
            )
        else:
            df['winrate_momentum_30_exp'] = 0.0

        # ROI momentum
        if 'rolling_10_roi' in df.columns and 'rolling_50_roi' in df.columns:
            df['roi_momentum_10_50'] = (
                df['rolling_10_roi'] - df['rolling_50_roi']
            )
        else:
            df['roi_momentum_10_50'] = 0.0

        # Consistency score (inverse of variance)
        if 'rolling_30_winrate_std' in df.columns:
            df['consistency_score'] = 1.0 / (1.0 + df['rolling_30_winrate_std'])
        else:
            df['consistency_score'] = 0.5

        # Hot/cold indicators
        df['is_hot'] = (df['winrate_momentum_10_100'] > 0.05).astype(int)
        df['is_cold'] = (df['winrate_momentum_10_100'] < -0.05).astype(int)

        # Improving/declining ROI
        df['roi_improving'] = (df['roi_momentum_10_50'] > 1.0).astype(int)
        df['roi_declining'] = (df['roi_momentum_10_50'] < -1.0).astype(int)

        # Streak calculation (consecutive days with >50% winrate)
        for market_key in df['market_key'].unique():
            mask = df['market_key'] == market_key
            market_data = df[mask].copy()

            # Calculate streaks
            winning_days = (market_data['winrate'] > 0.5).astype(int)

            # Identify streak changes
            streak_changes = winning_days.diff().fillna(0) != 0
            streak_ids = streak_changes.cumsum()

            # Count streak length
            streaks = market_data.groupby(streak_ids).cumcount() + 1
            streaks = streaks * winning_days.map({1: 1, 0: -1})

            df.loc[mask, 'current_streak'] = streaks.shift(1).fillna(0).values

        df['hot_streak'] = (df['current_streak'] >= 3).astype(int)
        df['cold_streak'] = (df['current_streak'] <= -3).astype(int)

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical validity features.

        Features include:
        - Sample size adequacy scores
        - Confidence intervals
        - Statistical significance indicators
        - Variance-based reliability scores

        Args:
            df: DataFrame with existing features

        Returns:
            DataFrame with statistical features added
        """
        # Sample size adequacy (0-1 score based on sqrt scaling)
        df['sample_adequacy'] = np.clip(
            np.sqrt(df['expanding_bet_count'] / self.min_sample_size),
            0, 1
        )

        # Confidence interval width (narrower = more reliable)
        # Using Wilson score interval for binomial proportions
        z = 1.96  # 95% confidence

        n = df['expanding_bet_count'].clip(lower=1)
        p = df['expanding_winrate'].clip(0.001, 0.999)

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p * (1-p) / n + z**2 / (4*n**2))) / denominator

        df['ci_lower'] = center - margin
        df['ci_upper'] = center + margin
        df['ci_width'] = df['ci_upper'] - df['ci_lower']

        # Reliability score (inverse of CI width)
        df['reliability_score'] = 1.0 / (1.0 + df['ci_width'])

        # Statistical significance (is winrate significantly different from 50%?)
        # Using binomial test approximation
        df['significantly_profitable'] = (
            (df['expanding_winrate'] > 0.53) &
            (df['ci_lower'] > 0.5) &
            (df['expanding_bet_count'] >= self.min_sample_size)
        ).astype(int)

        df['significantly_unprofitable'] = (
            (df['expanding_winrate'] < 0.47) &
            (df['ci_upper'] < 0.5) &
            (df['expanding_bet_count'] >= self.min_sample_size)
        ).astype(int)

        # Volatility score (coefficient of variation for ROI)
        if 'rolling_30_roi_std' in df.columns:
            roi_mean = df['rolling_30_roi'].clip(lower=0.1)  # Avoid division by zero
            df['roi_volatility'] = df['rolling_30_roi_std'] / roi_mean
        else:
            df['roi_volatility'] = 0.0

        # Recent sample size (for short-term predictions)
        if 'rolling_30_volume' in df.columns:
            df['recent_sample_adequacy'] = np.clip(
                np.sqrt(df['rolling_30_volume'] / self.min_sample_size),
                0, 1
            )
        else:
            df['recent_sample_adequacy'] = 0.0

        return df

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features about sport, league, market.

        Features include:
        - Sport/league popularity (total bet volume)
        - Market type indicators (props vs main markets)
        - Institution diversity (number of different books)
        - Recency features (days since last bet)

        Args:
            df: DataFrame with existing features

        Returns:
            DataFrame with context features added
        """
        # Market popularity (total bets across all history)
        market_popularity = df.groupby('market_key')['bet_count'].sum().to_dict()
        df['market_popularity'] = df['market_key'].map(market_popularity)

        # Normalize popularity to 0-1 scale
        max_pop = df['market_popularity'].max()
        df['market_popularity_norm'] = df['market_popularity'] / max_pop if max_pop > 0 else 0

        # Identify player props
        player_prop_markets = [
            'Player Points', 'Player Rebounds', 'Player Assists', 'Player PRA',
            'Player Props', 'Player Touchdowns', 'Player Strikeouts', 'Anytime Scorer'
        ]
        df['is_player_prop'] = df['market'].isin(player_prop_markets).astype(int)

        # Main market types
        main_markets = ['Spread', 'Moneyline', 'Total', 'Over/Under']
        df['is_main_market'] = df['market'].isin(main_markets).astype(int)

        # Sport category
        ball_sports = ['Basketball', 'American Football', 'Baseball', 'Soccer', 'Ice Hockey']
        df['is_ball_sport'] = df['sport'].isin(ball_sports).astype(int)

        # Major leagues
        major_leagues = ['NBA', 'NFL', 'MLB', 'NHL', 'Premier League', 'NCAAB', 'NCAAFB']
        df['is_major_league'] = df['league'].isin(major_leagues).astype(int)

        # Days since last bet (recency)
        date_diff = df.groupby('market_key')['date'].diff()
        df['days_since_prev_bet'] = date_diff.apply(lambda x: x.days if pd.notna(x) else 1)

        # Active betting frequency (bets per day over rolling window)
        if 'rolling_30_volume' in df.columns:
            df['betting_frequency'] = df['rolling_30_volume'] / 30
        else:
            df['betting_frequency'] = 0.0

        return df

    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal/temporal features.

        Features include:
        - Day of week patterns
        - Month/season indicators
        - Sport season relevance
        - Holiday/weekend indicators

        Args:
            df: DataFrame with date column

        Returns:
            DataFrame with seasonal features added
        """
        # Convert date to datetime for operations
        df['date_dt'] = pd.to_datetime(df['date'])

        # Basic temporal
        df['day_of_week'] = df['date_dt'].dt.dayofweek
        df['month'] = df['date_dt'].dt.month
        df['quarter'] = df['date_dt'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Sport season indicators
        df['in_nfl_season'] = df['month'].isin([9, 10, 11, 12, 1, 2]).astype(int)
        df['in_nba_season'] = df['month'].isin([10, 11, 12, 1, 2, 3, 4, 5, 6]).astype(int)
        df['in_mlb_season'] = df['month'].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
        df['in_nhl_season'] = df['month'].isin([10, 11, 12, 1, 2, 3, 4, 5, 6]).astype(int)

        # Playoff indicators (approximate)
        df['likely_nfl_playoffs'] = df['month'].isin([1, 2]).astype(int)
        df['likely_nba_playoffs'] = df['month'].isin([4, 5, 6]).astype(int)
        df['likely_mlb_playoffs'] = df['month'].isin([10]).astype(int)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Args:
            df: Bet-level DataFrame

        Returns:
            Market-level DataFrame with all features and targets
        """
        # Step 1: Aggregate to market-level
        market_df = self.prepare_market_data(df)

        # Step 2: Create target variables
        market_df = self.create_target_variables(market_df)

        # Step 3: Create features
        market_df = self.create_historical_features(market_df)
        market_df = self.create_trend_features(market_df)
        market_df = self.create_statistical_features(market_df)
        market_df = self.create_context_features(market_df)
        market_df = self.create_seasonal_features(market_df)

        self.fitted = True

        return market_df

    def get_feature_names(self) -> List[str]:
        """Return list of feature column names."""
        # All feature columns (excluding identifiers and targets)
        exclude_cols = [
            'market_key', 'date', 'date_dt', 'sport', 'league', 'market',
            'primary_institution', 'timestamp',
            'wins', 'bet_count', 'winrate', 'roi_mean', 'roi_std', 'roi_total',
            'clv_mean', 'clv_std', 'clv_count', 'odds_mean', 'odds_min', 'odds_max',
            'target_winrate', 'target_roi', 'target_bet_count', 'target_confidence',
            'cumsum_bets'
        ]

        # Features are all numerical columns not in exclude list
        return [col for col in exclude_cols if col not in exclude_cols]


def validate_sample_size(
    df: pd.DataFrame,
    min_samples: int = 20,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Statistical validation for sample size adequacy.

    Determines whether each market has sufficient historical data for
    reliable predictions using power analysis and confidence intervals.

    Args:
        df: Market-level DataFrame with bet counts
        min_samples: Minimum sample size threshold
        confidence_level: Confidence level for statistical tests

    Returns:
        DataFrame with validation metrics added
    """
    # Sample size flags
    df['sufficient_sample'] = (df['expanding_bet_count'] >= min_samples).astype(int)

    # Minimum detectable effect (MDE) based on sample size
    # MDE = z * sqrt(p(1-p)/n) where p = 0.5 (null hypothesis)
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    n = df['expanding_bet_count'].clip(lower=1)
    p = 0.5

    df['min_detectable_effect'] = z * np.sqrt(p * (1 - p) / n)

    # Power analysis: probability of detecting true effect
    # Assuming desired effect size of 0.05 (5% winrate difference)
    desired_effect = 0.05
    effect_size = desired_effect / np.sqrt(p * (1 - p))

    df['statistical_power'] = stats.norm.cdf(
        effect_size * np.sqrt(n) - z
    )

    # Adequate power indicator (>80% power)
    df['adequate_power'] = (df['statistical_power'] >= 0.8).astype(int)

    # Overall validation flag
    df['validated'] = (
        (df['sufficient_sample'] == 1) &
        (df['adequate_power'] == 1) &
        (df['sample_adequacy'] >= 0.7)
    ).astype(int)

    return df


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("MARKET PERFORMANCE FEATURE ENGINEERING")
    print("=" * 80)

    # Test with sample data
    np.random.seed(42)
    n_bets = 500

    sample_bets = pd.DataFrame({
        'sport': np.random.choice(['Basketball', 'American Football', 'Baseball'], n_bets),
        'league': np.random.choice(['NBA', 'NFL', 'MLB'], n_bets),
        'market': np.random.choice(['Spread', 'Moneyline', 'Total', 'Player Points'], n_bets),
        'institution_name': np.random.choice(['DraftKings', 'FanDuel', 'BetMGM'], n_bets),
        'is_win': np.random.random(n_bets) > 0.48,  # Slight edge
        'roi': np.random.normal(2, 15, n_bets),
        'clv_percentage': np.random.normal(0.5, 2, n_bets),
        'american_odds': np.random.choice([-110, -105, 100, 110], n_bets),
        'created_at': pd.date_range('2024-01-01', periods=n_bets, freq='6H')
    })

    print(f"\nSample data: {len(sample_bets)} bets")
    print(f"Unique markets: {sample_bets['sport'].nunique() * sample_bets['league'].nunique() * sample_bets['market'].nunique()}")

    # Create features
    fe = MarketPerformanceFeatureEngineer(
        lookback_windows=[10, 30, 50],
        min_sample_size=20
    )

    print("\nAggregating to market-level and creating features...")
    market_features = fe.fit_transform(sample_bets)

    print(f"\nMarket-level rows: {len(market_features)}")
    print(f"Feature columns: {len([c for c in market_features.columns if c.startswith(('rolling_', 'expanding_', 'winrate_', 'roi_'))])}")

    # Add validation
    market_features = validate_sample_size(market_features, min_samples=20)

    print("\nSample adequacy statistics:")
    print(f"  Sufficient sample: {market_features['sufficient_sample'].sum()} / {len(market_features)}")
    print(f"  Adequate power: {market_features['adequate_power'].sum()} / {len(market_features)}")
    print(f"  Fully validated: {market_features['validated'].sum()} / {len(market_features)}")

    # Show sample features
    print("\nSample feature values (first row):")
    feature_cols = [c for c in market_features.columns if c.startswith(('rolling_', 'expanding_', 'target_'))][:10]
    print(market_features[feature_cols].head(1).T)

    print("\n" + "=" * 80)
    print("Feature engineering complete!")
