#!/usr/bin/env python3
"""
Pikkit Advanced Feature Engineering Module
==========================================
Comprehensive feature engineering for sports betting ML models.

Feature Categories:
1. Player-Level Features (for props markets)
2. Line Movement Features (opening vs current odds)
3. Market Efficiency Indicators
4. Cross-Sport Correlation Features
5. Streak/Momentum Features
6. Advanced CLV Features
7. Bookmaker Efficiency Features
8. Temporal Pattern Features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline for sports betting data.
    Designed to extract predictive signals from betting history.
    """

    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: List of lookback periods for rolling features
                              Default: [5, 10, 20, 50, 100]
        """
        self.lookback_windows = lookback_windows or [5, 10, 20, 50, 100]
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineer and transform data.

        Args:
            df: DataFrame with betting data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = self._preprocess(df)

        # Core features
        df = self._create_base_features(df)
        df = self._create_implied_probability_features(df)
        df = self._create_clv_features(df)

        # Advanced features
        df = self._create_line_movement_features(df)
        df = self._create_market_efficiency_features(df)
        df = self._create_streak_momentum_features(df)
        df = self._create_bookmaker_features(df)
        df = self._create_temporal_features(df)
        df = self._create_player_prop_features(df)
        df = self._create_cross_sport_features(df)

        # Historical performance with multiple windows
        df = self._create_historical_features(df)

        # Categorical encoding
        df = self._encode_categoricals(df, fit=True)

        self.fitted = True
        self._store_feature_names(df)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineer.

        Args:
            df: DataFrame with new betting data

        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            raise ValueError("Feature engineer not fitted. Call fit_transform first.")

        df = df.copy()
        df = self._preprocess(df)

        # Apply same transformations
        df = self._create_base_features(df)
        df = self._create_implied_probability_features(df)
        df = self._create_clv_features(df)
        df = self._create_line_movement_features(df)
        df = self._create_market_efficiency_features(df)
        df = self._create_streak_momentum_features(df)
        df = self._create_bookmaker_features(df)
        df = self._create_temporal_features(df)
        df = self._create_player_prop_features(df)
        df = self._create_cross_sport_features(df)
        df = self._create_historical_features(df)
        df = self._encode_categoricals(df, fit=False)

        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing and data cleaning."""
        # Convert dates
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        if 'settled_at' in df.columns:
            df['settled_at'] = pd.to_datetime(df['settled_at'])

        # Sort by date
        if 'created_at' in df.columns:
            df = df.sort_values('created_at').reset_index(drop=True)

        # Handle target variable
        if 'is_win' in df.columns:
            df['won'] = df['is_win'].fillna(False).astype(int)
        elif 'result' in df.columns:
            df['won'] = (df['result'] == 'Won').astype(int)

        # Handle profit/ROI
        if 'roi' not in df.columns and 'profit' in df.columns and 'amount' in df.columns:
            df['roi'] = (df['profit'] / df['amount'].replace(0, 1)) * 100
        elif 'roi' not in df.columns:
            df['roi'] = 0

        return df

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features from raw data."""
        # Odds column handling
        odds_col = 'american_odds' if 'american_odds' in df.columns else 'odds'
        if odds_col in df.columns:
            df['odds'] = df[odds_col]
        else:
            df['odds'] = -110  # Default

        # Is live bet
        if 'is_live' in df.columns:
            df['is_live'] = df['is_live'].fillna(False).astype(int)
        else:
            df['is_live'] = 0

        # Bet amount features
        if 'amount' in df.columns:
            df['log_bet_amount'] = np.log1p(df['amount'])
            df['bet_amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        else:
            df['log_bet_amount'] = 0
            df['bet_amount_zscore'] = 0

        return df

    def _create_implied_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from implied probability."""
        def american_to_implied(odds):
            if pd.isna(odds):
                return 0.5
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            else:
                return 100 / (odds + 100)

        def american_to_decimal(odds):
            if pd.isna(odds):
                return 2.0
            if odds < 0:
                return 1 + (100 / abs(odds))
            else:
                return 1 + (odds / 100)

        # Basic implied probability
        df['implied_prob'] = df['odds'].apply(american_to_implied)
        df['decimal_odds'] = df['odds'].apply(american_to_decimal)

        # Probability-based features
        df['odds_value'] = df['decimal_odds'] - 1  # Potential profit per unit
        df['implied_prob_squared'] = df['implied_prob'] ** 2
        df['implied_prob_log'] = np.log(df['implied_prob'].clip(0.01, 0.99))

        # Favorite/underdog indicator
        df['is_favorite'] = (df['implied_prob'] > 0.5).astype(int)
        df['is_heavy_favorite'] = (df['implied_prob'] > 0.7).astype(int)
        df['is_heavy_underdog'] = (df['implied_prob'] < 0.3).astype(int)

        # Expected value assuming 50% win rate
        df['naive_ev'] = (df['decimal_odds'] * 0.5) - 1

        return df

    def _create_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced CLV (Closing Line Value) features."""
        # Basic CLV
        df['has_clv'] = df['clv_percentage'].notna().astype(int) if 'clv_percentage' in df.columns else 0
        df['clv_percentage'] = df['clv_percentage'].fillna(0) if 'clv_percentage' in df.columns else 0

        # CLV-derived features
        df['clv_positive'] = (df['clv_percentage'] > 0).astype(int)
        df['clv_strong_positive'] = (df['clv_percentage'] > 2).astype(int)
        df['clv_negative'] = (df['clv_percentage'] < -1).astype(int)

        # CLV interaction with implied probability
        df['clv_ev'] = df['clv_percentage'] * df['implied_prob']
        df['clv_adjusted_prob'] = (df['implied_prob'] + df['clv_percentage'] / 100).clip(0, 1)

        # CLV expected value
        df['clv_expected_roi'] = df['clv_percentage'] * df['decimal_odds']

        # Rolling CLV statistics
        if len(df) > 10:
            df['clv_rolling_mean'] = df['clv_percentage'].rolling(10, min_periods=1).mean()
            df['clv_rolling_std'] = df['clv_percentage'].rolling(10, min_periods=1).std().fillna(0)
            df['clv_zscore'] = np.where(
                df['clv_rolling_std'] > 0,
                (df['clv_percentage'] - df['clv_rolling_mean']) / df['clv_rolling_std'],
                0
            )
        else:
            df['clv_rolling_mean'] = df['clv_percentage']
            df['clv_rolling_std'] = 0
            df['clv_zscore'] = 0

        return df

    def _create_line_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to line movement.
        Note: These require opening_odds field. If not available, use proxies.
        """
        # Check for opening odds
        if 'opening_odds' in df.columns:
            df['opening_implied'] = df['opening_odds'].apply(
                lambda x: abs(x)/(abs(x)+100) if x < 0 else 100/(x+100) if pd.notna(x) else 0.5
            )
            df['line_movement'] = df['implied_prob'] - df['opening_implied']
            df['line_moved_favorable'] = (df['line_movement'] < 0).astype(int)  # Line moved in our favor
            df['line_movement_magnitude'] = abs(df['line_movement'])
        else:
            # Use CLV as proxy for line movement
            df['line_movement'] = -df['clv_percentage'] / 100  # CLV is inverse of line movement
            df['line_moved_favorable'] = (df['clv_percentage'] > 0).astype(int)
            df['line_movement_magnitude'] = abs(df['clv_percentage']) / 100

        # Steam move indicator (significant line movement)
        df['steam_move'] = (df['line_movement_magnitude'] > 0.03).astype(int)

        # Sharp money indicator (line moves against public)
        # Approximation: large moves in favorites suggest sharp action
        df['sharp_money_indicator'] = (
            (df['is_favorite'] == 1) & (df['line_moved_favorable'] == 1)
        ).astype(int)

        return df

    def _create_market_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture market efficiency.
        Efficient markets = harder to beat.
        """
        # Market type efficiency (some markets are more efficient than others)
        if 'market' in df.columns:
            # Calculate historical efficiency by market
            market_eff = df.groupby('market').agg({
                'won': 'mean',
                'roi': 'mean'
            }).reset_index()
            market_eff.columns = ['market', 'market_historical_winrate', 'market_historical_roi']

            df = df.merge(market_eff, on='market', how='left')
            df['market_historical_winrate'] = df['market_historical_winrate'].fillna(0.5)
            df['market_historical_roi'] = df['market_historical_roi'].fillna(0)

            # Market efficiency score (deviation from implied prob)
            df['market_efficiency'] = abs(df['market_historical_winrate'] - df['implied_prob'])
        else:
            df['market_historical_winrate'] = 0.5
            df['market_historical_roi'] = 0
            df['market_efficiency'] = 0

        # Time until game start as efficiency proxy
        # (Closer to game time = more efficient markets)
        if 'game_time' in df.columns and 'created_at' in df.columns:
            df['hours_until_game'] = (
                pd.to_datetime(df['game_time']) - df['created_at']
            ).dt.total_seconds() / 3600
            df['early_bet'] = (df['hours_until_game'] > 24).astype(int)
        else:
            df['hours_until_game'] = 12  # Default
            df['early_bet'] = 0

        # High volume market indicator (major sports/leagues)
        major_leagues = ['NBA', 'NFL', 'MLB', 'NHL', 'NCAAB', 'NCAAFB', 'Premier League']
        if 'league' in df.columns:
            df['major_league'] = df['league'].isin(major_leagues).astype(int)
        else:
            df['major_league'] = 0

        return df

    def _create_streak_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create streak and momentum features."""
        if 'won' not in df.columns or len(df) < 2:
            df['current_streak'] = 0
            df['hot_streak'] = 0
            df['cold_streak'] = 0
            df['momentum_score'] = 0
            return df

        # Calculate streaks
        streaks = []
        current_streak = 0
        prev_result = None

        for result in df['won'].values:
            if prev_result is None:
                current_streak = 1 if result == 1 else -1
            elif result == 1 and prev_result == 1:
                current_streak = max(1, current_streak) + 1
            elif result == 0 and prev_result == 0:
                current_streak = min(-1, current_streak) - 1
            else:
                current_streak = 1 if result == 1 else -1

            streaks.append(current_streak)
            prev_result = result

        # Shift by 1 to avoid lookahead (use previous streak for current bet)
        df['current_streak'] = [0] + streaks[:-1]

        # Streak indicators
        df['hot_streak'] = (df['current_streak'] >= 3).astype(int)
        df['cold_streak'] = (df['current_streak'] <= -3).astype(int)
        df['streak_magnitude'] = abs(df['current_streak'])

        # Momentum score (weighted recent performance)
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = (
                df['won'].shift(1).rolling(window, min_periods=1).apply(
                    lambda x: np.average(x, weights=np.exp(np.linspace(0, 1, len(x))))
                ).fillna(0.5)
            )

        # Combined momentum score
        df['momentum_score'] = (
            0.5 * df['momentum_5'] +
            0.3 * df['momentum_10'] +
            0.2 * df['momentum_20']
        )

        # Variance in recent results (consistency)
        df['result_variance'] = df['won'].shift(1).rolling(10, min_periods=2).var().fillna(0.25)

        return df

    def _create_bookmaker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bookmaker-specific features."""
        if 'institution_name' not in df.columns:
            df['bookmaker_historical_winrate'] = 0.5
            df['bookmaker_historical_roi'] = 0
            df['bookmaker_value_score'] = 0
            return df

        # Calculate historical performance by bookmaker
        book_stats = df.groupby('institution_name').agg({
            'won': 'mean',
            'roi': 'mean',
            'clv_percentage': 'mean'
        }).reset_index()
        book_stats.columns = ['institution_name', 'bookmaker_historical_winrate',
                              'bookmaker_historical_roi', 'bookmaker_avg_clv']

        df = df.merge(book_stats, on='institution_name', how='left')
        df['bookmaker_historical_winrate'] = df['bookmaker_historical_winrate'].fillna(0.5)
        df['bookmaker_historical_roi'] = df['bookmaker_historical_roi'].fillna(0)
        df['bookmaker_avg_clv'] = df['bookmaker_avg_clv'].fillna(0)

        # Sharp bookmaker indicator (bookmakers known to have sharper lines)
        sharp_books = ['Pinnacle', 'Circa', 'bet365']  # Known sharps
        soft_books = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars']  # Recreational

        df['is_sharp_book'] = df['institution_name'].isin(sharp_books).astype(int)
        df['is_soft_book'] = df['institution_name'].isin(soft_books).astype(int)

        # Bookmaker value score
        df['bookmaker_value_score'] = (
            df['bookmaker_historical_roi'] * 0.4 +
            df['bookmaker_avg_clv'] * 0.6
        )

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'created_at' not in df.columns:
            # Default temporal features
            df['day_of_week'] = 0
            df['hour_of_day'] = 12
            df['is_weekend'] = 0
            df['is_primetime'] = 0
            df['days_since_first_bet'] = 0
            df['bet_frequency_7d'] = 0
            return df

        # Basic temporal
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['hour_of_day'] = df['created_at'].dt.hour
        df['day_of_month'] = df['created_at'].dt.day
        df['month'] = df['created_at'].dt.month
        df['quarter'] = df['created_at'].dt.quarter

        # Cyclical encoding for time features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Time-based indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_primetime'] = ((df['hour_of_day'] >= 18) & (df['hour_of_day'] <= 23)).astype(int)
        df['is_morning'] = (df['hour_of_day'] < 12).astype(int)
        df['is_late_night'] = ((df['hour_of_day'] >= 23) | (df['hour_of_day'] < 6)).astype(int)

        # Days since first bet (experience proxy)
        first_bet = df['created_at'].min()
        df['days_since_first_bet'] = (df['created_at'] - first_bet).dt.days
        df['weeks_since_first_bet'] = df['days_since_first_bet'] / 7

        # Betting frequency
        df['bet_frequency_7d'] = df['created_at'].apply(
            lambda x: len(df[(df['created_at'] >= x - timedelta(days=7)) &
                            (df['created_at'] < x)])
        )

        # Season indicator (for seasonal sports)
        # NFL: Sept-Feb, MLB: Apr-Oct, NBA/NHL: Oct-Jun
        df['is_nfl_season'] = ((df['month'] >= 9) | (df['month'] <= 2)).astype(int)
        df['is_mlb_season'] = ((df['month'] >= 4) & (df['month'] <= 10)).astype(int)
        df['is_nba_season'] = ((df['month'] >= 10) | (df['month'] <= 6)).astype(int)

        return df

    def _create_player_prop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for player prop markets.
        These are particularly important for props which have different dynamics.
        """
        if 'market' not in df.columns:
            df['is_player_prop'] = 0
            df['prop_category'] = 0
            return df

        # Identify player props
        player_prop_markets = [
            'Player Points', 'Player Rebounds', 'Player Assists',
            'Player PRA', 'Player P+R', 'Player P+A', 'Player R+A',
            'Player 3-Pointers', 'Player Blocks', 'Player Steals+Blocks',
            'Player Double-Double', 'Player Triple-Double',
            'Player Passing Yards', 'Player Rushing Yards', 'Player Receiving Yards',
            'Player Receptions', 'Player Passing TDs', 'Player Anytime TD',
            'Player Goals', 'Player Shots', 'Player Saves',
            'Pitcher Strikeouts', 'Player Hits', 'Player Home Runs',
            'Player Total Bases', 'Other Player Props'
        ]

        df['is_player_prop'] = df['market'].isin(player_prop_markets).astype(int)

        # Prop category encoding
        scoring_props = ['Player Points', 'Player Goals', 'Player Home Runs', 'Player Anytime TD']
        combo_props = ['Player PRA', 'Player P+R', 'Player P+A', 'Player R+A',
                       'Player Double-Double', 'Player Triple-Double']
        efficiency_props = ['Player 3-Pointers', 'Player Blocks', 'Player Steals+Blocks']
        volume_props = ['Player Rebounds', 'Player Assists', 'Player Receptions',
                        'Player Shots', 'Player Hits']

        df['is_scoring_prop'] = df['market'].isin(scoring_props).astype(int)
        df['is_combo_prop'] = df['market'].isin(combo_props).astype(int)
        df['is_efficiency_prop'] = df['market'].isin(efficiency_props).astype(int)
        df['is_volume_prop'] = df['market'].isin(volume_props).astype(int)

        # Props tend to have higher variance
        df['prop_variance_multiplier'] = np.where(
            df['is_player_prop'] == 1,
            np.where(df['is_combo_prop'] == 1, 1.5, 1.2),  # Combo props have highest variance
            1.0
        )

        # Historical prop performance
        if 'won' in df.columns:
            prop_perf = df[df['is_player_prop'] == 1].groupby('market').agg({
                'won': ['mean', 'count'],
                'roi': 'mean'
            }).reset_index()
            if len(prop_perf) > 0:
                prop_perf.columns = ['market', 'prop_market_winrate', 'prop_market_count', 'prop_market_roi']
                df = df.merge(prop_perf, on='market', how='left')
                df['prop_market_winrate'] = df['prop_market_winrate'].fillna(0.5)
                df['prop_market_count'] = df['prop_market_count'].fillna(0)
                df['prop_market_roi'] = df['prop_market_roi'].fillna(0)
            else:
                df['prop_market_winrate'] = 0.5
                df['prop_market_count'] = 0
                df['prop_market_roi'] = 0
        else:
            df['prop_market_winrate'] = 0.5
            df['prop_market_count'] = 0
            df['prop_market_roi'] = 0

        return df

    def _create_cross_sport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture cross-sport patterns.
        Some bettors may be better at certain types of sports.
        """
        if 'sport' not in df.columns or 'won' not in df.columns:
            df['sport_expertise_score'] = 0
            df['sport_consistency'] = 0
            return df

        # Sport groupings (similar sports may have transferable skills)
        ball_sports = ['Basketball', 'American Football', 'Baseball', 'Soccer', 'Ice Hockey']
        individual_sports = ['Tennis', 'Golf', 'MMA', 'Boxing']

        df['is_ball_sport'] = df['sport'].isin(ball_sports).astype(int)
        df['is_individual_sport'] = df['sport'].isin(individual_sports).astype(int)

        # Calculate sport group performance
        if len(df) > 10:
            # Ball sports performance
            ball_perf = df[df['is_ball_sport'] == 1].groupby('sport').agg({
                'won': 'mean',
                'roi': 'mean'
            })
            if len(ball_perf) > 0:
                df['ball_sport_avg_winrate'] = ball_perf['won'].mean()
                df['ball_sport_avg_roi'] = ball_perf['roi'].mean()
            else:
                df['ball_sport_avg_winrate'] = 0.5
                df['ball_sport_avg_roi'] = 0

            # Individual sports performance
            ind_perf = df[df['is_individual_sport'] == 1].groupby('sport').agg({
                'won': 'mean',
                'roi': 'mean'
            })
            if len(ind_perf) > 0:
                df['individual_sport_avg_winrate'] = ind_perf['won'].mean()
                df['individual_sport_avg_roi'] = ind_perf['roi'].mean()
            else:
                df['individual_sport_avg_winrate'] = 0.5
                df['individual_sport_avg_roi'] = 0
        else:
            df['ball_sport_avg_winrate'] = 0.5
            df['ball_sport_avg_roi'] = 0
            df['individual_sport_avg_winrate'] = 0.5
            df['individual_sport_avg_roi'] = 0

        # Sport expertise score (relative performance vs average)
        sport_stats = df.groupby('sport').agg({
            'won': ['mean', 'std', 'count'],
            'roi': 'mean'
        }).reset_index()
        sport_stats.columns = ['sport', 'sport_mean_win', 'sport_std_win',
                               'sport_count', 'sport_mean_roi']

        df = df.merge(sport_stats, on='sport', how='left')
        df['sport_mean_win'] = df['sport_mean_win'].fillna(0.5)
        df['sport_std_win'] = df['sport_std_win'].fillna(0.1)
        df['sport_count'] = df['sport_count'].fillna(0)
        df['sport_mean_roi'] = df['sport_mean_roi'].fillna(0)

        # Expertise score (higher is better)
        df['sport_expertise_score'] = (
            (df['sport_mean_win'] - 0.5) * 10 +  # Win rate contribution
            df['sport_mean_roi'] / 10  # ROI contribution
        )

        # Consistency (lower variance = more consistent)
        df['sport_consistency'] = 1 - df['sport_std_win'].clip(0, 0.5) * 2

        return df

    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical performance features with multiple lookback windows.
        Uses expanding window to avoid lookahead bias.
        """
        if 'won' not in df.columns:
            return df

        # Group combinations for historical features
        group_configs = [
            (['sport'], 'sport'),
            (['market'], 'market'),
            (['sport', 'market'], 'sport_market'),
            (['sport', 'league'], 'sport_league'),
            (['sport', 'league', 'market'], 'sport_league_market'),
            (['institution_name'], 'institution'),
            (['sport', 'institution_name'], 'sport_institution'),
        ]

        for group_cols, prefix in group_configs:
            # Check if columns exist
            if not all(col in df.columns for col in group_cols):
                continue

            # Expanding window (all history) - shifted to avoid lookahead
            df[f'{prefix}_expanding_winrate'] = (
                df.groupby(group_cols)['won']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0.5)
            )

            df[f'{prefix}_expanding_roi'] = (
                df.groupby(group_cols)['roi']
                .transform(lambda x: x.shift(1).expanding().mean())
                .fillna(0)
            )

            # Count (sample size)
            df[f'{prefix}_count'] = df.groupby(group_cols).cumcount()

            # Rolling windows
            for window in self.lookback_windows:
                df[f'{prefix}_rolling_{window}_winrate'] = (
                    df.groupby(group_cols)['won']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    .fillna(0.5)
                )

                df[f'{prefix}_rolling_{window}_roi'] = (
                    df.groupby(group_cols)['roi']
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    .fillna(0)
                )

        # Recent trend (is performance improving or declining?)
        if 'sport_market_rolling_5_winrate' in df.columns and 'sport_market_rolling_20_winrate' in df.columns:
            df['winrate_trend'] = (
                df['sport_market_rolling_5_winrate'] - df['sport_market_rolling_20_winrate']
            )
            df['improving_trend'] = (df['winrate_trend'] > 0.05).astype(int)
            df['declining_trend'] = (df['winrate_trend'] < -0.05).astype(int)
        else:
            df['winrate_trend'] = 0
            df['improving_trend'] = 0
            df['declining_trend'] = 0

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = ['sport', 'league', 'market', 'institution_name', 'bet_type']

        for col in categorical_cols:
            if col not in df.columns:
                df[f'{col}_encoded'] = 0
                continue

            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                else:
                    df[f'{col}_encoded'] = 0

        return df

    def _store_feature_names(self, df: pd.DataFrame):
        """Store list of feature names for later use."""
        exclude_cols = [
            'id', 'bet_id', 'created_at', 'settled_at', 'updated_at',
            'result', 'won', 'is_win', 'profit', 'roi', 'amount', 'wager',
            'to_win', 'american_odds', 'opening_odds', 'game_time',
            'sport', 'league', 'market', 'institution_name', 'bet_type',
            'clv_percentage'  # Keep derived clv features, not raw
        ]

        self.feature_names = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return features grouped by category for analysis."""
        groups = {
            'odds_probability': [f for f in self.feature_names if any(
                x in f for x in ['implied', 'odds', 'decimal', 'favorite', 'underdog', 'naive_ev']
            )],
            'clv': [f for f in self.feature_names if 'clv' in f.lower()],
            'line_movement': [f for f in self.feature_names if any(
                x in f for x in ['line_', 'steam', 'sharp_money']
            )],
            'market_efficiency': [f for f in self.feature_names if any(
                x in f for x in ['efficiency', 'major_league', 'hours_until', 'early_bet']
            )],
            'streak_momentum': [f for f in self.feature_names if any(
                x in f for x in ['streak', 'momentum', 'hot_', 'cold_', 'variance']
            )],
            'bookmaker': [f for f in self.feature_names if any(
                x in f for x in ['bookmaker', 'institution', 'sharp_book', 'soft_book']
            )],
            'temporal': [f for f in self.feature_names if any(
                x in f for x in ['day_of', 'hour', 'weekend', 'primetime', 'morning',
                                'season', 'month', 'quarter', 'frequency', 'since_first']
            )],
            'player_props': [f for f in self.feature_names if any(
                x in f for x in ['prop', 'player', 'scoring', 'combo', 'volume']
            )],
            'cross_sport': [f for f in self.feature_names if any(
                x in f for x in ['ball_sport', 'individual_sport', 'expertise', 'consistency']
            )],
            'historical': [f for f in self.feature_names if any(
                x in f for x in ['expanding', 'rolling', 'trend', '_count']
            )],
            'categorical': [f for f in self.feature_names if f.endswith('_encoded')],
        }

        return groups


def select_features_for_model(
    df: pd.DataFrame,
    feature_engineer: AdvancedFeatureEngineer,
    include_categories: List[str] = None,
    exclude_categories: List[str] = None,
    min_importance: float = 0.0
) -> List[str]:
    """
    Select features for model training.

    Args:
        df: DataFrame with features
        feature_engineer: Fitted AdvancedFeatureEngineer
        include_categories: List of feature categories to include (None = all)
        exclude_categories: List of feature categories to exclude
        min_importance: Minimum feature importance threshold

    Returns:
        List of selected feature names
    """
    all_features = feature_engineer.get_feature_names()
    groups = feature_engineer.get_feature_importance_groups()

    if include_categories:
        selected = []
        for cat in include_categories:
            if cat in groups:
                selected.extend(groups[cat])
        all_features = list(set(selected))

    if exclude_categories:
        excluded = []
        for cat in exclude_categories:
            if cat in groups:
                excluded.extend(groups[cat])
        all_features = [f for f in all_features if f not in excluded]

    # Filter to features that exist in dataframe
    available_features = [f for f in all_features if f in df.columns]

    return available_features


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("PIKKIT ADVANCED FEATURE ENGINEERING MODULE")
    print("=" * 70)

    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000

    sample_data = {
        'id': range(n_samples),
        'sport': np.random.choice(['Basketball', 'American Football', 'Baseball', 'Ice Hockey'], n_samples),
        'league': np.random.choice(['NBA', 'NFL', 'MLB', 'NHL', 'NCAAB'], n_samples),
        'market': np.random.choice(['Spread', 'Moneyline', 'Total', 'Player Points', 'Player Rebounds'], n_samples),
        'institution_name': np.random.choice(['DraftKings', 'FanDuel', 'BetMGM', 'Caesars'], n_samples),
        'bet_type': np.random.choice(['straight', 'parlay'], n_samples, p=[0.9, 0.1]),
        'american_odds': np.random.choice([-110, -115, -105, 100, 105, 110, 120, 150], n_samples),
        'amount': np.random.uniform(10, 500, n_samples),
        'clv_percentage': np.where(np.random.random(n_samples) < 0.3,
                                   np.random.normal(0, 3, n_samples), np.nan),
        'is_live': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'is_win': np.random.choice([True, False], n_samples),
        'profit': np.random.normal(0, 50, n_samples),
        'created_at': pd.date_range('2024-01-01', periods=n_samples, freq='2H'),
    }
    sample_data['roi'] = sample_data['profit'] / sample_data['amount'] * 100

    df = pd.DataFrame(sample_data)

    print(f"\nSample data: {len(df)} bets")
    print(f"Columns: {list(df.columns)}")

    # Initialize and fit feature engineer
    print("\n" + "-" * 70)
    print("Fitting feature engineer...")

    fe = AdvancedFeatureEngineer(lookback_windows=[5, 10, 20, 50])
    df_features = fe.fit_transform(df)

    print(f"\nTotal features created: {len(fe.get_feature_names())}")

    # Show feature groups
    print("\n" + "-" * 70)
    print("Feature Groups:")
    groups = fe.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")

    # Show sample of features
    print("\n" + "-" * 70)
    print("Sample Feature Values (first 3 rows):")
    sample_features = fe.get_feature_names()[:10]
    print(df_features[sample_features].head(3).to_string())

    print("\n" + "=" * 70)
    print("Feature engineering complete!")
    print("=" * 70)
