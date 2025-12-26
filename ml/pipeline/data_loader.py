#!/usr/bin/env python3
"""
Data Loading for Pikkit ML Pipeline

Handles:
- Supabase data fetching
- Data validation
- Sample data generation for testing
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from .config import PipelineConfig, DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading from Supabase or sample generation.

    Features:
    - Paginated data fetching
    - Data filtering and validation
    - Sample data for testing
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize data loader.

        Args:
            config: Pipeline configuration
        """
        if config is None:
            config = PipelineConfig()

        self.config = config.data
        self._client = None

        if SUPABASE_AVAILABLE and self.config.supabase_key:
            self._init_supabase()

    def _init_supabase(self) -> None:
        """Initialize Supabase client"""
        try:
            self._client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            logger.info(f"Connected to Supabase: {self.config.supabase_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self._client = None

    def fetch_bets(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch betting data from Supabase.

        Args:
            limit: Maximum number of bets to fetch

        Returns:
            DataFrame with betting data
        """
        limit = limit or self.config.fetch_limit

        if self._client is None:
            logger.warning("Supabase not available, generating sample data")
            return self.generate_sample_data(n_samples=min(limit, 5000))

        logger.info(f"Fetching up to {limit} bets from Supabase...")

        all_bets = []
        offset = 0
        batch_size = self.config.batch_size

        while len(all_bets) < limit:
            try:
                # Build query with filters
                query = (self._client.table('bets')
                    .select('id,sport,league,market,institution_name,bet_type,'
                           'american_odds,amount,is_win,is_live,is_settled,'
                           'clv_percentage,profit,roi,created_at,updated_at'))

                # Apply filters
                filters = self.config.filters
                if filters.get('is_settled', True):
                    query = query.eq('is_settled', True)

                if filters.get('min_date'):
                    query = query.gte('created_at', filters['min_date'])

                if filters.get('max_date'):
                    query = query.lte('created_at', filters['max_date'])

                # Paginate
                response = query.range(offset, offset + batch_size - 1).execute()

                if not response.data:
                    break

                all_bets.extend(response.data)
                logger.debug(f"Fetched {len(all_bets)} bets...")

                if len(response.data) < batch_size:
                    break

                offset += batch_size

            except Exception as e:
                logger.error(f"Fetch error at offset {offset}: {e}")
                if all_bets:
                    break
                raise

        df = pd.DataFrame(all_bets[:limit])
        logger.info(f"Loaded {len(df)} bets from Supabase")

        return df

    def generate_sample_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate realistic sample betting data for testing.

        Args:
            n_samples: Number of sample bets to generate

        Returns:
            DataFrame with sample betting data
        """
        logger.info(f"Generating {n_samples} sample bets...")

        np.random.seed(42)

        sports = ['NBA', 'NFL', 'MLB', 'NHL', 'NCAAF', 'NCAAB']
        markets = ['Moneyline', 'Spread', 'Total', 'Player Props']
        bookmakers = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']
        bet_types = ['Over', 'Under', 'Home', 'Away']

        data = []
        base_date = datetime.now() - timedelta(days=365)

        for i in range(n_samples):
            sport = np.random.choice(sports)
            market = np.random.choice(markets)
            bookmaker = np.random.choice(bookmakers)

            # Generate odds (American format)
            odds = np.random.choice(
                [-110, -105, -115, -120, 100, 105, 110, 120, 130, 140, 150]
            )

            # Calculate implied probability
            if odds < 0:
                implied_prob = abs(odds) / (abs(odds) + 100)
            else:
                implied_prob = 100 / (odds + 100)

            # Generate outcome with skill factor
            true_prob = implied_prob + np.random.normal(0, 0.05)
            won = np.random.random() < max(0, min(1, true_prob))

            # CLV (available for ~20% of bets)
            has_clv = np.random.random() < 0.2
            clv_percentage = np.random.normal(0, 3) if has_clv else None

            # Calculate P&L
            wager = 100
            if odds < 0:
                to_win = wager / (abs(odds) / 100)
            else:
                to_win = wager * (odds / 100)

            profit = to_win if won else -wager
            roi = (profit / wager) * 100

            # Generate timestamp
            days_ago = np.random.randint(0, 365)
            created_at = base_date + timedelta(
                days=days_ago,
                hours=np.random.randint(0, 24)
            )

            data.append({
                'id': f'bet_{i}',
                'sport': sport,
                'league': sport,
                'market': market,
                'institution_name': bookmaker,
                'bet_type': np.random.choice(bet_types),
                'american_odds': odds,
                'implied_prob': implied_prob,
                'amount': wager,
                'is_win': won,
                'is_settled': True,
                'clv_percentage': clv_percentage,
                'is_live': np.random.choice([True, False], p=[0.1, 0.9]),
                'profit': profit,
                'roi': roi,
                'created_at': created_at.isoformat(),
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample bets")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data.

        Args:
            df: DataFrame to validate

        Returns:
            Validation report dictionary
        """
        report = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'missing_values': {},
            'value_ranges': {},
            'issues': []
        }

        # Check required columns
        required_cols = ['sport', 'market', 'american_odds', 'is_settled']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report['issues'].append(f"Missing required columns: {missing_cols}")

        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                report['missing_values'][col] = missing

        # Check value ranges
        if 'american_odds' in df.columns:
            odds = df['american_odds'].dropna()
            report['value_ranges']['odds'] = {
                'min': float(odds.min()),
                'max': float(odds.max()),
                'mean': float(odds.mean())
            }

        if 'roi' in df.columns:
            roi = df['roi'].dropna()
            report['value_ranges']['roi'] = {
                'min': float(roi.min()),
                'max': float(roi.max()),
                'mean': float(roi.mean())
            }

        # Check date range
        if 'created_at' in df.columns:
            dates = pd.to_datetime(df['created_at'])
            report['date_range'] = {
                'min': dates.min().isoformat(),
                'max': dates.max().isoformat()
            }

        # Check outcome balance
        if 'is_win' in df.columns:
            win_rate = df['is_win'].mean()
            report['win_rate'] = float(win_rate)
            if win_rate < 0.3 or win_rate > 0.7:
                report['issues'].append(
                    f"Unusual win rate: {win_rate:.2%}"
                )

        return report

    def split_by_time(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> tuple:
        """
        Split data by time for train/test.

        Args:
            df: DataFrame with 'created_at' column
            test_size: Fraction for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.copy()
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at').reset_index(drop=True)

        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Time-based split: {len(train_df)} train, {len(test_df)} test")

        return train_df, test_df
