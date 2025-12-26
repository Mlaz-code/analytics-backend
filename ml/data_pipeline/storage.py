#!/usr/bin/env python3
"""
Pikkit Storage Architecture
Three-layer storage: Raw, Processed, Feature
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List
import os
import json


class StorageLayer:
    """Base class for storage layers"""

    def __init__(self, name: str, base_path: str):
        self.name = name
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save(self, df: pd.DataFrame, key: str):
        raise NotImplementedError

    def load(self, key: str) -> pd.DataFrame:
        raise NotImplementedError


class RawLayer(StorageLayer):
    """
    Raw data layer - mirrors Supabase bets table
    Stores snapshots for audit/debugging
    """

    def __init__(self, supabase_client=None, base_path: str = '/root/pikkit/ml/data/raw'):
        super().__init__('raw', base_path)
        self.supabase = supabase_client

    def fetch_from_source(self, limit: int = None) -> pd.DataFrame:
        """Fetch raw data from Supabase"""
        if self.supabase is None:
            print("Warning: No Supabase client configured")
            return pd.DataFrame()

        all_records = []
        offset = 0
        batch_size = 1000

        while True:
            query = self.supabase.table('bets').select('*')
            query = query.range(offset, offset + batch_size - 1)

            try:
                response = query.execute()
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

            if not response.data:
                break

            all_records.extend(response.data)

            if len(response.data) < batch_size:
                break

            offset += batch_size

            if limit and len(all_records) >= limit:
                all_records = all_records[:limit]
                break

        print(f"Fetched {len(all_records)} records from Supabase")
        return pd.DataFrame(all_records)

    def save_snapshot(self, df: pd.DataFrame, tag: str = None) -> str:
        """Save a snapshot of raw data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"snapshot_{timestamp}" if not tag else f"snapshot_{tag}_{timestamp}"

        filepath = f"{self.base_path}/{key}.parquet"
        df.to_parquet(filepath, index=False)

        # Save metadata
        meta = {
            'timestamp': timestamp,
            'records': len(df),
            'columns': list(df.columns),
            'sports': df['sport'].value_counts().to_dict() if 'sport' in df.columns else {}
        }
        with open(f"{self.base_path}/{key}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Saved raw snapshot: {filepath}")
        return filepath

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """Load a raw snapshot"""
        filepath = f"{self.base_path}/{key}.parquet"
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def list_snapshots(self) -> List[str]:
        """List all saved snapshots"""
        import glob
        files = glob.glob(f"{self.base_path}/snapshot_*.parquet")
        return [os.path.basename(f).replace('.parquet', '') for f in sorted(files, reverse=True)]


class ProcessedLayer(StorageLayer):
    """
    Processed data layer - cleaned and validated
    Partitioned by sport for efficient access
    """

    def __init__(self, base_path: str = '/root/pikkit/ml/data/processed'):
        super().__init__('processed', base_path)

    def process_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to processed format"""
        df = raw_df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Parse timestamps
        for col in ['created_at', 'updated_at', 'time_placed']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Ensure numeric types
        for col in ['odds', 'american_odds', 'amount', 'profit', 'roi', 'clv_percentage']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute derived fields if missing
        if 'is_settled' not in df.columns and 'status' in df.columns:
            df['is_settled'] = df['status'].str.startswith('SETTLED_', na=False)

        if 'is_win' not in df.columns and 'status' in df.columns:
            df['is_win'] = df['status'] == 'SETTLED_WIN'

        if 'roi' not in df.columns and 'profit' in df.columns and 'amount' in df.columns:
            df['roi'] = (df['profit'] / df['amount'] * 100).where(df['amount'] > 0, 0)

        print(f"Processed {len(df)} records")
        return df

    def save(self, df: pd.DataFrame, partition_key: str = 'all') -> str:
        """Save processed data"""
        filepath = f"{self.base_path}/{partition_key}.parquet"
        df.to_parquet(filepath, index=False)
        print(f"Saved processed data: {filepath}")
        return filepath

    def load(self, partition_key: str = 'all') -> Optional[pd.DataFrame]:
        """Load processed data"""
        filepath = f"{self.base_path}/{partition_key}.parquet"
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_by_sport(self, df: pd.DataFrame):
        """Partition and save by sport"""
        for sport in df['sport'].dropna().unique():
            sport_df = df[df['sport'] == sport]
            key = sport.lower().replace(' ', '_')
            self.save(sport_df, f"sport_{key}")

    def list_partitions(self) -> List[str]:
        """List all saved partitions"""
        import glob
        files = glob.glob(f"{self.base_path}/*.parquet")
        return [os.path.basename(f).replace('.parquet', '') for f in sorted(files)]


class FeatureLayer(StorageLayer):
    """
    Feature layer - aggregated stats ready for ML
    Includes both Supabase tables and local cache
    """

    def __init__(self, supabase_client=None,
                 base_path: str = '/root/pikkit/ml/data/features'):
        super().__init__('feature', base_path)
        self.supabase = supabase_client

    def compute_sport_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sport-level aggregated features"""
        if 'is_settled' not in df.columns:
            print("Warning: is_settled column missing")
            return pd.DataFrame()

        settled = df[df['is_settled'] == True]
        if len(settled) == 0:
            return pd.DataFrame()

        features = settled.groupby('sport').agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
            'roi': 'mean'
        }).reset_index()

        features.columns = ['sport', 'wins', 'total_bets', 'total_amount',
                           'total_profit', 'avg_roi']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)
        features['updated_at'] = datetime.utcnow().isoformat()

        return features

    def compute_sport_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sport+market aggregated features"""
        if 'is_settled' not in df.columns:
            return pd.DataFrame()

        settled = df[df['is_settled'] == True]
        if len(settled) == 0:
            return pd.DataFrame()

        features = settled.groupby(['sport', 'market']).agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
        }).reset_index()

        features.columns = ['sport', 'market', 'wins', 'total_bets',
                           'total_amount', 'total_profit']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)

        # Recent win rate (last 50 bets per sport/market)
        recent = settled.groupby(['sport', 'market']).tail(50)
        recent_wr = recent.groupby(['sport', 'market'])['is_win'].mean().reset_index()
        recent_wr.columns = ['sport', 'market', 'recent_win_rate']

        features = features.merge(recent_wr, on=['sport', 'market'], how='left')
        features['recent_win_rate'] = features['recent_win_rate'].fillna(0.5)
        features['updated_at'] = datetime.utcnow().isoformat()

        return features

    def compute_institution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute institution (sportsbook) aggregated features"""
        if 'is_settled' not in df.columns:
            return pd.DataFrame()

        settled = df[df['is_settled'] == True]
        if len(settled) == 0:
            return pd.DataFrame()

        features = settled.groupby('institution_name').agg({
            'is_win': ['sum', 'count'],
            'amount': 'sum',
            'profit': 'sum',
        }).reset_index()

        features.columns = ['institution_name', 'wins', 'total_bets',
                           'total_amount', 'total_profit']
        features['win_rate'] = features['wins'] / features['total_bets']
        features['roi'] = (features['total_profit'] / features['total_amount'] * 100
                          ).where(features['total_amount'] > 0, 0)
        features['updated_at'] = datetime.utcnow().isoformat()

        return features

    def save(self, df: pd.DataFrame, key: str) -> str:
        """Save feature data locally"""
        filepath = f"{self.base_path}/{key}.parquet"
        df.to_parquet(filepath, index=False)
        return filepath

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """Load feature data"""
        filepath = f"{self.base_path}/{key}.parquet"
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def save_to_supabase(self, df: pd.DataFrame, table: str):
        """Upsert feature data to Supabase"""
        if self.supabase is None:
            print(f"Warning: No Supabase client, saving to local only")
            return

        records = df.to_dict('records')

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            try:
                self.supabase.table(table).upsert(batch).execute()
            except Exception as e:
                print(f"Error upserting to {table}: {e}")

    def refresh_all_features(self, processed_df: pd.DataFrame):
        """Refresh all feature tables"""
        print("Refreshing feature layer...")

        # Sport features
        sport_features = self.compute_sport_features(processed_df)
        if len(sport_features) > 0:
            self.save(sport_features, 'sport_stats')
            print(f"  Sport features: {len(sport_features)} rows")

        # Sport+market features
        sport_market_features = self.compute_sport_market_features(processed_df)
        if len(sport_market_features) > 0:
            self.save(sport_market_features, 'sport_market_stats')
            print(f"  Sport+Market features: {len(sport_market_features)} rows")

        # Institution features
        institution_features = self.compute_institution_features(processed_df)
        if len(institution_features) > 0:
            self.save(institution_features, 'institution_stats')
            print(f"  Institution features: {len(institution_features)} rows")

        # Save to Supabase if available
        if self.supabase:
            if len(sport_features) > 0:
                self.save_to_supabase(sport_features, 'feature_sport_stats')
            if len(sport_market_features) > 0:
                self.save_to_supabase(sport_market_features, 'feature_sport_market_stats')
            if len(institution_features) > 0:
                self.save_to_supabase(institution_features, 'feature_institution_stats')


class DataPipelineOrchestrator:
    """
    Orchestrate the full data pipeline across all layers
    """

    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self.raw = RawLayer(supabase_client)
        self.processed = ProcessedLayer()
        self.features = FeatureLayer(supabase_client)

    def run_full_pipeline(self, save_snapshot: bool = True) -> Dict:
        """Run complete ETL pipeline"""
        stats = {
            'started_at': datetime.now().isoformat(),
            'layers': {}
        }

        # Layer 1: Raw
        print("=" * 60)
        print("LAYER 1: RAW")
        print("=" * 60)
        raw_df = self.raw.fetch_from_source()
        stats['layers']['raw'] = {'records': len(raw_df)}

        if len(raw_df) == 0:
            print("No data fetched, aborting pipeline")
            return stats

        if save_snapshot:
            snapshot_path = self.raw.save_snapshot(raw_df)
            stats['layers']['raw']['snapshot'] = snapshot_path

        # Layer 2: Processed
        print("\n" + "=" * 60)
        print("LAYER 2: PROCESSED")
        print("=" * 60)
        processed_df = self.processed.process_raw(raw_df)
        self.processed.save(processed_df, 'all')
        self.processed.save_by_sport(processed_df)
        stats['layers']['processed'] = {
            'records': len(processed_df),
            'settled': int(processed_df['is_settled'].sum()) if 'is_settled' in processed_df.columns else 0,
            'sports': int(processed_df['sport'].nunique()) if 'sport' in processed_df.columns else 0
        }

        # Layer 3: Features
        print("\n" + "=" * 60)
        print("LAYER 3: FEATURES")
        print("=" * 60)
        self.features.refresh_all_features(processed_df)
        stats['layers']['features'] = {
            'sport_stats': len(self.features.compute_sport_features(processed_df)),
            'sport_market_stats': len(self.features.compute_sport_market_features(processed_df)),
        }

        stats['completed_at'] = datetime.now().isoformat()

        # Save pipeline stats
        with open('/root/pikkit/ml/data_pipeline/last_run.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Raw records: {stats['layers']['raw']['records']}")
        print(f"Processed records: {stats['layers']['processed']['records']}")
        print(f"Sports: {stats['layers']['processed']['sports']}")

        return stats


if __name__ == '__main__':
    # Test storage layers
    import numpy as np

    print("Testing storage layers...")

    # Create test data
    test_df = pd.DataFrame({
        'id': [f'bet_{i}' for i in range(100)],
        'sport': np.random.choice(['Basketball', 'Football', 'Baseball'], 100),
        'market': np.random.choice(['Spread', 'Moneyline', 'Total'], 100),
        'institution_name': np.random.choice(['DraftKings', 'FanDuel'], 100),
        'american_odds': np.random.choice([-110, 100, 120], 100),
        'amount': np.random.uniform(10, 500, 100),
        'profit': np.random.uniform(-100, 200, 100),
        'status': np.random.choice(['SETTLED_WIN', 'SETTLED_LOSS', 'PENDING'], 100),
        'created_at': pd.date_range('2024-01-01', periods=100, freq='H'),
        'updated_at': pd.date_range('2024-01-01', periods=100, freq='H'),
    })

    # Test ProcessedLayer
    processed = ProcessedLayer()
    processed_df = processed.process_raw(test_df)
    print(f"\nProcessed columns: {list(processed_df.columns)}")
    print(f"is_settled: {processed_df['is_settled'].sum()} / {len(processed_df)}")
    print(f"is_win: {processed_df['is_win'].sum()} / {len(processed_df)}")

    # Test FeatureLayer
    features = FeatureLayer()
    sport_features = features.compute_sport_features(processed_df)
    print(f"\nSport features:\n{sport_features}")

    sport_market_features = features.compute_sport_market_features(processed_df)
    print(f"\nSport+Market features (first 5):\n{sport_market_features.head()}")
