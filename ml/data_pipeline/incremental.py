#!/usr/bin/env python3
"""
Pikkit Incremental Data Loading
CDC (Change Data Capture) strategy for efficient data refresh
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json
import os


class IncrementalLoader:
    """
    Incremental data loading using CDC (Change Data Capture)
    Tracks last sync timestamp to only fetch changed records
    """

    def __init__(self, supabase_client=None, state_file: str = '/root/pikkit/ml/data_pipeline/sync_state.json'):
        self.supabase = supabase_client
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load sync state from file"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'last_sync_timestamp': None,
                'last_sync_count': 0,
                'total_synced': 0,
                'partitions': {}
            }

    def _save_state(self):
        """Persist sync state to file"""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_changes_since(self, last_timestamp: str = None,
                          table: str = 'bets',
                          batch_size: int = 1000) -> pd.DataFrame:
        """
        Fetch records changed since last timestamp
        Uses updated_at for CDC
        """
        if self.supabase is None:
            print("Warning: No Supabase client configured")
            return pd.DataFrame()

        if last_timestamp is None:
            last_timestamp = self.state.get('last_sync_timestamp')

        all_records = []
        offset = 0

        while True:
            query = self.supabase.table(table).select('*')

            if last_timestamp:
                query = query.gt('updated_at', last_timestamp)

            query = query.order('updated_at').range(offset, offset + batch_size - 1)

            try:
                response = query.execute()
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

            if not response.data:
                break

            all_records.extend(response.data)
            print(f"  Fetched {len(all_records)} records...")

            if len(response.data) < batch_size:
                break

            offset += batch_size

        df = pd.DataFrame(all_records)

        if len(df) > 0:
            # Update state
            max_timestamp = df['updated_at'].max()
            self.state['last_sync_timestamp'] = max_timestamp
            self.state['last_sync_count'] = len(df)
            self.state['total_synced'] = self.state.get('total_synced', 0) + len(df)
            self._save_state()

        return df

    def get_partition(self, sport: str = None,
                      date_start: str = None,
                      date_end: str = None) -> pd.DataFrame:
        """
        Fetch data by partition (sport/date range)
        Efficient for targeted queries
        """
        if self.supabase is None:
            return pd.DataFrame()

        query = self.supabase.table('bets').select('*')

        if sport:
            query = query.eq('sport', sport)

        if date_start:
            query = query.gte('created_at', date_start)

        if date_end:
            query = query.lte('created_at', date_end)

        try:
            response = query.execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error fetching partition: {e}")
            return pd.DataFrame()

    def sync_partition(self, sport: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Sync a specific sport partition
        Returns new/updated records and sync stats
        """
        partition_key = f"sport:{sport}"
        last_sync = self.state.get('partitions', {}).get(partition_key)

        query = self.supabase.table('bets').select('*').eq('sport', sport)

        if last_sync:
            query = query.gt('updated_at', last_sync)

        query = query.order('updated_at')

        try:
            response = query.execute()
            df = pd.DataFrame(response.data)
        except Exception as e:
            print(f"Error syncing partition {sport}: {e}")
            df = pd.DataFrame()

        stats = {
            'sport': sport,
            'records_synced': len(df),
            'last_sync': last_sync,
            'new_sync': df['updated_at'].max() if len(df) > 0 else last_sync
        }

        # Update partition state
        if len(df) > 0:
            if 'partitions' not in self.state:
                self.state['partitions'] = {}
            self.state['partitions'][partition_key] = df['updated_at'].max()
            self._save_state()

        return df, stats

    def full_refresh(self, table: str = 'bets') -> pd.DataFrame:
        """
        Full table refresh (use sparingly)
        Resets sync state
        """
        if self.supabase is None:
            return pd.DataFrame()

        print(f"Performing full refresh of {table}...")

        all_records = []
        offset = 0
        batch_size = 1000

        while True:
            try:
                response = (self.supabase.table(table)
                           .select('*')
                           .range(offset, offset + batch_size - 1)
                           .execute())
            except Exception as e:
                print(f"Error during full refresh: {e}")
                break

            if not response.data:
                break

            all_records.extend(response.data)
            print(f"  Fetched {len(all_records)} records...")

            if len(response.data) < batch_size:
                break

            offset += batch_size

        df = pd.DataFrame(all_records)

        # Reset state
        self.state = {
            'last_sync_timestamp': df['updated_at'].max() if len(df) > 0 else None,
            'last_sync_count': len(df),
            'total_synced': len(df),
            'partitions': {}
        }
        self._save_state()

        print(f"Full refresh complete: {len(df)} records")
        return df

    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        return {
            'last_sync': self.state.get('last_sync_timestamp'),
            'records_synced': self.state.get('last_sync_count', 0),
            'total_synced': self.state.get('total_synced', 0),
            'partitions': list(self.state.get('partitions', {}).keys())
        }


class PartitionManager:
    """
    Manage data partitions by sport and date
    Enables efficient incremental processing
    """

    SPORTS = [
        'American Football', 'Basketball', 'Baseball', 'Ice Hockey',
        'Soccer', 'Tennis', 'MMA', 'Golf', 'Cricket'
    ]

    def __init__(self, data_dir: str = '/root/pikkit/ml/data/partitions'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def partition_by_sport(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split dataframe by sport"""
        partitions = {}
        for sport in df['sport'].dropna().unique():
            partitions[sport] = df[df['sport'] == sport].copy()
        return partitions

    def partition_by_date(self, df: pd.DataFrame,
                          freq: str = 'M') -> Dict[str, pd.DataFrame]:
        """
        Split dataframe by date period
        freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
        """
        df = df.copy()
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['partition_date'] = df['created_at'].dt.to_period(freq)

        partitions = {}
        for period in df['partition_date'].dropna().unique():
            key = str(period)
            partitions[key] = df[df['partition_date'] == period].copy()

        return partitions

    def save_partition(self, df: pd.DataFrame, partition_key: str):
        """Save partition to parquet file"""
        safe_key = partition_key.replace('/', '_').replace(' ', '_')
        filepath = f"{self.data_dir}/{safe_key}.parquet"
        df.to_parquet(filepath, index=False)
        return filepath

    def load_partition(self, partition_key: str) -> Optional[pd.DataFrame]:
        """Load partition from parquet file"""
        safe_key = partition_key.replace('/', '_').replace(' ', '_')
        filepath = f"{self.data_dir}/{safe_key}.parquet"
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    def list_partitions(self) -> List[str]:
        """List all saved partitions"""
        import glob
        files = glob.glob(f"{self.data_dir}/*.parquet")
        return [os.path.basename(f).replace('.parquet', '') for f in files]

    def get_partition_stats(self) -> List[Dict]:
        """Get statistics for all partitions"""
        import glob

        stats = []
        for filepath in glob.glob(f"{self.data_dir}/*.parquet"):
            try:
                df = pd.read_parquet(filepath)
                filename = os.path.basename(filepath)
                stats.append({
                    'partition': filename.replace('.parquet', ''),
                    'records': len(df),
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                    'date_range': f"{df['created_at'].min()} to {df['created_at'].max()}" if 'created_at' in df.columns else 'N/A'
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        return stats


if __name__ == '__main__':
    # Test incremental loading
    print("Testing IncrementalLoader...")

    loader = IncrementalLoader()
    print(f"Sync status: {loader.get_sync_status()}")

    print("\nTesting PartitionManager...")
    pm = PartitionManager()
    print(f"Existing partitions: {pm.list_partitions()}")

    # Create test dataframe
    import numpy as np
    test_df = pd.DataFrame({
        'id': [f'bet_{i}' for i in range(100)],
        'sport': np.random.choice(['Basketball', 'Football', 'Baseball'], 100),
        'created_at': pd.date_range('2024-01-01', periods=100, freq='H'),
        'amount': np.random.uniform(10, 500, 100),
    })

    # Test partitioning
    sport_partitions = pm.partition_by_sport(test_df)
    print(f"\nSport partitions: {list(sport_partitions.keys())}")
    for sport, df in sport_partitions.items():
        print(f"  {sport}: {len(df)} records")

    date_partitions = pm.partition_by_date(test_df, freq='W')
    print(f"\nDate partitions (weekly): {len(date_partitions)} partitions")
