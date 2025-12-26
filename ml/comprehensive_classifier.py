#!/usr/bin/env python3
"""
Comprehensive Multi-Field Classifier for Pikkit Betting Data
Predicts: Market, Sport, and League for betting data

This classifier trains on ALL bets in Supabase (~28,000) and can predict:
- Market type (79+ specific markets like "Player Passing Yards", "Pitcher Strikeouts")
- Sport (Basketball, American Football, Baseball, Tennis, etc.)
- League (NBA, NFL, MLB, WTA, etc.)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from supabase import create_client, Client
from scipy.sparse import hstack, csr_matrix


class ComprehensiveClassifier:
    """Multi-output classifier for market, sport, and league"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or '/root/pikkit/ml/models/comprehensive_classifier.joblib'

        # Shared text vectorizer (reduced features for memory efficiency)
        self.text_vectorizer = TfidfVectorizer(
            max_features=800,
            ngram_range=(1, 3),
            lowercase=True,
            strip_accents='unicode',
            min_df=2
        )

        # Label encoders for each output
        self.market_encoder = LabelEncoder()
        self.sport_encoder = LabelEncoder()
        self.league_encoder = LabelEncoder()

        # Separate classifier for each output (optimized for speed)
        self.market_classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.15,
            random_state=42,
            verbose=1
        )

        self.sport_classifier = GradientBoostingClassifier(
            n_estimators=40,
            max_depth=5,
            learning_rate=0.15,
            random_state=42,
            verbose=1
        )

        self.league_classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.15,
            random_state=42,
            verbose=1
        )

    def _create_features(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        """Engineer features from raw bet data"""
        features = df.copy()

        # Clean text fields
        features['market_clean'] = features['market'].fillna('').str.lower().str.strip()
        features['pick_name_clean'] = features['pick_name'].fillna('').str.lower().str.strip()
        features['sport_clean'] = features['sport'].fillna('').str.lower().str.strip()
        features['league_clean'] = features['league'].fillna('').str.lower().str.strip()

        # Combined text for TF-IDF (more comprehensive)
        features['combined_text'] = (
            features['market_clean'] + ' | ' +
            features['pick_name_clean'] + ' | ' +
            features['sport_clean'] + ' | ' +
            features['league_clean']
        )

        # Extract patterns from pick_name
        features['has_over'] = features['pick_name_clean'].str.contains('over', regex=False).astype(int)
        features['has_under'] = features['pick_name_clean'].str.contains('under', regex=False).astype(int)
        features['has_player_name'] = features['pick_name_clean'].str.contains(r'[a-z]\.\s[a-z]', regex=True).astype(int)
        features['has_numbers'] = features['pick_name_clean'].str.contains(r'\d+', regex=True).astype(int)
        features['has_plus_minus'] = features['pick_name_clean'].str.contains(r'[+-]\d', regex=True).astype(int)

        # Categorical features
        features['is_parlay'] = (features['bet_type'] == 'parlay').astype(int)
        features['picks_count'] = features['picks_count'].fillna(1).astype(int)
        features['is_live'] = features['is_live'].fillna(False).astype(int)

        # Period/Quarter indicators
        features['is_1st_half'] = features['market_clean'].str.contains('1st half', regex=False).astype(int)
        features['is_1st_quarter'] = features['market_clean'].str.contains('1st quarter|1st period', regex=True).astype(int)
        features['is_2nd_half'] = features['market_clean'].str.contains('2nd half', regex=False).astype(int)
        features['is_period'] = features['market_clean'].str.contains('period|inning|set', regex=True).astype(int)

        # Sport-specific keywords
        features['has_football_terms'] = features['pick_name_clean'].str.contains('passing|rushing|receiving|touchdown|td', regex=True).astype(int)
        features['has_baseball_terms'] = features['pick_name_clean'].str.contains('pitcher|strikeout|run|hit|inning', regex=True).astype(int)
        features['has_basketball_terms'] = features['pick_name_clean'].str.contains('points|rebounds|assists|pra|blocks', regex=True).astype(int)
        features['has_hockey_terms'] = features['pick_name_clean'].str.contains('goal|puck|period|shots', regex=True).astype(int)
        features['has_tennis_terms'] = features['pick_name_clean'].str.contains('set|game|ace|break', regex=True).astype(int)

        return features

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Convert features to model input format"""
        # TF-IDF features
        text_features = self.text_vectorizer.transform(features['combined_text'])

        # Numerical features
        numeric_cols = [
            'is_parlay', 'picks_count', 'is_live',
            'has_over', 'has_under', 'has_player_name', 'has_numbers', 'has_plus_minus',
            'is_1st_half', 'is_1st_quarter', 'is_2nd_half', 'is_period',
            'has_football_terms', 'has_baseball_terms', 'has_basketball_terms',
            'has_hockey_terms', 'has_tennis_terms'
        ]
        numeric_features = features[numeric_cols].values

        # Combine features
        X = hstack([text_features, numeric_features])
        return X

    def train(self, df: pd.DataFrame, min_samples_market: int = 5, min_samples_sport: int = 5, min_samples_league: int = 5) -> Dict:
        """Train all three classifiers"""

        print(f"Training on {len(df)} total bets...")
        print(f"\nData overview:")
        print(f"  Sports: {df['sport'].nunique()} unique")
        print(f"  Leagues: {df['league'].nunique()} unique")
        print(f"  Markets: {df['market'].nunique()} unique")

        # Filter each output separately based on minimum samples
        # For market
        market_counts = df['market'].value_counts()
        valid_markets = market_counts[market_counts >= min_samples_market].index
        df_market = df[df['market'].isin(valid_markets)].copy()
        print(f"\n[Market] Training on {len(df_market)} samples, {len(valid_markets)} market types")

        # For sport
        sport_counts = df['sport'].value_counts()
        valid_sports = sport_counts[sport_counts >= min_samples_sport].index
        df_sport = df[df['sport'].isin(valid_sports)].copy()
        print(f"[Sport] Training on {len(df_sport)} samples, {len(valid_sports)} sports")

        # For league
        league_counts = df['league'].value_counts()
        valid_leagues = league_counts[league_counts >= min_samples_league].index
        df_league = df[df['league'].isin(valid_leagues)].copy()
        print(f"[League] Training on {len(df_league)} samples, {len(valid_leagues)} leagues")

        # Create features (fit vectorizer on all data)
        print("\nCreating features...")
        features_all = self._create_features(df, training=True)
        self.text_vectorizer.fit(features_all['combined_text'])

        # Train Market Classifier
        print("\n" + "="*60)
        print("Training MARKET Classifier")
        print("="*60)
        features_market = self._create_features(df_market, training=False)
        X_market = self._prepare_features(features_market)
        y_market = self.market_encoder.fit_transform(df_market['market'])

        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_market, y_market, test_size=0.2, random_state=42
        )

        self.market_classifier.fit(X_train_m, y_train_m)
        market_score = self.market_classifier.score(X_test_m, y_test_m)
        print(f"Market Test Accuracy: {market_score:.4f}")

        # Train Sport Classifier
        print("\n" + "="*60)
        print("Training SPORT Classifier")
        print("="*60)
        features_sport = self._create_features(df_sport, training=False)
        X_sport = self._prepare_features(features_sport)
        y_sport = self.sport_encoder.fit_transform(df_sport['sport'])

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_sport, y_sport, test_size=0.2, random_state=42
        )

        self.sport_classifier.fit(X_train_s, y_train_s)
        sport_score = self.sport_classifier.score(X_test_s, y_test_s)
        print(f"Sport Test Accuracy: {sport_score:.4f}")

        # Show sport classification report
        y_pred_s = self.sport_classifier.predict(X_test_s)
        print("\nSport Classification Report:")
        # Get unique classes in test set to avoid mismatch
        unique_test_classes = sorted(set(y_test_s))
        test_class_names = self.sport_encoder.inverse_transform(unique_test_classes)
        print(classification_report(
            y_test_s,
            y_pred_s,
            labels=unique_test_classes,
            target_names=test_class_names,
            zero_division=0
        ))

        # Train League Classifier
        print("\n" + "="*60)
        print("Training LEAGUE Classifier")
        print("="*60)
        features_league = self._create_features(df_league, training=False)
        X_league = self._prepare_features(features_league)
        y_league = self.league_encoder.fit_transform(df_league['league'])

        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            X_league, y_league, test_size=0.2, random_state=42
        )

        self.league_classifier.fit(X_train_l, y_train_l)
        league_score = self.league_classifier.score(X_test_l, y_test_l)
        print(f"League Test Accuracy: {league_score:.4f}")

        return {
            'market_accuracy': market_score,
            'sport_accuracy': sport_score,
            'league_accuracy': league_score,
            'n_samples': len(df),
            'n_markets': len(valid_markets),
            'n_sports': len(valid_sports),
            'n_leagues': len(valid_leagues)
        }

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict market, sport, and league for unlabeled data"""
        features = self._create_features(df, training=False)
        X = self._prepare_features(features)

        # Predict each field
        market_pred = self.market_encoder.inverse_transform(self.market_classifier.predict(X))
        market_proba = self.market_classifier.predict_proba(X).max(axis=1)

        sport_pred = self.sport_encoder.inverse_transform(self.sport_classifier.predict(X))
        sport_proba = self.sport_classifier.predict_proba(X).max(axis=1)

        league_pred = self.league_encoder.inverse_transform(self.league_classifier.predict(X))
        league_proba = self.league_classifier.predict_proba(X).max(axis=1)

        # Create results dataframe
        predictions = pd.DataFrame({
            'predicted_market': market_pred,
            'market_confidence': market_proba,
            'predicted_sport': sport_pred,
            'sport_confidence': sport_proba,
            'predicted_league': league_pred,
            'league_confidence': league_proba
        })

        confidences = pd.DataFrame({
            'market_confidence': market_proba,
            'sport_confidence': sport_proba,
            'league_confidence': league_proba
        })

        return predictions, confidences

    def save(self):
        """Save all models to disk"""
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'text_vectorizer': self.text_vectorizer,
            'market_encoder': self.market_encoder,
            'sport_encoder': self.sport_encoder,
            'league_encoder': self.league_encoder,
            'market_classifier': self.market_classifier,
            'sport_classifier': self.sport_classifier,
            'league_classifier': self.league_classifier
        }, self.model_path)
        print(f"\nModel saved to {self.model_path}")

    def load(self):
        """Load all models from disk"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        saved = joblib.load(self.model_path)
        self.text_vectorizer = saved['text_vectorizer']
        self.market_encoder = saved['market_encoder']
        self.sport_encoder = saved['sport_encoder']
        self.league_encoder = saved['league_encoder']
        self.market_classifier = saved['market_classifier']
        self.sport_classifier = saved['sport_classifier']
        self.league_classifier = saved['league_classifier']
        print(f"Model loaded from {self.model_path}")


def get_supabase_client() -> Client:
    """Create Supabase client from environment"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    return create_client(url, key)


def fetch_all_bets(supabase: Client) -> pd.DataFrame:
    """Fetch ALL bets from Supabase for comprehensive training"""

    print("Fetching ALL bets from Supabase...")
    all_data = []

    # Fetch in batches of 1000 to avoid timeout
    batch_size = 1000
    offset = 0

    while True:
        print(f"Fetching batch {offset // batch_size + 1} (offset: {offset})...")

        response = supabase.table('bets').select(
            'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
        ).order('id').range(offset, offset + batch_size - 1).execute()

        if not response.data:
            break

        all_data.extend(response.data)
        print(f"  Fetched {len(response.data)} bets (total: {len(all_data)})")

        if len(response.data) < batch_size:
            break

        offset += batch_size

    df = pd.DataFrame(all_data)

    print(f"\nTotal fetched (with any duplicates): {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")

    df = df.drop_duplicates(subset=['id'])
    print(f"After deduplication: {len(df)} bets")

    # Filter to only bets with all three fields populated (for training)
    df_complete = df[
        df['market'].notna() &
        df['sport'].notna() &
        df['league'].notna()
    ].copy()

    print(f"Complete bets (all fields): {len(df_complete)}")

    return df_complete


def main():
    """Main training pipeline"""

    from dotenv import load_dotenv
    load_dotenv('/root/pikkit/.env')

    # Initialize
    classifier = ComprehensiveClassifier()

    # Load training data from CSV (already fetched with pagination)
    print("Loading training data from CSV...")
    train_df = pd.read_csv('/root/pikkit/ml/data/all_bets_training.csv')
    print(f"Loaded {len(train_df)} bets from CSV")

    # Train model
    metrics = classifier.train(train_df, min_samples_market=5, min_samples_sport=5, min_samples_league=5)

    # Save model
    classifier.save()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Market accuracy: {metrics['market_accuracy']:.4f} ({metrics['n_markets']} types)")
    print(f"Sport accuracy: {metrics['sport_accuracy']:.4f} ({metrics['n_sports']} types)")
    print(f"League accuracy: {metrics['league_accuracy']:.4f} ({metrics['n_leagues']} types)")
    print(f"Total samples: {metrics['n_samples']}")


if __name__ == '__main__':
    main()
