#!/usr/bin/env python3
"""
Granular Market Classifier for Pikkit Betting Data
Classifies bets into specific market types (100+ categories) like:
- "Player Passing Yards", "1st Quarter Spread", "Pitcher Strikeouts", etc.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from supabase import create_client, Client
from collections import Counter


class GranularMarketClassifier:
    """Classify bets into specific market types (sport-aware)"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or '/root/pikkit/ml/models/granular_market_classifier.joblib'
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 4),
            lowercase=True,
            strip_accents='unicode',
            min_df=2
        )
        self.sport_encoder = LabelEncoder()
        self.market_encoder = LabelEncoder()

        # Use Gradient Boosting for better performance on multi-class
        self.classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )

        self.market_mapping = {}  # Map predictions to canonical names

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw bet data"""
        features = df.copy()

        # Text features
        features['market_clean'] = features['market'].fillna('').str.lower().str.strip()
        features['pick_name_clean'] = features['pick_name'].fillna('').str.lower().str.strip()
        features['sport_clean'] = features['sport'].fillna('').str.lower().str.strip()
        features['league_clean'] = features['league'].fillna('').str.lower().str.strip()

        # Combined text for TF-IDF
        features['combined_text'] = (
            features['market_clean'] + ' | ' +
            features['pick_name_clean'] + ' | ' +
            features['sport_clean'] + ' | ' +
            features['league_clean']
        )

        # Extract key patterns from pick_name
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
        features['is_1st_quarter'] = features['market_clean'].str.contains('1st quarter', regex=False).astype(int)
        features['is_2nd_half'] = features['market_clean'].str.contains('2nd half', regex=False).astype(int)
        features['is_period'] = features['market_clean'].str.contains('period|inning', regex=True).astype(int)

        return features

    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels"""
        # TF-IDF features from combined text
        text_features = self.text_vectorizer.fit_transform(features['combined_text'])

        # Encode sport as categorical feature
        sport_encoded = self.sport_encoder.fit_transform(features['sport_clean'])

        # Numerical features
        numeric_cols = [
            'is_parlay', 'picks_count', 'is_live',
            'has_over', 'has_under', 'has_player_name', 'has_numbers', 'has_plus_minus',
            'is_1st_half', 'is_1st_quarter', 'is_2nd_half', 'is_period'
        ]
        numeric_features = features[numeric_cols].values

        # Combine features
        from scipy.sparse import hstack, csr_matrix
        sport_sparse = csr_matrix(sport_encoded.reshape(-1, 1))
        X = hstack([text_features, sport_sparse, numeric_features])

        # Encode target labels
        y = self.market_encoder.fit_transform(features['market'])

        return X, y, sport_encoded

    def train(self, df: pd.DataFrame, min_samples: int = 5) -> Dict:
        """Train the classifier on labeled data"""

        # Filter markets with sufficient samples
        market_counts = df['market'].value_counts()
        valid_markets = market_counts[market_counts >= min_samples].index
        df_filtered = df[df['market'].isin(valid_markets)].copy()

        print(f"Training on {len(df_filtered)} samples across {len(valid_markets)} market types...")
        print(f"Min samples per market: {min_samples}")
        print(f"\nTop 20 markets:")
        print(df_filtered['market'].value_counts().head(20))

        # Create features
        features = self._create_features(df_filtered)

        # Prepare data
        X, y, _ = self._prepare_training_data(features)

        # Split data with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            print("Warning: Stratified split failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Train model
        print("\nTraining Gradient Boosting classifier...")
        self.classifier.fit(X_train, y_train)

        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)

        print(f"\nTrain accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")

        # Detailed metrics (top 20 classes only to avoid huge output)
        y_pred = self.classifier.predict(X_test)

        # Get top 20 most common classes in test set
        test_classes = pd.Series(y_test).value_counts().head(20).index.tolist()
        test_class_names = self.market_encoder.inverse_transform(test_classes)

        print("\nClassification Report (Top 20 markets):")
        print(classification_report(
            y_test,
            y_pred,
            labels=test_classes,
            target_names=test_class_names,
            zero_division=0
        ))

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': len(df_filtered),
            'n_markets': len(valid_markets)
        }

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict market types for unlabeled data"""
        features = self._create_features(df)

        # Fill missing sports with 'unknown'
        features['sport_clean'] = features['sport_clean'].replace('', 'unknown')

        # Transform using fitted vectorizer
        text_features = self.text_vectorizer.transform(features['combined_text'])

        # Encode sport (handle unseen sports)
        sport_encoded = []
        for sport in features['sport_clean']:
            try:
                sport_encoded.append(self.sport_encoder.transform([sport])[0])
            except ValueError:
                # Unseen sport - use most common sport from training
                sport_encoded.append(self.sport_encoder.transform(['basketball'])[0])
        sport_encoded = np.array(sport_encoded)

        # Numerical features
        numeric_cols = [
            'is_parlay', 'picks_count', 'is_live',
            'has_over', 'has_under', 'has_player_name', 'has_numbers', 'has_plus_minus',
            'is_1st_half', 'is_1st_quarter', 'is_2nd_half', 'is_period'
        ]
        numeric_features = features[numeric_cols].values

        # Combine features
        from scipy.sparse import hstack, csr_matrix
        sport_sparse = csr_matrix(sport_encoded.reshape(-1, 1))
        X = hstack([text_features, sport_sparse, numeric_features])

        # Predict
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)

        # Decode predictions
        predictions = self.market_encoder.inverse_transform(y_pred)
        probabilities = y_proba.max(axis=1)

        return predictions, probabilities

    def save(self):
        """Save model to disk"""
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'text_vectorizer': self.text_vectorizer,
            'sport_encoder': self.sport_encoder,
            'market_encoder': self.market_encoder,
            'classifier': self.classifier
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """Load model from disk"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        saved_model = joblib.load(self.model_path)
        self.text_vectorizer = saved_model['text_vectorizer']
        self.sport_encoder = saved_model['sport_encoder']
        self.market_encoder = saved_model['market_encoder']
        self.classifier = saved_model['classifier']
        print(f"Model loaded from {self.model_path}")


def get_supabase_client() -> Client:
    """Create Supabase client from environment"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    return create_client(url, key)


def fetch_training_data(supabase: Client, limit_per_sport: int = 3000) -> pd.DataFrame:
    """Fetch well-classified bets for training using batched approach"""

    print(f"Fetching training data from Supabase...")
    all_data = []

    # Fetch samples from each major sport
    sports = ['Basketball', 'American Football', 'Baseball', 'Tennis', 'Ice Hockey', 'Soccer']

    for sport in sports:
        print(f"Fetching {sport} samples...")
        response = supabase.table('bets').select(
            'id,market,pick_name,sport,league,bet_type,picks_count,is_live'
        ).eq('sport', sport).not_.is_('market', 'null').limit(limit_per_sport).execute()

        all_data.extend(response.data)
        print(f"  Fetched {len(response.data)} {sport} samples")

    df = pd.DataFrame(all_data)

    # Remove duplicates
    df = df.drop_duplicates(subset=['id'])

    print(f"Total fetched: {len(df)} training samples")

    return df


def main():
    """Main training pipeline"""

    # Load environment
    from dotenv import load_dotenv
    load_dotenv('/root/pikkit/.env')

    # Initialize
    supabase = get_supabase_client()
    classifier = GranularMarketClassifier()

    # Fetch training data
    train_df = fetch_training_data(supabase)

    # Train model
    metrics = classifier.train(train_df, min_samples=5)

    # Save model
    classifier.save()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Market types: {metrics['n_markets']}")
    print(f"Model saved to: {classifier.model_path}")


if __name__ == '__main__':
    main()
