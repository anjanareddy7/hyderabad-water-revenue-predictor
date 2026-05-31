import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS    = Path(__file__).resolve().parents[1] / "models"


# ── Features used for anomaly detection ──────────────────────────────────────
# These capture the billing pattern of a section-category pair in a given month
# We use relative features (ratios, growth rates) not absolute amounts
# so the model works across sections of very different sizes

ANOMALY_FEATURES = [
    'demand_growth_rate',      # how much demand changed vs last month
    'noofcans_growth',         # how much connection count changed
    'demand_per_can',          # average bill per connection
    'rolling_3m_efficiency_std',  # how volatile efficiency has been
    'lag_1_efficiency',        # last month efficiency
]


def load_training_data() -> pd.DataFrame:
    """Load feature matrix, filter to training period only."""
    df = pd.read_parquet(PROCESSED / "billing_clean.parquet")
    features = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data" / "features" / "feature_matrix.parquet"
    )
    # Use 2022-2024 training period only to avoid leakage
    train = features[features['year'] <= 2024].copy()
    train = train.dropna(subset=ANOMALY_FEATURES)
    logger.info(f"Training anomaly detector on {len(train):,} rows")
    return train


def train_anomaly_detector(contamination: float = 0.05) -> tuple:
    """
    Train an Isolation Forest on the training period feature matrix.
    contamination=0.05 means we expect ~5% of rows to be anomalous.
    Returns fitted model and scaler.
    """
    train = load_training_data()

    X = train[ANOMALY_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Evaluate on training data
    labels = model.predict(X_scaled)
    n_anomalies = (labels == -1).sum()
    logger.info(
        f"Anomaly detector trained — flagged {n_anomalies:,} anomalies "
        f"({n_anomalies/len(train)*100:.1f}%) in training data"
    )

    # Save artifacts
    joblib.dump(model,  MODELS / "anomaly_detector.pkl")
    joblib.dump(scaler, MODELS / "anomaly_scaler.pkl")
    logger.info("Saved: models/anomaly_detector.pkl, anomaly_scaler.pkl")

    return model, scaler


def score_anomaly(features_dict: dict) -> dict:
    """
    Score a single observation for anomaly.
    Returns anomaly_score (lower = more anomalous) and is_anomaly flag.
    Isolation Forest returns -1 for anomalies and 1 for normal points.
    score_samples returns negative average path length — more negative = more anomalous.
    """
    model  = joblib.load(MODELS / "anomaly_detector.pkl")
    scaler = joblib.load(MODELS / "anomaly_scaler.pkl")

    values = np.array([[
        features_dict.get(f, 0.0) for f in ANOMALY_FEATURES
    ]])
    values_scaled = scaler.transform(values)

    prediction    = model.predict(values_scaled)[0]
    anomaly_score = float(model.score_samples(values_scaled)[0])
    is_anomaly    = prediction == -1

    return {
        "is_anomaly":    bool(is_anomaly),
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_label": "Anomalous" if is_anomaly else "Normal",
    }


def flag_anomalies_in_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all rows in a dataframe for anomaly.
    Used for batch scoring and dashboard visualization.
    """
    model  = joblib.load(MODELS / "anomaly_detector.pkl")
    scaler = joblib.load(MODELS / "anomaly_scaler.pkl")

    valid = df.dropna(subset=ANOMALY_FEATURES).copy()
    X_scaled = scaler.transform(valid[ANOMALY_FEATURES].values)

    valid['anomaly_label'] = model.predict(X_scaled)
    valid['anomaly_score'] = model.score_samples(X_scaled)
    valid['is_anomaly']    = valid['anomaly_label'] == -1

    logger.info(
        f"Flagged {valid['is_anomaly'].sum():,} anomalies "
        f"out of {len(valid):,} rows "
        f"({valid['is_anomaly'].mean()*100:.1f}%)"
    )
    return valid


if __name__ == "__main__":
    # Train and save the model
    model, scaler = train_anomaly_detector()

    # Score the test period and show top anomalies
    features = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "data" / "features" / "feature_matrix.parquet"
    )
    test = features[features['year'] >= 2025].copy()
    test = flag_anomalies_in_dataset(test)

    print(f"\nTop 15 anomalous section-category pairs in test period:\n")
    anomalies = (
        test[test['is_anomaly']]
        .sort_values('anomaly_score')
        [['section', 'category', 'year', 'month',
          'anomaly_score', 'demand_growth_rate', 'demand_per_can',
          'lag_1_efficiency']]
        .head(15)
    )
    print(anomalies.to_string(index=False))

    # Save scored test predictions for dashboard
    test.to_parquet(PROCESSED / "anomaly_scores.parquet", index=False)
    print(f"\nSaved anomaly scores to data/processed/anomaly_scores.parquet")