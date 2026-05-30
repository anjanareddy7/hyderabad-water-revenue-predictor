import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import mlflow
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURES  = Path(__file__).resolve().parents[1] / "data" / "features"
MODELS    = Path(__file__).resolve().parents[1] / "models"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

FEATURE_COLS = [
    'lag_1_efficiency', 'lag_2_efficiency', 'lag_3_efficiency',
    'rolling_3m_efficiency_mean', 'rolling_3m_efficiency_std',
    'demand_growth_rate', 'noofcans_growth', 'demand_per_can',
    'rolling_3m_demand_mean', 'category_group_encoded', 'risk_tier_encoded',
    'is_monsoon', 'is_financial_year_end', 'is_january', 'is_2022_baseline',
    'month_sin', 'month_cos', 'approval_rate', 'is_structural_change',
]

TARGET = 'efficiency'


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.clip(0.01))) * 100
    if label:
        logger.info(f"{label} — RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.1f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ── Train/test split ──────────────────────────────────────────────────────────

def time_split(df):
    """Train on 2022–2024, test on 2025+."""
    train = df[df['year'] <= 2024].copy()
    test  = df[df['year'] >= 2025].copy()
    logger.info(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# ── Target encoding ───────────────────────────────────────────────────────────

def target_encode(train, test, col, target, smoothing=10):
    """
    Encode a categorical column using target mean with smoothing.
    Fit on train only, apply to both.
    """
    global_mean = train[target].mean()
    stats = train.groupby(col)[target].agg(['mean', 'count'])
    stats['encoded'] = (
        (stats['mean'] * stats['count'] + global_mean * smoothing)
        / (stats['count'] + smoothing)
    )
    mapping = stats['encoded'].to_dict()
    train[f'{col}_te'] = train[col].map(mapping).fillna(global_mean)
    test[f'{col}_te']  = test[col].map(mapping).fillna(global_mean)
    return train, test, mapping


# ── Baseline model ────────────────────────────────────────────────────────────

def run_baseline(train, test):
    """Naive baseline: predict lag_1_efficiency as the forecast."""
    y_test  = test[TARGET]
    y_pred  = test['lag_1_efficiency'].fillna(test['lag_1_efficiency'].median())
    metrics = compute_metrics(y_test, y_pred, "Baseline (lag-1)")
    return metrics


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(train, test, feature_cols):
    X_train = train[feature_cols].fillna(-999)
    y_train = train[TARGET]
    X_test  = test[feature_cols].fillna(-999)
    y_test  = test[TARGET]

    params = {
        "n_estimators":     500,
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
        "n_jobs":           -1,
        "early_stopping_rounds": 30,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred  = model.predict(X_test).clip(0, 2)
    metrics = compute_metrics(y_test, y_pred, "XGBoost")
    return model, y_pred, metrics


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(train, test, feature_cols):
    X_train = train[feature_cols].fillna(-999)
    y_train = train[TARGET]
    X_test  = test[feature_cols].fillna(-999)
    y_test  = test[TARGET]

    params = {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "max_depth":     6,
        "subsample":     0.8,
        "random_state":  42,
        "n_jobs":        -1,
        "verbose":       -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    y_pred  = model.predict(X_test).clip(0, 2)
    metrics = compute_metrics(y_test, y_pred, "LightGBM")
    return model, y_pred, metrics


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnostic_by_group(test, y_pred, group_col, label):
    """MAE breakdown by a categorical column."""
    df = test.copy()
    df['y_pred'] = y_pred
    df['ae'] = (df[TARGET] - df['y_pred']).abs()
    result = df.groupby(group_col)['ae'].mean().sort_values(ascending=False)
    logger.info(f"\n=== {label} ===")
    for grp, mae in result.items():
        logger.info(f"  {grp:<25} MAE: {mae:.4f}")
    return result


def diagnostic_blind_spots(test, y_pred):
    """Find sections where model consistently over-predicts."""
    df = test.copy()
    df['y_pred'] = y_pred
    blind = df[(df['y_pred'] > 0.8) & (df[TARGET] < 0.5)]
    if len(blind) == 0:
        logger.info("No blind spots found")
        return pd.DataFrame()
    summary = (
        blind.groupby(['section', 'category_group'])
        .agg(count=('y_pred', 'count'),
             mean_predicted=('y_pred', 'mean'),
             mean_actual=(TARGET, 'mean'))
        .sort_values('count', ascending=False)
    )
    logger.info(f"\n=== Blind spots (predicted >0.8, actual <0.5): {len(summary)} pairs ===")
    logger.info(summary.head(10).to_string())
    return summary


# ── Main training run ─────────────────────────────────────────────────────────

def run_training():
    mlflow.set_experiment("hmwssb-collection-efficiency")

    df = pd.read_parquet(FEATURES / "feature_matrix.parquet")
    logger.info(f"Loaded feature matrix: {df.shape}")

    train, test = time_split(df)

    # Target encode section and division
    train, test, section_mapping  = target_encode(train, test, 'section',  TARGET)
    train, test, division_mapping = target_encode(train, test, 'division', TARGET)

    feature_cols = FEATURE_COLS + ['section_te', 'division_te']

    # ── Baseline ──
    with mlflow.start_run(run_name="baseline"):
        baseline_metrics = run_baseline(train, test)
        mlflow.log_metrics(baseline_metrics)
    logger.info("Baseline logged to MLflow")

    # ── XGBoost ──
    with mlflow.start_run(run_name="xgboost"):
        xgb_model, xgb_preds, xgb_metrics = train_xgboost(train, test, feature_cols)
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_params({"model": "xgboost", "n_estimators": 500,
                           "learning_rate": 0.05, "max_depth": 6})

        # Feature importance
        importance = pd.Series(
            xgb_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        logger.info(f"\n=== XGBoost top 10 features ===")
        logger.info(importance.head(10).to_string())

        # Diagnostics
        diagnostic_by_group(test, xgb_preds, 'category_group', 'MAE by category group')
        diagnostic_by_group(test, xgb_preds, 'risk_tier',      'MAE by risk tier')
        diagnostic_by_group(test, xgb_preds, 'month',          'MAE by month')
        blind_spots = diagnostic_blind_spots(test, xgb_preds)

    # ── LightGBM ──
    with mlflow.start_run(run_name="lightgbm"):
        lgb_model, lgb_preds, lgb_metrics = train_lightgbm(train, test, feature_cols)
        mlflow.log_metrics(lgb_metrics)
        mlflow.log_params({"model": "lightgbm", "n_estimators": 500,
                           "learning_rate": 0.05, "max_depth": 6})

    # ── Pick best model ──
    if xgb_metrics['rmse'] <= lgb_metrics['rmse']:
        best_model  = xgb_model
        best_name   = "xgboost"
        best_preds  = xgb_preds
        best_metrics = xgb_metrics
    else:
        best_model  = lgb_model
        best_name   = "lightgbm"
        best_preds  = lgb_preds
        best_metrics = lgb_metrics

    logger.info(f"\nBest model: {best_name} — RMSE {best_metrics['rmse']:.4f}")

    # ── Save artifacts ──
    MODELS.mkdir(exist_ok=True)
    joblib.dump(best_model,       MODELS / "best_model.pkl")
    joblib.dump(section_mapping,  MODELS / "section_te_mapping.pkl")
    joblib.dump(division_mapping, MODELS / "division_te_mapping.pkl")
    joblib.dump(feature_cols,     MODELS / "feature_cols.pkl")

    # Save test predictions for dashboard
    test_out = test.copy()
    test_out['predicted_efficiency'] = best_preds
    test_out.to_parquet(PROCESSED / "test_predictions.parquet", index=False)

    logger.info("All artifacts saved")
    logger.info(f"\nBaseline RMSE: {baseline_metrics['rmse']:.4f}")
    logger.info(f"Best model RMSE: {best_metrics['rmse']:.4f}")
    logger.info(f"Improvement over baseline: "
                f"{(baseline_metrics['rmse'] - best_metrics['rmse'])/baseline_metrics['rmse']*100:.1f}%")

    return best_model, test_out


if __name__ == "__main__":
    run_training()