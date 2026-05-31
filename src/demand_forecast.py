import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS    = Path(__file__).resolve().parents[1] / "models"


def build_prophet_df(series: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a section-category time series into Prophet format.
    Prophet requires columns: ds (datetime) and y (value to forecast).
    """
    df = series.copy()
    df['ds'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    df['y'] = df['demand']
    return df[['ds', 'y']].sort_values('ds').reset_index(drop=True)


def train_division_demand_forecaster(division_id: int) -> dict | None:
    """
    Train a Prophet model on monthly total demand for a single division.
    Aggregates all section-category pairs within the division.
    Returns forecast for next 3 months.
    """
    billing = pd.read_parquet(PROCESSED / "billing_clean.parquet")

    div_data = billing[
        (billing['division'] == division_id) &
        (~billing['is_zero_demand'])
    ].copy()

    if len(div_data) < 12:
        logger.warning(f"Division {division_id} has insufficient data")
        return None

    # Aggregate to monthly total demand
    monthly = (
        div_data.groupby(['year', 'month'])['demand']
        .sum()
        .reset_index()
    )

    prophet_df = build_prophet_df(monthly)

    if len(prophet_df) < 6:
        return None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
    )
    model.fit(prophet_df)

    # Forecast next 3 months
    future   = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)

    # Return last 12 actual + next 3 forecast
    actual_months   = len(prophet_df)
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_result['is_forecast'] = False
    forecast_result.iloc[-3:, forecast_result.columns.get_loc('is_forecast')] = True
    forecast_result['yhat']       = forecast_result['yhat'].clip(lower=0)
    forecast_result['yhat_lower'] = forecast_result['yhat_lower'].clip(lower=0)
    forecast_result['yhat_upper'] = forecast_result['yhat_upper'].clip(lower=0)

    return {
        "division":        division_id,
        "actual_months":   actual_months,
        "forecast":        forecast_result.tail(15).to_dict(orient='records'),
        "next_month_demand_crore": round(
            float(forecast_result[forecast_result['is_forecast']].iloc[0]['yhat']) / 1e7, 2
        ),
    }


def compute_revenue_forecast(division_id: int, efficiency_pct: float) -> dict | None:
    """
    Combine Prophet demand forecast with XGBoost efficiency prediction
    to produce a complete revenue forecast for a division.

    expected_collection = forecasted_demand * predicted_efficiency
    """
    demand_result = train_division_demand_forecaster(division_id)
    if demand_result is None:
        return None

    next_month_demand     = demand_result['next_month_demand_crore']
    expected_collection   = round(next_month_demand * (efficiency_pct / 100), 2)
    expected_shortfall    = round(next_month_demand - expected_collection, 2)

    return {
        "division":                  division_id,
        "forecasted_demand_crore":   next_month_demand,
        "efficiency_pct":            efficiency_pct,
        "expected_collection_crore": expected_collection,
        "expected_shortfall_crore":  expected_shortfall,
        "forecast_series":           demand_result['forecast'],
    }


def batch_forecast_top_divisions(top_n: int = 10) -> pd.DataFrame:
    """
    Run demand forecast for the top N divisions by total demand.
    Returns a summary DataFrame.
    """
    billing = pd.read_parquet(PROCESSED / "billing_clean.parquet")

    top_divisions = (
        billing[~billing['is_zero_demand']]
        .groupby('division')['demand']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    results = []
    for div in top_divisions:
        logger.info(f"Forecasting division {div}...")
        result = train_division_demand_forecaster(div)
        if result:
            results.append({
                "division":                int(div),
                "next_month_demand_crore": result['next_month_demand_crore'],
                "actual_months":           result['actual_months'],
            })

    df = pd.DataFrame(results)
    logger.info(f"Completed demand forecast for {len(df)} divisions")
    return df


if __name__ == "__main__":
    logger.info("Running batch demand forecast for top 10 divisions...")
    summary = batch_forecast_top_divisions(top_n=10)
    print("\nTop 10 divisions — next month demand forecast:\n")
    print(summary.to_string(index=False))

    logger.info("\nRunning full revenue forecast for division 28...")
    revenue = compute_revenue_forecast(division_id=28, efficiency_pct=61.5)
    if revenue:
        print(f"\nDivision 28 revenue forecast:")
        print(f"  Forecasted demand:      ₹{revenue['forecasted_demand_crore']:.2f} crore")
        print(f"  Expected collection:    ₹{revenue['expected_collection_crore']:.2f} crore")
        print(f"  Expected shortfall:     ₹{revenue['expected_shortfall_crore']:.2f} crore")