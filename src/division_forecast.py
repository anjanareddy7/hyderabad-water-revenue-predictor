import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_latest_section_predictions() -> pd.DataFrame:
    """
    Load test predictions and return the most recent month
    for each section-category pair.
    """
    preds = pd.read_parquet(PROCESSED / "test_predictions.parquet")
    latest = (
        preds
        .sort_values(['section', 'category', 'year', 'month'])
        .groupby(['section', 'category'])
        .last()
        .reset_index()
    )
    return latest


def compute_division_forecast(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up section-level predictions to division level.

    For each division, compute:
    - total_demand: sum of current demand across all sections
    - expected_collection: sum of (demand * predicted_efficiency)
    - expected_shortfall: total_demand - expected_collection
    - weighted_efficiency: expected_collection / total_demand
    - sections_count: number of section-category pairs
    - high_risk_count: number of High Risk section-category pairs
    - high_risk_shortfall: shortfall attributable to High Risk sections
    """
    if 'predicted_efficiency' not in preds.columns:
        preds['predicted_efficiency'] = preds['efficiency']

    preds = preds.copy()
    preds['expected_collection'] = preds['demand'] * preds['predicted_efficiency']
    preds['expected_shortfall']  = preds['demand'] - preds['expected_collection']
    preds['is_high_risk'] = preds.get(
        'risk_tier', pd.Series(['Medium Risk'] * len(preds))
    ) == 'High Risk'

    division_forecast = preds.groupby('division').agg(
        total_demand          = ('demand',               'sum'),
        expected_collection   = ('expected_collection',  'sum'),
        expected_shortfall    = ('expected_shortfall',   'sum'),
        sections_count        = ('section',              'count'),
        high_risk_count       = ('is_high_risk',         'sum'),
    ).reset_index()

    division_forecast['weighted_efficiency'] = (
        division_forecast['expected_collection'] /
        division_forecast['total_demand'].replace(0, np.nan)
    )

    # High risk shortfall
    high_risk = preds[preds['is_high_risk']].groupby('division').agg(
        high_risk_shortfall=('expected_shortfall', 'sum')
    ).reset_index()

    division_forecast = division_forecast.merge(high_risk, on='division', how='left')
    division_forecast['high_risk_shortfall'] = (
        division_forecast['high_risk_shortfall'].fillna(0)
    )

    # Convert to crore
    for col in ['total_demand', 'expected_collection',
                'expected_shortfall', 'high_risk_shortfall']:
        division_forecast[f'{col}_crore'] = division_forecast[col] / 1e7

    division_forecast = division_forecast.sort_values(
        'expected_shortfall', ascending=False
    ).reset_index(drop=True)

    logger.info(f"Division forecast computed for {len(division_forecast)} divisions")
    return division_forecast


def get_division_summary(division_id: int) -> dict | None:
    """
    Get forecast summary for a single division.
    Returns None if division not found.
    """
    preds    = load_latest_section_predictions()
    forecast = compute_division_forecast(preds)

    row = forecast[forecast['division'] == division_id]
    if len(row) == 0:
        return None

    row = row.iloc[0]
    return {
        "division":                    int(division_id),
        "total_demand_crore":          round(float(row['total_demand_crore']), 2),
        "expected_collection_crore":   round(float(row['expected_collection_crore']), 2),
        "expected_shortfall_crore":    round(float(row['expected_shortfall_crore']), 2),
        "weighted_efficiency_pct":     round(float(row['weighted_efficiency']) * 100, 1),
        "sections_count":              int(row['sections_count']),
        "high_risk_count":             int(row['high_risk_count']),
        "high_risk_shortfall_crore":   round(float(row['high_risk_shortfall_crore']), 2),
    }


def get_all_divisions_forecast() -> list[dict]:
    """Get forecast for all divisions, sorted by shortfall descending."""
    preds    = load_latest_section_predictions()
    forecast = compute_division_forecast(preds)
    return forecast.to_dict(orient='records')


if __name__ == "__main__":
    preds    = load_latest_section_predictions()
    forecast = compute_division_forecast(preds)
    print(f"\nDivision forecast — top 10 by shortfall:\n")
    print(forecast[[
        'division', 'total_demand_crore', 'expected_collection_crore',
        'expected_shortfall_crore', 'weighted_efficiency',
        'sections_count', 'high_risk_count'
    ]].head(10).round(3).to_string(index=False))