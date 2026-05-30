import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS    = Path(__file__).resolve().parents[1] / "models"
FEATURES  = Path(__file__).resolve().parents[1] / "data" / "features"

# ── Category mapping ─────────────────────────────────────────────────────────

CATEGORY_MAP = {
    'D': 'Domestic', 'DM': 'Domestic', 'M0': 'Domestic', 'M1': 'Domestic',
    'M2': 'Domestic', 'M3': 'Domestic', 'M4': 'Domestic', 'MS': 'Domestic',
    'DP': 'Domestic', 'T1': 'Domestic', 'T2': 'Domestic', 'T3': 'Domestic',
    'T4': 'Domestic',
    'DS': 'Slum', 'PS': 'Slum', 'GP': 'Slum',
    'C': 'Commercial', 'BC': 'Commercial', 'C1': 'Commercial',
    'FS': 'Commercial', 'XO': 'Commercial', 'X': 'Commercial',
    'I': 'Industrial', 'I1': 'Industrial', 'IW': 'Industrial',
    'G': 'GovtInstitutional', 'H': 'GovtInstitutional', 'RC': 'GovtInstitutional',
    'CH': 'GovtInstitutional', 'CB': 'GovtInstitutional', 'MB': 'GovtInstitutional',
    'N': 'GovtInstitutional', 'O': 'GovtInstitutional', 'S': 'GovtInstitutional',
    'V': 'GovtInstitutional'
}

CATEGORY_GROUP_ENCODING = {
    'Slum': 0, 'GovtInstitutional': 1,
    'Domestic': 2, 'Commercial': 3, 'Industrial': 4
}

RISK_TIER_ENCODING = {
    'High Risk': 0, 'Medium Risk': 1, 'Moderate Risk': 2, 'Low Risk': 3
}


# ── Target ───────────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute collection_efficiency = collection / demand, clipped to [0, 2].
    Excludes zero-demand rows — they have no valid target.
    """
    df = df[~df['is_zero_demand']].copy()
    df['efficiency'] = (df['collection'] / df['demand']).clip(0, 2)
    logger.info(f"Target computed on {len(df):,} non-zero-demand rows")
    return df


# ── Lag and rolling features ─────────────────────────────────────────────────

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features within each (section, category) group.
    Requires df sorted by (section, category, year, month).
    """
    df = df.sort_values(
        ['section', 'category', 'year', 'month']
    ).reset_index(drop=True)

    grp = df.groupby(['section', 'category'])

    df['lag_1_efficiency'] = grp['efficiency'].shift(1)
    df['lag_2_efficiency'] = grp['efficiency'].shift(2)
    df['lag_3_efficiency'] = grp['efficiency'].shift(3)
    df['lag_1_collection'] = grp['collection'].shift(1)
    df['lag_1_demand']     = grp['demand'].shift(1)
    df['lag_1_noofcans']   = grp['noofcans'].shift(1)

    logger.info("Lag features computed")
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 3-month features within each (section, category) group.
    min_periods=2 so we get values after 2 months, not just 3.
    """
    grp = df.groupby(['section', 'category'])

    df['rolling_3m_efficiency_mean'] = (
        grp['efficiency']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )
    df['rolling_3m_efficiency_std'] = (
        grp['efficiency']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    )
    df['rolling_3m_demand_mean'] = (
        grp['demand']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )

    logger.info("Rolling features computed")
    return df


# ── Growth features ───────────────────────────────────────────────────────────

def compute_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Demand growth, connection growth, demand per connection."""
    df['demand_growth_rate'] = (
        (df['demand'] - df['lag_1_demand']) / df['lag_1_demand']
    ).replace([np.inf, -np.inf], np.nan)

    df['noofcans_growth'] = (
        (df['noofcans'] - df['lag_1_noofcans']) / df['lag_1_noofcans']
    ).replace([np.inf, -np.inf], np.nan)

    df['demand_per_can'] = (
        df['demand'] / df['noofcans'].replace(0, np.nan)
    )

    logger.info("Growth features computed")
    return df


# ── Calendar features ─────────────────────────────────────────────────────────

def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month cyclical encoding, seasonal flags."""
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter']   = ((df['month'] - 1) // 3) + 1

    # Flags derived from EDA — January anomaly is stronger than monsoon
    df['is_january']             = (df['month'] == 1).astype(int)
    df['is_monsoon']             = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_financial_year_end']  = (df['month'] == 3).astype(int)
    df['is_2022_baseline']       = (df['year'] == 2022).astype(int)

    logger.info("Calendar features computed")
    return df


# ── Category features ─────────────────────────────────────────────────────────

def compute_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw category codes to groups and encode."""
    df['category_group'] = (
        df['category'].map(CATEGORY_MAP).fillna('Other')
    )
    df['category_group_encoded'] = (
        df['category_group'].map(CATEGORY_GROUP_ENCODING).fillna(-1).astype(int)
    )
    logger.info("Category features computed")
    return df


# ── Risk tier join ────────────────────────────────────────────────────────────

def join_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-computed risk tiers from clustering.
    Falls back to 'Medium Risk' for unseen section-category pairs.
    """
    tiers = pd.read_csv(PROCESSED / "section_risk_tiers.csv")
    tiers = tiers[['section', 'category_group', 'risk_tier']]

    df = df.merge(tiers, on=['section', 'category_group'], how='left')
    missing = df['risk_tier'].isna().sum()
    if missing > 0:
        logger.info(f"Risk tier: {missing} rows unmatched — filling with Medium Risk")
        df['risk_tier'] = df['risk_tier'].fillna('Medium Risk')

    df['risk_tier_encoded'] = (
        df['risk_tier'].map(RISK_TIER_ENCODING).fillna(1).astype(int)
    )
    logger.info("Risk tier joined")
    return df


# ── Connection features ───────────────────────────────────────────────────────

def join_connection_features(
    df: pd.DataFrame,
    conn: pd.DataFrame
) -> pd.DataFrame:
    """
    Join water connection data on (year, month, section).
    Fills unmatched rows with section-level medians.
    """
    conn_agg = conn.groupby(['year', 'month', 'section']).agg(
        applied=('applied', 'sum'),
        approved=('approved', 'sum')
    ).reset_index()
    conn_agg['approval_rate'] = (
        conn_agg['approved'] / conn_agg['applied'].replace(0, np.nan)
    )

    df = df.merge(conn_agg[['year', 'month', 'section', 'approval_rate']],
                  on=['year', 'month', 'section'], how='left')

    # Fill missing with section-level median
    section_median = (
        df.groupby('section')['approval_rate']
        .transform('median')
    )
    df['approval_rate'] = df['approval_rate'].fillna(section_median)
    df['approval_rate'] = df['approval_rate'].fillna(0.5)  # global fallback

    matched = df['approval_rate'].notna().sum()
    logger.info(f"Connection features joined — {matched:,} rows with approval_rate")
    return df


# ── Warmup row removal ────────────────────────────────────────────────────────

def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where lag_1_efficiency is NaN.
    These are the first months of each section-category group
    where there is no history to build features from.
    """
    before = len(df)
    df = df[df['lag_1_efficiency'].notna()].copy()
    logger.info(f"Dropped {before - len(df):,} warmup rows (no lag history)")
    return df.reset_index(drop=True)


# ── Full pipeline ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'lag_1_efficiency', 'lag_2_efficiency', 'lag_3_efficiency',
    'rolling_3m_efficiency_mean', 'rolling_3m_efficiency_std',
    'demand_growth_rate', 'noofcans_growth', 'demand_per_can',
    'rolling_3m_demand_mean', 'category_group_encoded', 'risk_tier_encoded',
    'is_monsoon', 'is_financial_year_end', 'is_january', 'is_2022_baseline',
    'month_sin', 'month_cos', 'approval_rate', 'is_structural_change',
]

def run_feature_pipeline() -> pd.DataFrame:
    """Full feature engineering pipeline. Returns final feature matrix."""
    # Load cleaned data
    df   = pd.read_parquet(PROCESSED / "billing_clean.parquet")
    conn = pd.read_parquet(PROCESSED / "connections_clean.parquet")

    logger.info(f"Loaded billing: {len(df):,} rows")

    # Build features step by step
    df = compute_target(df)
    df = compute_category_features(df)
    df = compute_lag_features(df)
    df = compute_rolling_features(df)
    df = compute_growth_features(df)
    df = compute_calendar_features(df)
    df = join_risk_tier(df)
    df = join_connection_features(df, conn)
    df = drop_warmup_rows(df)

    # Keep only what we need
    keep_cols = (
        ['year', 'month', 'division', 'section', 'category',
         'category_group', 'risk_tier', 'demand', 'collection',
         'efficiency', 'is_zero_demand', 'is_overcollection']
        + FEATURE_COLS
    )
    # Deduplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c in df.columns
                 and not (c in seen or seen.add(c))]
    df = df[keep_cols]

    # Save
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = FEATURES / "feature_matrix.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Feature matrix saved: {df.shape} → {out}")

    # Summary
    logger.info(f"NaN counts in feature cols:")
    for col in FEATURE_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                logger.info(f"  {col}: {n:,} NaNs")

    return df


if __name__ == "__main__":
    df = run_feature_pipeline()
    print(f"\nFinal feature matrix shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")