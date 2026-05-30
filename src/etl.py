import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

RAW       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_billing(raw_dir: Path = RAW) -> pd.DataFrame:
    """Load and concatenate all monthly billing CSVs."""
    files = sorted(raw_dir.glob("billing_and_collection_report_*.csv"))
    if not files:
        raise FileNotFoundError(f"No billing CSVs found in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f"Loaded {len(files)} billing files → {len(df):,} rows")
    return df


def load_connections(raw_dir: Path = RAW) -> pd.DataFrame:
    """Load and concatenate all monthly water connection CSVs."""
    files = sorted(raw_dir.glob("water_connection_report_*.csv"))
    if not files:
        raise FileNotFoundError(f"No connection CSVs found in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f"Loaded {len(files)} connection files → {len(df):,} rows")
    return df


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_billing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw billing dataframe.
    - Drop rows where both section and division are null
    - Fill null division with 0 (sentinel for unassigned sections)
    - Cast division to int
    - Strip whitespace from section and category
    """
    before = len(df)

    # Drop completely empty rows (null section AND null division AND zero demand)
    empty_mask = df['section'].isna() & df['division'].isna()
    df = df[~empty_mask].copy()
    logger.info(f"Dropped {before - len(df)} empty placeholder rows")

    # Fill null division with sentinel 0
    df['division'] = df['division'].fillna(0).infer_objects(copy=False).astype(int)

    # Strip whitespace
    df['section']  = df['section'].str.strip()
    df['category'] = df['category'].str.strip()

    return df.reset_index(drop=True)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three boolean quality flag columns.
    - is_zero_demand: demand == 0
    - is_overcollection: collection > demand * 1.5
    - is_structural_change: abs month-over-month noofcans change > 5%
    """
    df = df.copy()
    
    df['is_zero_demand'] = df['demand'] == 0
    df['is_2022_baseline'] = df['year'] == 2022
    df['is_overcollection'] = (df['demand'] > 0) & (
        df['collection'] > df['demand'] * 1.5
    )

    # Sort for correct lag computation
    df = df.sort_values(['section', 'category', 'year', 'month']).reset_index(drop=True)

    df['_prev_cans'] = df.groupby(['section', 'category'])['noofcans'].shift(1)
    df['is_structural_change'] = (
        df['_prev_cans'].notna() &
        (df['_prev_cans'] > 0) &
        (
            (df['noofcans'] - df['_prev_cans']).abs() / df['_prev_cans'] > 0.05
        )
    )
    df = df.drop(columns=['_prev_cans'])

    logger.info(
        f"Quality flags — zero_demand: {df['is_zero_demand'].sum():,} | "
        f"overcollection: {df['is_overcollection'].sum():,} | "
        f"structural_change: {df['is_structural_change'].sum():,}"
    )
    return df


def standardise_section_names(
    df: pd.DataFrame,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Fuzzy-match section names to their canonical form.
    Uses the most frequent spelling of each name as the canonical version.
    Logs all replacements made.
    """
    df = df.copy()

    # Build canonical list — most frequent spelling wins
    canonical = (
        df['section']
        .dropna()
        .value_counts()
        .index
        .tolist()
    )

    replacements = {}
    for name in df['section'].dropna().unique():
        match, score, _ = process.extractOne(
            name, canonical, scorer=fuzz.token_sort_ratio
        )
        if score >= similarity_threshold and match != name:
            replacements[name] = match

    if replacements:
        logger.info(f"Section name replacements: {len(replacements)}")
        for old, new in list(replacements.items())[:10]:
            logger.info(f"  '{old}' → '{new}'")
        df['section'] = df['section'].replace(replacements)

    return df


# ── Writer ───────────────────────────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe to parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df):,} rows → {path}")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_billing_etl() -> pd.DataFrame:
    """Full billing ETL pipeline. Returns cleaned dataframe."""
    df = load_billing()
    df = clean_billing(df)
    df = standardise_section_names(df)
    df = add_quality_flags(df)
    write_parquet(df, PROCESSED / "billing_clean.parquet")
    return df


def run_connection_etl() -> pd.DataFrame:
    """Full connection ETL pipeline. Returns cleaned dataframe."""
    df = load_connections()
    df['section']  = df['section'].str.strip()
    df['category'] = df['category'].str.strip()
    write_parquet(df, PROCESSED / "connections_clean.parquet")
    return df


if __name__ == "__main__":
    run_billing_etl()
    run_connection_etl()