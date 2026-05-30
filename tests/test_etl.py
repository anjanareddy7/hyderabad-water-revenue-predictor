import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.etl import (
    clean_billing,
    add_quality_flags,
    standardise_section_names,
    load_billing,
    write_parquet,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_billing_df(**overrides):
    """Create a minimal valid billing dataframe for testing."""
    base = {
        "year":       [2022, 2022, 2022],
        "month":      [1,    1,    1   ],
        "division":   [1.0,  1.0,  None],
        "section":    ["BAHADURPURA", "KPHB", None],
        "collection": [1000.0, 500.0, 0.0],
        "demand":     [1200.0, 500.0, 0.0],
        "noofcans":   [10,     5,     0  ],
        "category":   ["D",   "C",   "D" ],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ── clean_billing tests ───────────────────────────────────────────────────────

def test_clean_billing_drops_empty_rows():
    """Rows with both null section and null division should be dropped."""
    df = make_billing_df()
    # Row 2 has null section AND null division — should be dropped
    result = clean_billing(df)
    assert len(result) == 2


def test_clean_billing_fills_null_division():
    """Null division should be filled with 0, not dropped."""
    # Make a df where section is present but division is null
    df = pd.DataFrame({
        "year": [2022], "month": [1],
        "division": [None], "section": ["CHANDANAGAR"],
        "collection": [0.0], "demand": [100.0],
        "noofcans": [5], "category": ["DM"],
    })
    result = clean_billing(df)
    assert result['division'].iloc[0] == 0
    assert result['division'].dtype == int


def test_clean_billing_strips_whitespace():
    """Section and category names should have whitespace stripped."""
    df = pd.DataFrame({
        "year": [2022], "month": [1],
        "division": [1.0], "section": ["  BAHADURPURA  "],
        "collection": [100.0], "demand": [200.0],
        "noofcans": [5], "category": [" D "],
    })
    result = clean_billing(df)
    assert result['section'].iloc[0] == "BAHADURPURA"
    assert result['category'].iloc[0] == "D"


# ── add_quality_flags tests ───────────────────────────────────────────────────

def test_zero_demand_flag():
    """Rows with demand == 0 should have is_zero_demand = True."""
    df = pd.DataFrame({
        "year": [2022, 2022], "month": [1, 1],
        "division": [1, 1], "section": ["A", "B"],
        "collection": [0.0, 100.0], "demand": [0.0, 200.0],
        "noofcans": [5, 5], "category": ["D", "D"],
    })
    result = add_quality_flags(df)
    assert result.loc[result['section'] == 'A', 'is_zero_demand'].values[0] == True
    assert result.loc[result['section'] == 'B', 'is_zero_demand'].values[0] == False


def test_overcollection_flag():
    """collection > demand * 1.5 should set is_overcollection = True."""
    df = pd.DataFrame({
        "year": [2022, 2022], "month": [1, 1],
        "division": [1, 1], "section": ["A", "A"],
        "collection": [1600.0, 100.0], "demand": [1000.0, 200.0],
        "noofcans": [5, 5], "category": ["D", "C"],
    })
    result = add_quality_flags(df)
    assert result.loc[result['category'] == 'D', 'is_overcollection'].values[0] == True
    assert result.loc[result['category'] == 'C', 'is_overcollection'].values[0] == False


def test_structural_change_flag():
    """noofcans change > 5% month-over-month should set is_structural_change."""
    df = pd.DataFrame({
        "year":       [2022, 2022],
        "month":      [1,    2   ],
        "division":   [1,    1   ],
        "section":    ["A", "A"  ],
        "collection": [100.0, 100.0],
        "demand":     [200.0, 200.0],
        "noofcans":   [100,   115 ],  # 15% increase — above 5% threshold
        "category":   ["D",  "D" ],
    })
    result = add_quality_flags(df)
    # Month 2 should be flagged, month 1 has no previous value
    month2 = result[result['month'] == 2]
    assert month2['is_structural_change'].values[0] == True


def test_no_structural_change_below_threshold():
    """noofcans change <= 5% should NOT set is_structural_change."""
    df = pd.DataFrame({
        "year":       [2022, 2022],
        "month":      [1,    2   ],
        "division":   [1,    1   ],
        "section":    ["A", "A"  ],
        "collection": [100.0, 100.0],
        "demand":     [200.0, 200.0],
        "noofcans":   [100,   103 ],  # 3% increase — below threshold
        "category":   ["D",  "D" ],
    })
    result = add_quality_flags(df)
    month2 = result[result['month'] == 2]
    assert month2['is_structural_change'].values[0] == False


# ── standardise_section_names tests ──────────────────────────────────────────

def test_standardise_fixes_trailing_space():
    """Section names differing only by trailing space should be unified."""
    df = pd.DataFrame({
        "year": [2022, 2022, 2022], "month": [1, 1, 1],
        "division": [1, 1, 1], "section": ["BAHADURPURA", "BAHADURPURA ", "BAHADURPURA"],
        "collection": [100.0, 200.0, 150.0], "demand": [200.0, 300.0, 250.0],
        "noofcans": [5, 5, 5], "category": ["D", "C", "I"],
    })
    # Clean first to strip whitespace, then standardise
    from src.etl import clean_billing
    df_clean = clean_billing(df)
    result = standardise_section_names(df_clean)
    assert result['section'].nunique() == 1


# ── write_parquet tests ───────────────────────────────────────────────────────

def test_write_parquet(tmp_path):
    """Written parquet should be readable and match original dataframe."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = tmp_path / "test.parquet"
    write_parquet(df, out)
    result = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df, result)