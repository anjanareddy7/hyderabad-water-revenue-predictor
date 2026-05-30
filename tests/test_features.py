import pandas as pd
import numpy as np
import pytest
from src.features import (
    compute_target,
    compute_lag_features,
    compute_rolling_features,
    compute_growth_features,
    compute_calendar_features,
    compute_category_features,
    drop_warmup_rows,
    CATEGORY_MAP,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_group(efficiencies, section="A", category="D"):
    """Build a minimal billing df with known efficiency sequence."""
    n = len(efficiencies)
    df = pd.DataFrame({
        "year":       [2022] * n,
        "month":      list(range(1, n + 1)),
        "division":   [1] * n,
        "section":    [section] * n,
        "collection": [e * 1000 for e in efficiencies],
        "demand":     [1000.0] * n,
        "noofcans":   [100] * n,
        "category":   [category] * n,
        "is_zero_demand":    [False] * n,
        "is_overcollection": [False] * n,
        "is_structural_change": [False] * n,
    })
    df["efficiency"] = efficiencies
    return df


# ── compute_target ────────────────────────────────────────────────────────────

def test_target_excludes_zero_demand():
    df = pd.DataFrame({
        "year": [2022, 2022], "month": [1, 2],
        "division": [1, 1], "section": ["A", "A"],
        "collection": [0.0, 100.0], "demand": [0.0, 200.0],
        "noofcans": [5, 5], "category": ["D", "D"],
        "is_zero_demand": [True, False],
        "is_overcollection": [False, False],
        "is_structural_change": [False, False],
    })
    result = compute_target(df)
    assert len(result) == 1
    assert result['efficiency'].iloc[0] == pytest.approx(0.5)


def test_target_clips_overcollection():
    df = pd.DataFrame({
        "year": [2022], "month": [1],
        "division": [1], "section": ["A"],
        "collection": [5000.0], "demand": [1000.0],
        "noofcans": [5], "category": ["D"],
        "is_zero_demand": [False],
        "is_overcollection": [True],
        "is_structural_change": [False],
    })
    result = compute_target(df)
    assert result['efficiency'].iloc[0] == pytest.approx(2.0)


# ── compute_lag_features ──────────────────────────────────────────────────────

def test_lag_values_correct():
    """lag_1 of month 5 should equal month 4's efficiency, and so on."""
    efficiencies = [0.8, 0.7, 0.6, 0.5, 0.4]
    df = make_group(efficiencies)
    result = compute_lag_features(df)
    row5 = result[result['month'] == 5].iloc[0]
    assert row5['lag_1_efficiency'] == pytest.approx(0.5)  # month 4
    assert row5['lag_2_efficiency'] == pytest.approx(0.6)  # month 3
    assert row5['lag_3_efficiency'] == pytest.approx(0.7)  # month 2


def test_lag_does_not_cross_groups():
    """Lag from section A should not bleed into section B."""
    df_a = make_group([0.9, 0.8], section="A")
    df_b = make_group([0.1, 0.2], section="B")
    df = pd.concat([df_a, df_b], ignore_index=True)
    result = compute_lag_features(df)

    # Month 1 of each section should have NaN lag (no prior row)
    a_month1 = result[(result['section'] == 'A') & (result['month'] == 1)]
    b_month1 = result[(result['section'] == 'B') & (result['month'] == 1)]
    assert pd.isna(a_month1['lag_1_efficiency'].values[0])
    assert pd.isna(b_month1['lag_1_efficiency'].values[0])


# ── compute_calendar_features ─────────────────────────────────────────────────

def test_is_monsoon_flag():
    df = make_group([0.5] * 12)
    df['month'] = list(range(1, 13))
    result = compute_calendar_features(df)
    assert result[result['month'] == 7]['is_monsoon'].values[0] == 1
    assert result[result['month'] == 11]['is_monsoon'].values[0] == 0


def test_is_january_flag():
    df = make_group([0.5, 0.5])
    df['month'] = [1, 2]
    result = compute_calendar_features(df)
    assert result[result['month'] == 1]['is_january'].values[0] == 1
    assert result[result['month'] == 2]['is_january'].values[0] == 0


def test_is_financial_year_end():
    df = make_group([0.5] * 3)
    df['month'] = [2, 3, 4]
    result = compute_calendar_features(df)
    assert result[result['month'] == 3]['is_financial_year_end'].values[0] == 1


def test_month_cyclical_encoding():
    """month_sin and month_cos should be in [-1, 1]."""
    df = make_group([0.5] * 12)
    df['month'] = list(range(1, 13))
    result = compute_calendar_features(df)
    assert result['month_sin'].between(-1, 1).all()
    assert result['month_cos'].between(-1, 1).all()


# ── compute_category_features ─────────────────────────────────────────────────

def test_category_group_mapping():
    df = make_group([0.5], category="DS")
    result = compute_category_features(df)
    assert result['category_group'].iloc[0] == 'Slum'


def test_industrial_mapping():
    df = make_group([0.5], category="I1")
    result = compute_category_features(df)
    assert result['category_group'].iloc[0] == 'Industrial'


# ── compute_growth_features ───────────────────────────────────────────────────

def test_demand_per_can():
    df = make_group([0.5, 0.5])
    df['demand'] = [1000.0, 1000.0]
    df['noofcans'] = [50, 50]
    df = compute_lag_features(df)
    result = compute_growth_features(df)
    assert result['demand_per_can'].iloc[0] == pytest.approx(20.0)


# ── drop_warmup_rows ──────────────────────────────────────────────────────────

def test_drop_warmup_removes_null_lag():
    """Rows with null lag_1_efficiency should be dropped."""
    df = make_group([0.8, 0.7, 0.6])
    df = compute_lag_features(df)
    result = drop_warmup_rows(df)
    assert result['lag_1_efficiency'].isna().sum() == 0
    assert len(result) == 2  # month 1 dropped, months 2 and 3 kept


def test_drop_warmup_short_group():
    """A group with only 1 row should have 0 rows after warmup drop."""
    df = make_group([0.5])
    df = compute_lag_features(df)
    result = drop_warmup_rows(df)
    assert len(result) == 0