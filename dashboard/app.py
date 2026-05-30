import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

API_URL  = "http://localhost:8000"
ROOT     = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="HMWSSB Collection Efficiency Dashboard",
    page_icon="💧",
    layout="wide"
)

# ── Load static data ──────────────────────────────────────────────────────────

@st.cache_data
def load_risk_tiers():
    return pd.read_csv(PROCESSED / "section_risk_tiers.csv")

@st.cache_data
def load_test_predictions():
    return pd.read_parquet(PROCESSED / "test_predictions.parquet")

@st.cache_data
def get_sections():
    try:
        r = requests.get(f"{API_URL}/sections", timeout=5)
        return r.json()["sections"]
    except:
        risk_tiers = load_risk_tiers()
        return sorted(risk_tiers["section"].unique().tolist())

# ── API helpers ───────────────────────────────────────────────────────────────

def predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

# ── Risk tier badge ───────────────────────────────────────────────────────────

def tier_badge(tier: str) -> str:
    colors = {
        "High Risk":     "#ef4444",
        "Medium Risk":   "#f97316",
        "Moderate Risk": "#eab308",
        "Low Risk":      "#22c55e",
    }
    color = colors.get(tier, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem">{tier}</span>'

# ── Main app ──────────────────────────────────────────────────────────────────

st.title("💧 HMWSSB Collection Efficiency Dashboard")
st.caption("Hyderabad Metropolitan Water Supply & Sewerage Board — Revenue Intelligence")

# API status
api_ok = check_api()
if api_ok:
    st.success("API connected", icon="✅")
else:
    st.warning("API not reachable — start uvicorn locally or check Docker container", icon="⚠️")

# ── Top metrics ───────────────────────────────────────────────────────────────

risk_tiers = load_risk_tiers()
preds      = load_test_predictions()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sections tracked",   f"{risk_tiers['section'].nunique():,}")
col2.metric("High Risk sections",        f"{(risk_tiers['risk_tier'] == 'High Risk').sum():,}")
col3.metric("Avg collection efficiency", f"{preds['efficiency'].mean()*100:.1f}%")
col4.metric("Annual shortfall (crore)",  "₹531")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🏢 Field Manager View", "📊 Data / Ops View"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Field Manager View
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Division-level prediction summary")
    st.caption("Select a division to see next-month predictions for all sections, sorted by risk.")

    divisions = sorted(preds['division'].unique().tolist())
    selected_division = st.selectbox("Select division", divisions, key="div_select")

    if st.button("Generate predictions", type="primary"):
        division_sections = preds[preds['division'] == selected_division].copy()
        section_latest = (
            division_sections
            .sort_values(['section', 'category', 'year', 'month'])
            .groupby(['section', 'category'])
            .last()
            .reset_index()
        )

        results = []
        progress = st.progress(0)
        total = len(section_latest)

        for i, row in section_latest.iterrows():
            payload = {
                "section":             row['section'],
                "division":            int(selected_division),
                "category":            row['category'],
                "year":                2026,
                "month":               5,
                "last_3_efficiency":   [
                    float(row.get('lag_1_efficiency', 0.5) or 0.5),
                    float(row.get('lag_2_efficiency', 0.5) or 0.5),
                    float(row.get('lag_3_efficiency', 0.5) or 0.5),
                ],
                "last_month_demand":   float(row['demand']),
                "last_month_noofcans": int(row.get('noofcans', 100)),
                "current_demand":      float(row['demand']),
                "current_noofcans":    int(row.get('noofcans', 100)),
            }
            result = predict(payload)
            if result:
                results.append({
                    "Section":       row['section'],
                    "Category":      row['category'],
                    "Predicted %":   f"{result['predicted_efficiency']*100:.1f}%",
                    "Shortfall (₹)": f"₹{result['predicted_shortfall_rupees']:,.0f}",
                    "Risk Tier":     result['risk_tier'],
                    "_efficiency":   result['predicted_efficiency'],
                    "_shortfall":    result['predicted_shortfall_rupees'],
                })
            progress.progress(min((list(section_latest.index).index(i) + 1) / total, 1.0))

        if results:
            df_results = pd.DataFrame(results).sort_values('_efficiency')

            # Total rupees at risk for High Risk sections
            high_risk_shortfall = sum(
                r['_shortfall'] for r in results if r['Risk Tier'] == 'High Risk'
            )
            st.metric(
                "Total rupees at risk (High Risk sections)",
                f"₹{high_risk_shortfall/1e7:.2f} crore"
            )

            # Table with badges
            st.markdown("### Sections sorted by predicted efficiency (worst first)")
            for _, row in df_results.iterrows():
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
                c1.write(row['Section'])
                c2.write(row['Category'])
                c3.write(row['Predicted %'])
                c4.write(row['Shortfall (₹)'])
                c5.markdown(tier_badge(row['Risk Tier']), unsafe_allow_html=True)
        else:
            st.info("No predictions returned — check API connection.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data / Ops View
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Section-level analysis")

    sections   = get_sections()
    categories = sorted(preds['category'].unique().tolist())

    col_a, col_b = st.columns(2)
    selected_section  = col_a.selectbox("Select section",  sections,   key="sec_select")
    selected_category = col_b.selectbox("Select category", categories, key="cat_select")

    section_data = preds[
        (preds['section'] == selected_section) &
        (preds['category'] == selected_category)
    ].sort_values(['year', 'month']).copy()

    if len(section_data) == 0:
        st.info("No data found for this section-category combination.")
    else:
        section_data['period'] = (
            section_data['year'].astype(str) + "-" +
            section_data['month'].astype(str).str.zfill(2)
        )

        # ── Actual vs predicted line chart ──
        st.markdown("#### Actual vs predicted efficiency (last 12 months)")
        last_12 = section_data.tail(12)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=last_12['period'], y=last_12['efficiency'],
            name="Actual", line=dict(color="#3b82f6", width=2),
            mode="lines+markers"
        ))
        if 'predicted_efficiency' in last_12.columns:
            fig.add_trace(go.Scatter(
                x=last_12['period'], y=last_12['predicted_efficiency'],
                name="Predicted", line=dict(color="#f97316", width=2, dash="dash"),
                mode="lines+markers"
            ))
        fig.update_layout(
            yaxis_title="Collection efficiency",
            xaxis_title="Month",
            yaxis=dict(tickformat=".0%", range=[0, 1.2]),
            height=350, margin=dict(t=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Model scorecard ──
        st.markdown("#### Model scorecard")
        if 'predicted_efficiency' in section_data.columns:
            mae_model    = (section_data['efficiency'] - section_data['predicted_efficiency']).abs().mean()
            mae_baseline = (section_data['efficiency'] - section_data['lag_1_efficiency']).abs().mean()
            improvement  = (mae_baseline - mae_model) / mae_baseline * 100 if mae_baseline > 0 else 0

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Model MAE",    f"{mae_model:.3f}")
            sc2.metric("Baseline MAE", f"{mae_baseline:.3f}")
            sc3.metric("Improvement",  f"{improvement:.1f}%")

        # ── Risk tier and shortfall ──
        st.markdown("#### Section summary")
        tier_row = risk_tiers[
            (risk_tiers['section'] == selected_section) &
            (risk_tiers['category_group'].isin(
                [section_data['category_group'].iloc[0]]
                if 'category_group' in section_data.columns else ['Domestic']
            ))
        ]
        if len(tier_row) > 0:
            tier = tier_row['risk_tier'].values[0]
            mean_eff = tier_row['mean_efficiency'].values[0]
            st.markdown(
                f"Risk tier: {tier_badge(tier)} &nbsp;&nbsp; "
                f"Historical mean efficiency: **{mean_eff*100:.1f}%**",
                unsafe_allow_html=True
            )

        total_shortfall = section_data.get(
            'shortfall',
            (section_data['demand'] - section_data['collection']).clip(lower=0)
        ).sum()
        st.metric(
            "Total shortfall this section (test period)",
            f"₹{total_shortfall/1e7:.2f} crore"
        )

        # ── Category distribution chart ──
        st.markdown("#### Efficiency distribution by category group (full dataset)")
        cat_dist = preds.groupby('category_group')['efficiency'].median().reset_index()
        cat_dist.columns = ['Category Group', 'Median Efficiency']
        cat_dist = cat_dist.sort_values('Median Efficiency')

        fig2 = px.bar(
            cat_dist, x='Median Efficiency', y='Category Group',
            orientation='h', color='Median Efficiency',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1],
            title="Median collection efficiency by category group"
        )
        fig2.update_layout(height=300, margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)