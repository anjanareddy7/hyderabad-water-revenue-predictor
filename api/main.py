from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from pathlib import Path
from src.division_forecast import get_division_summary, get_all_divisions_forecast
from src.anomaly import score_anomaly
import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parents[1]
MODELS    = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

# ── Globals (populated on startup) ───────────────────────────────────────────

model            = None
section_mapping  = None
division_mapping = None
feature_cols     = None
risk_tiers       = None
fallback_medians = None

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, section_mapping, division_mapping
    global feature_cols, risk_tiers, fallback_medians

    model            = joblib.load(MODELS / "best_model.pkl")
    section_mapping  = joblib.load(MODELS / "section_te_mapping.pkl")
    division_mapping = joblib.load(MODELS / "division_te_mapping.pkl")
    feature_cols     = joblib.load(MODELS / "feature_cols.pkl")
    risk_tiers       = pd.read_csv(PROCESSED / "section_risk_tiers.csv")

    preds = pd.read_parquet(PROCESSED / "test_predictions.parquet")
    fallback_medians = (
        preds.groupby('category_group')['efficiency'].median().to_dict()
    )
    logger.info("All artifacts loaded successfully")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HMWSSB Collection Efficiency API",
    description="Predicts monthly water bill collection efficiency for Hyderabad sections",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Known values ──────────────────────────────────────────────────────────────

KNOWN_CATEGORIES = {
    'C','D','M0','DS','M1','DM','M3','M2','N','O','RC','PS',
    'I1','M4','FS','I','CH','V','IW','BC','S','X','CB','GP','XO','MB','MS'
}

CATEGORY_MAP = {
    'D':'Domestic','DM':'Domestic','M0':'Domestic','M1':'Domestic',
    'M2':'Domestic','M3':'Domestic','M4':'Domestic','MS':'Domestic',
    'DP':'Domestic','T1':'Domestic','T2':'Domestic','T3':'Domestic','T4':'Domestic',
    'DS':'Slum','PS':'Slum','GP':'Slum',
    'C':'Commercial','BC':'Commercial','C1':'Commercial',
    'FS':'Commercial','XO':'Commercial','X':'Commercial',
    'I':'Industrial','I1':'Industrial','IW':'Industrial',
    'G':'GovtInstitutional','H':'GovtInstitutional','RC':'GovtInstitutional',
    'CH':'GovtInstitutional','CB':'GovtInstitutional','MB':'GovtInstitutional',
    'N':'GovtInstitutional','O':'GovtInstitutional','S':'GovtInstitutional',
    'V':'GovtInstitutional',
}

CATEGORY_GROUP_ENCODING = {
    'Slum':0,'GovtInstitutional':1,'Domestic':2,'Commercial':3,'Industrial':4
}

RISK_TIER_ENCODING = {
    'High Risk':0,'Medium Risk':1,'Moderate Risk':2,'Low Risk':3
}


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    section:             str
    division:            int
    category:            str
    year:                int
    month:               int
    last_3_efficiency:   list[float]
    last_month_demand:   float
    last_month_noofcans: int
    current_demand:      float
    current_noofcans:    int

    @field_validator('category')
    @classmethod
    def category_must_be_known(cls, v):
        if v not in KNOWN_CATEGORIES:
            raise ValueError(f"Unknown category '{v}'")
        return v

    @field_validator('last_3_efficiency')
    @classmethod
    def efficiency_range(cls, v):
        if any(e < 0 or e > 2 for e in v):
            raise ValueError("Efficiency values must be between 0 and 2")
        return v

    @field_validator('current_demand')
    @classmethod
    def demand_positive(cls, v):
        if v <= 0:
            raise ValueError("current_demand must be greater than 0")
        return v

    @field_validator('year')
    @classmethod
    def year_range(cls, v):
        if not (2022 <= v <= 2027):
            raise ValueError("year must be between 2022 and 2027")
        return v

    @field_validator('month')
    @classmethod
    def month_range(cls, v):
        if not (1 <= v <= 12):
            raise ValueError("month must be between 1 and 12")
        return v


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_efficiency:       float
    predicted_shortfall_rupees: float
    predicted_shortfall_crore:  float
    risk_tier:                  str
    cold_start:                 bool
    high_uncertainty:           bool
    is_anomaly:                 bool
    anomaly_score:              float
    model_version:              str = "1.0.0"


# ── Prediction logic ──────────────────────────────────────────────────────────

def build_features(req: PredictRequest):
    category_group = CATEGORY_MAP.get(req.category, 'Other')

    cold_start = len(req.last_3_efficiency) < 1 or all(
        e == 0 for e in req.last_3_efficiency
    )

    if cold_start:
        fallback     = fallback_medians.get(category_group, 0.5)
        lag1 = lag2 = lag3 = fallback
        rolling_mean = fallback
        rolling_std  = 0.2
    else:
        vals         = req.last_3_efficiency
        lag1         = vals[0] if len(vals) > 0 else fallback_medians.get(category_group, 0.5)
        lag2         = vals[1] if len(vals) > 1 else lag1
        lag3         = vals[2] if len(vals) > 2 else lag2
        rolling_mean = float(np.mean(vals))
        rolling_std  = float(np.std(vals)) if len(vals) > 1 else 0.1

    demand_growth = (
        (req.current_demand - req.last_month_demand) / req.last_month_demand
        if req.last_month_demand > 0 else 0.0
    )
    noofcans_growth = (
        (req.current_noofcans - req.last_month_noofcans) / req.last_month_noofcans
        if req.last_month_noofcans > 0 else 0.0
    )
    structural_change = abs(noofcans_growth) > 0.05

    tier_row  = risk_tiers[
        (risk_tiers['section'] == req.section) &
        (risk_tiers['category_group'] == category_group)
    ]
    risk_tier         = tier_row['risk_tier'].values[0] if len(tier_row) > 0 else 'Medium Risk'
    risk_tier_encoded = RISK_TIER_ENCODING.get(risk_tier, 1)

    global_mean = 0.557
    section_te  = section_mapping.get(req.section, global_mean)
    division_te = division_mapping.get(req.division, global_mean)

    month    = req.month
    features = {
        'lag_1_efficiency':           lag1,
        'lag_2_efficiency':           lag2,
        'lag_3_efficiency':           lag3,
        'rolling_3m_efficiency_mean': rolling_mean,
        'rolling_3m_efficiency_std':  rolling_std,
        'demand_growth_rate':         demand_growth,
        'noofcans_growth':            noofcans_growth,
        'demand_per_can':             req.current_demand / max(req.current_noofcans, 1),
        'rolling_3m_demand_mean':     req.last_month_demand,
        'category_group_encoded':     CATEGORY_GROUP_ENCODING.get(category_group, 2),
        'risk_tier_encoded':          risk_tier_encoded,
        'is_monsoon':                 int(month in [6, 7, 8, 9]),
        'is_financial_year_end':      int(month == 3),
        'is_january':                 int(month == 1),
        'is_2022_baseline':           int(req.year == 2022),
        'month_sin':                  float(np.sin(2 * np.pi * month / 12)),
        'month_cos':                  float(np.cos(2 * np.pi * month / 12)),
        'approval_rate':              0.5,
        'is_structural_change':       int(structural_change),
        'section_te':                 section_te,
        'division_te':                division_te,
    }
    return features, risk_tier, cold_start, structural_change


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    known_sections = set(risk_tiers['section'].unique()) | set(section_mapping.keys())
    if req.section not in known_sections:
        raise HTTPException(status_code=404, detail=f"Section '{req.section}' not found")

    features, risk_tier, cold_start, structural_change = build_features(req)

    X    = pd.DataFrame([features])[feature_cols].fillna(-999)
    pred = float(np.clip(model.predict(X)[0], 0, 1))

    shortfall       = req.current_demand * (1 - pred)
    shortfall_crore = shortfall / 1e7

    high_uncertainty = (
        structural_change or
        features['rolling_3m_efficiency_std'] > 0.2 or
        cold_start
    )

    anomaly_result = score_anomaly(features)

    return PredictResponse(
        predicted_efficiency       = round(pred, 4),
        predicted_shortfall_rupees = round(shortfall, 2),
        predicted_shortfall_crore  = round(shortfall_crore, 4),
        risk_tier                  = risk_tier,
        cold_start                 = cold_start,
        high_uncertainty           = high_uncertainty,
        is_anomaly                 = anomaly_result['is_anomaly'],
        anomaly_score              = anomaly_result['anomaly_score'],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0.0"}


@app.get("/sections")
def get_sections():
    return {"sections": sorted(risk_tiers['section'].unique().tolist())}


@app.get("/categories")
def get_categories():
    return {"categories": sorted(list(KNOWN_CATEGORIES))}


@app.get("/forecast/divisions")
def forecast_all_divisions():
    """Get predicted collection forecast for all divisions."""
    forecasts = get_all_divisions_forecast()
    return {"divisions": forecasts}


@app.get("/forecast/division/{division_id}")
def forecast_division(division_id: int):
    """Get predicted collection forecast for a single division."""
    result = get_division_summary(division_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Division {division_id} not found"
        )
    return result


@app.get("/anomaly/summary")
def anomaly_summary():
    """Get summary of anomalous sections in the test period."""
    try:
        scores    = pd.read_parquet(PROCESSED / "anomaly_scores.parquet")
        anomalies = scores[scores['is_anomaly']].copy()

        top = (
            anomalies
            .sort_values('anomaly_score')
            [['section', 'category', 'year', 'month',
              'anomaly_score', 'demand_growth_rate', 'demand_per_can']]
            .head(20)
        )

        return {
            "total_scored":     int(len(scores)),
            "total_anomalies":  int(len(anomalies)),
            "anomaly_rate_pct": round(len(anomalies) / len(scores) * 100, 1),
            "top_anomalies":    top.fillna(0).to_dict(orient='records'),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))