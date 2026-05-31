[![CI](https://github.com/anjanareddy7/hyderabad-water-revenue-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/anjanareddy7/hyderabad-water-revenue-predictor/actions/workflows/ci.yml)
# Hyderabad Water Revenue Predictor

Predicts monthly water bill collection efficiency across 500+ sections of Hyderabad's HMWSSB utility using 4 years of real government billing data (2022–2026, Telangana Open Data Portal).

## The problem

HMWSSB bills ₹531 crore in water charges every year that goes uncollected. Field collection teams have no way to know in advance which sections will underperform — they treat all defaulters the same. This project builds a prediction system that identifies high-risk sections a month in advance so intervention can be targeted where it recovers the most revenue.

## Key findings from EDA

- **₹531 crore lost annually** (2023–2026 average, excluding 2022 data artifact)
- **Domestic connections drive 65.8% of shortfall** despite only 44.8% of total demand
- **Slum connections have 1.4% median efficiency** but only 6.8% of total shortfall due to low volume
- **Industrial connections pay reliably** at 100% median efficiency
- **January anomaly**: ₹9.5 crore shortfall in January alone vs ₹1.6–2.8 crore other months

## Architecture

```
Raw CSVs (52 files, 2022-2026)
    ↓ src/etl.py
Cleaned Parquet (data/processed/)
    ↓ src/features.py
Feature matrix (93,973 rows x 31 features)
    ↓ src/train.py
XGBoost model + MLflow tracking
    ↓ api/main.py
FastAPI (Docker) → Streamlit dashboard
```

## Model performance

| Model | RMSE | MAE | vs Baseline |
|-------|------|-----|-------------|
| Naive lag-1 baseline | 0.4582 | 0.2294 | — |
| XGBoost | 0.3429 | 0.2119 | +25.2% |
| LightGBM | 0.3444 | 0.2136 | +24.9% |

Top features: `rolling_3m_efficiency_mean` (0.375), `lag_1_efficiency` (0.110), `risk_tier_encoded` (0.096)

No overfitting: train RMSE 0.3463 vs test RMSE 0.3429 (-1% gap)

## Risk tier clustering (KMeans, k=4, silhouette=0.46)

| Tier | Sections | Mean efficiency | Dominant category |
|------|----------|----------------|-------------------|
| High Risk | 214 | 6% | Slum |
| Medium Risk | 275 | 52% | Domestic |
| Moderate Risk | 148 | 61% | GovtInstitutional |
| Low Risk | 254 | 98% | Industrial/Commercial |

## Tech stack

- **Data**: Pandas, PyArrow, rapidfuzz
- **ML**: XGBoost, LightGBM, Scikit-learn, MLflow
- **API**: FastAPI, Pydantic, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Infrastructure**: Docker, DockerHub
- **Testing**: pytest (37 tests across ETL, features, API)

## Running locally

```bash
git clone https://github.com/anjanareddy7/hyderabad-water-revenue-predictor.git
cd hyderabad-water-revenue-predictor
pip install -r requirements.txt

python src/etl.py
python src/features.py
python src/train.py

uvicorn api.main:app --port 8000

streamlit run dashboard/app.py
```

## Running with Docker

```bash
docker pull anjanareddy7/hmwssb-api:latest
docker run -p 8000:8000 anjanareddy7/hmwssb-api:latest
```

## API usage

```bash
POST /predict

{
  "section": "KPHB",
  "division": 7,
  "category": "D",
  "year": 2026,
  "month": 6,
  "last_3_efficiency": [0.55, 0.48, 0.61],
  "last_month_demand": 850000,
  "last_month_noofcans": 4500,
  "current_demand": 870000,
  "current_noofcans": 4520
}

Response:

{
  "predicted_efficiency": 0.5364,
  "predicted_shortfall_rupees": 403293.56,
  "predicted_shortfall_crore": 0.0403,
  "risk_tier": "Medium Risk",
  "cold_start": false,
  "high_uncertainty": false,
  "model_version": "1.0.0"
}
```

## Tests

```bash
python -m pytest tests/ -v
# 37 tests, all passing
```

## Data source

Telangana Open Data Portal — HMWSSB billing and collection data 2022–2026
https://data.telangana.gov.in