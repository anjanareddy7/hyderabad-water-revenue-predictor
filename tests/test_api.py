import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "section": "KPHB",
    "division": 7,
    "category": "D",
    "year": 2026,
    "month": 6,
    "last_3_efficiency": [0.55, 0.48, 0.61],
    "last_month_demand": 850000,
    "last_month_noofcans": 4500,
    "current_demand": 870000,
    "current_noofcans": 4520,
}


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_unknown_category_returns_422(client):
    payload = {**VALID_PAYLOAD, "category": "ZZ"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_negative_demand_returns_422(client):
    payload = {**VALID_PAYLOAD, "current_demand": -500}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_invalid_efficiency_range_returns_422(client):
    payload = {**VALID_PAYLOAD, "last_3_efficiency": [0.5, 0.6, 3.5]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_invalid_month_returns_422(client):
    payload = {**VALID_PAYLOAD, "month": 13}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_invalid_year_returns_422(client):
    payload = {**VALID_PAYLOAD, "year": 2030}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_unknown_section_returns_404(client):
    payload = {**VALID_PAYLOAD, "section": "NONEXISTENT SECTION XYZ"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_slum_section_predicts_low_efficiency(client):
    payload = {
        **VALID_PAYLOAD,
        "section": "BAHADURPURA",
        "category": "DS",
        "last_3_efficiency": [0.03, 0.02, 0.04],
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_efficiency"] < 0.4


def test_industrial_section_predicts_high_efficiency(client):
    payload = {
        **VALID_PAYLOAD,
        "section": "BAHADURPURA",
        "category": "I1",
        "last_3_efficiency": [0.97, 0.98, 0.96],
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_efficiency"] > 0.7


def test_shortfall_computed_correctly(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    # Use the rounded efficiency from the response, not the raw prediction
    expected = VALID_PAYLOAD["current_demand"] * (1 - data["predicted_efficiency"])
    assert abs(data["predicted_shortfall_rupees"] - expected) < 50.0


def test_high_uncertainty_when_structural_change(client):
    payload = {
        **VALID_PAYLOAD,
        "last_month_noofcans": 4000,
        "current_noofcans": 4600,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["high_uncertainty"] is True


def test_cold_start_flag(client):
    payload = {**VALID_PAYLOAD, "last_3_efficiency": [0.0, 0.0, 0.0]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["cold_start"] is True


def test_response_has_all_fields(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    for field in [
        "predicted_efficiency", "predicted_shortfall_rupees",
        "predicted_shortfall_crore", "risk_tier",
        "cold_start", "high_uncertainty", "model_version"
    ]:
        assert field in data


def test_sections_endpoint(client):
    r = client.get("/sections")
    assert r.status_code == 200
    assert "sections" in r.json()
    assert len(r.json()["sections"]) > 0


def test_categories_endpoint(client):
    r = client.get("/categories")
    assert r.status_code == 200
    assert "categories" in r.json()
    assert "D" in r.json()["categories"]