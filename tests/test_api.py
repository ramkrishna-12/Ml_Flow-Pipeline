from fastapi.testclient import TestClient

from app.main import create_app
from src import pipeline

GOOD = {
    "amount": 950.0,
    "hour": 3,
    "merchant_risk": 0.8,
    "distance_from_home_km": 400.0,
    "txn_count_24h": 9,
    "account_age_days": 20,
    "is_foreign": 1,
}
SAFE = {
    **GOOD,
    "amount": 12.0,
    "hour": 14,
    "merchant_risk": 0.02,
    "distance_from_home_km": 2.0,
    "txn_count_24h": 1,
    "account_age_days": 3000,
    "is_foreign": 0,
}


def test_api_not_ready_without_model(env):
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503


def test_api_predicts_with_champion(env):
    pipeline.run(sweep=False, rows=4000, out=str(env / "r.json"))
    with TestClient(create_app()) as c:
        r = c.get("/ready")
        assert r.status_code == 200 and r.json()["version"] == "1"
        out = c.post("/predict", json={"transactions": [GOOD, SAFE]}).json()
        risky, safe = (p["fraud_probability"] for p in out["predictions"])
        assert 0 <= safe <= 1 and 0 <= risky <= 1
        assert risky > safe  # sanity: model learned the obvious signal


def test_api_validates_input(env):
    with TestClient(create_app()) as c:
        assert c.post("/predict", json={"transactions": [{**GOOD, "hour": 99}]}).status_code == 422
