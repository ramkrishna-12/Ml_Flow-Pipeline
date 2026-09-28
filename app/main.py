
import logging
import os
import threading

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from src.config import FEATURES, get_settings

log = logging.getLogger("fraud-api")


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1)
    distance_from_home_km: float = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    account_age_days: int = Field(ge=0)
    is_foreign: int = Field(ge=0, le=1)



class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool


class PredictResponse(BaseModel):
    model_name: str
    model_version: str | None
    predictions: list[Prediction]
    
    

class ModelHolder:
    def __init__(self) -> None:
        self.s = get_settings()
        self.uri = os.getenv("MODEL_URI", f"models:/{self.s.model_name}@{self.s.champion_alias}")
        self.model = None
        self.version: str | None = None
        self.run_id: str | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            try:
                mlflow.set_tracking_uri(self.s.tracking_uri)
                self.model = mlflow.pyfunc.load_model(self.uri)
                self.run_id = self.model.metadata.run_id
                self.version = None
                if self.uri.startswith("models:/") and "@" in self.uri:
                    name, alias = self.uri[len("models:/") :].split("@")
                    self.version = str(MlflowClient().get_model_version_by_alias(name, alias).version)
                log.info("loaded %s (version=%s run=%s)", self.uri, self.version, self.run_id)
                return True
            except Exception:  # noqa: BLE001 - we want to keep the process alive and report not-ready
                log.exception("model load failed for %s", self.uri)
                self.model = None
                return False



def create_app() -> FastAPI:
    holder = ModelHolder()
    app = FastAPI(title="Fraud Detection API", version="1.0.0")
    app.state.holder = holder

    @app.on_event("startup")
    def _startup() -> None:
        holder.load()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if holder.model is None and not holder.load():  # lazy retry so a late MLflow doesn't need a restart
            raise HTTPException(503, "model not loaded")
        return {
            "status": "ready",
            "model_uri": holder.uri,
            "version": holder.version,
            "run_id": holder.run_id,
        }

    @app.post("/reload")
    def reload():
        """Re-resolve the alias without restarting the container (e.g. after a promotion)."""
        if not holder.load():
            raise HTTPException(503, "reload failed; previous model unavailable")
        return {"version": holder.version, "run_id": holder.run_id}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if holder.model is None and not holder.load():
            raise HTTPException(503, "model not loaded")
        df = pd.DataFrame([t.model_dump() for t in req.transactions])[FEATURES].astype("float64")
        out = np.asarray(holder.model.predict(df))
        proba = out[:, 1] if out.ndim == 2 else out
        thr = holder.s.decision_threshold
        return PredictResponse(
            model_name=holder.s.model_name,
            model_version=holder.version,
            predictions=[Prediction(fraud_probability=float(p), is_fraud=bool(p >= thr)) for p in proba],
        )

    return app


app = create_app()