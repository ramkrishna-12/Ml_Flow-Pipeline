
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