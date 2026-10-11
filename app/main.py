"""Expense account classifier API.

Run locally:  uv run uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from app.features import build_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/linear_svc.joblib"))
TOP_K = 3

state = {}


@asynccontextmanager
async def lifespan(app):
    # Load the model once at startup (saved by model_training.ipynb, section 7)
    state["model"] = joblib.load(MODEL_PATH)
    meta = MODEL_PATH.with_suffix(".json")
    state["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    yield


app = FastAPI(title="Peakflo expense classifier", lifespan=lifespan)


class Bill(BaseModel):
    vendorId: str
    itemName: str
    itemDescription: str = ""
    itemTotalAmount: float


class Candidate(BaseModel):
    accountName: str
    score: float  # SVM margin, higher = more confident (not a probability)


class Prediction(BaseModel):
    accountName: str
    top: list[Candidate]


def predict(bills):
    model = state["model"]
    scores = model.decision_function(build_features([b.model_dump() for b in bills]))
    classes = model.classes_
    results = []
    for row in np.atleast_2d(scores):
        best = np.argsort(row)[::-1][:TOP_K]
        top = [Candidate(accountName=classes[i], score=round(float(row[i]), 3)) for i in best]
        results.append(Prediction(accountName=top[0].accountName, top=top))
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model": state.get("meta", {})}


@app.post("/predict", response_model=Prediction)
def predict_one(bill: Bill):
    return predict([bill])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(bills: list[Bill]):
    return predict(bills)
