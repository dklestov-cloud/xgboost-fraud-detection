"""
app/main.py — FastAPI backend for fraud detection.

Endpoints:
    GET  /health                — service and model status
    POST /predict               — predict on transaction list (JSON)
    POST /predict/csv           — predict on loaded CSV-file
    GET  /sample?source=val&n=10 — test data from val_sample / test_sample
    POST /evaluate              — prediction + metrics (only for val, i.e.with isFraud)
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import roc_auc_score

from app.model import load_artifacts, get_artifacts, MODEL_VERSION
from app.preprocessing import preprocess_for_inference
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
    SampleRecord,
    SampleResponse,
)


# ── lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
        print(f"✅ Model loaded (version {MODEL_VERSION})")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    yield


# ── app ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based fraud detection for IEEE-CIS dataset",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────

def _predict_df(raw_df: pd.DataFrame) -> list[PredictionItem]:
    """Preprocessing + inference. Returns list of PredictionItem."""
    arts = get_artifacts()
    if not arts.loaded:
        raise HTTPException(503, "Model not loaded. Run scripts/train.py first.")

    transaction_ids = raw_df["TransactionID"].tolist()
    X = preprocess_for_inference(
        raw_df,
        arts.cat_cols,
        arts.encoders,
        arts.feature_columns,
    )

    probs = arts.model.predict_proba(X)[:, 1]

    return [
        PredictionItem(
            TransactionID=int(tid),
            fraud_probability=float(p),
            is_fraud=bool(p >= 0.5),
        )
        for tid, p in zip(transaction_ids, probs)
    ]


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Fraud Detection API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }

@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    arts = get_artifacts()
    return HealthResponse(
        status="ok",
        model_loaded=arts.loaded,
        model_version=MODEL_VERSION,
        val_sample_size=len(arts.val_sample) if arts.val_sample is not None else 0,
        test_sample_size=len(arts.test_sample) if arts.test_sample is not None else 0,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(body: PredictRequest):
    """
    Takes transaction list (JSON). Returns fraud probabilities.
    Fields V1–V339 are excluded from the model.
    """
    raw_df = pd.DataFrame([t.model_dump() for t in body.transactions])
    items = _predict_df(raw_df)
    return PredictResponse(predictions=items, model_version=MODEL_VERSION)


@app.post("/predict/csv", response_model=PredictResponse, tags=["prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Takes CSV-file with transaction data, returns prediction
    CSV should have TransactionID column.
    """
    contents = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse CSV: {e}")

    if "TransactionID" not in raw_df.columns:
        raise HTTPException(400, "CSV must contain 'TransactionID' column")

    items = _predict_df(raw_df)
    return PredictResponse(predictions=items, model_version=MODEL_VERSION)


@app.get("/sample", response_model=SampleResponse, tags=["testing"])
def get_sample(
    source: Literal["val", "test"] = Query("val", description="Source: val (with groundtruth) or test (without)"),
    n: int = Query(10, ge=1, le=200, description="Num of records to return"),
    random: bool = Query(True, description="Random samples"),
):
    """
    Returns test records from saved samples
    Use source=val for retrieving data with known groundtruth,
    source=test — without groundtruth (predicitons only).
    """
    arts = get_artifacts()
    if source == "val":
        df = arts.val_sample
        has_gt = True
    else:
        df = arts.test_sample
        has_gt = False

    if df is None:
        raise HTTPException(404, f"'{source}' sample not found. Run scripts/train.py.")

    total = len(df)
    subset = df.sample(min(n, total)) if random else df.head(n)

    records = []
    for _, row in subset.iterrows():
        row_dict = row.to_dict()
        gt = int(row_dict.pop("isFraud")) if (has_gt and "isFraud" in row_dict) else None
        records.append(SampleRecord(transaction=row_dict, ground_truth=gt))

    return SampleResponse(
        records=records,
        source=source,
        total_available=total,
    )


@app.post("/evaluate", tags=["testing"])
def evaluate(body: PredictRequest):
    """
    Like /predict, but estimates accuracy metrics.
    Requires isFraud associated with each transaction
    """
    raw_list = [t.model_dump() for t in body.transactions]
    raw_df = pd.DataFrame(raw_list)

    if "isFraud" not in raw_df.columns:
        raise HTTPException(400, "Field 'isFraud' required for evaluation. Use /predict for inference-only.")

    y_true = raw_df["isFraud"].values
    items = _predict_df(raw_df)
    y_prob = np.array([i.fraud_probability for i in items])
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None  # only one class in batch

    accuracy = float((y_pred == y_true).mean())
    fraud_rate_true = float(y_true.mean())
    fraud_rate_pred = float(y_pred.mean())

    return {
        "predictions": [i.model_dump() for i in items],
        "metrics": {
            "roc_auc": auc,
            "accuracy": accuracy,
            "fraud_rate_true": fraud_rate_true,
            "fraud_rate_pred": fraud_rate_pred,
            "n_samples": len(items),
        },
        "model_version": MODEL_VERSION,
    }