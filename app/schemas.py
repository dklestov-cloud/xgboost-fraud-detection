"""
app/schemas.py — Pydantic-schemes for API queries and responses.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, Literal


# ── input data ─────────────────────────────────────────────────────────

class TransactionRecord(BaseModel):
    """
    Single transaction. All fields are optional but TransactionID, since the real data are sparse
    """
    TransactionID: int
    TransactionDT: Optional[float] = None
    TransactionAmt: Optional[float] = None
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D6: Optional[float] = None
    D7: Optional[float] = None
    D8: Optional[float] = None
    D9: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D12: Optional[float] = None
    D13: Optional[float] = None
    D14: Optional[float] = None
    D15: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None
    # identity
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_01: Optional[float] = None
    id_02: Optional[float] = None
    id_03: Optional[float] = None
    id_04: Optional[float] = None
    id_05: Optional[float] = None
    id_06: Optional[float] = None
    id_07: Optional[float] = None
    id_08: Optional[float] = None
    id_09: Optional[float] = None
    id_10: Optional[float] = None
    id_11: Optional[float] = None
    id_12: Optional[str] = None
    id_13: Optional[float] = None
    id_14: Optional[float] = None
    id_15: Optional[str] = None
    id_16: Optional[str] = None
    id_17: Optional[float] = None
    id_18: Optional[float] = None
    id_19: Optional[float] = None
    id_20: Optional[float] = None
    id_21: Optional[float] = None
    id_22: Optional[float] = None
    id_23: Optional[str] = None
    id_24: Optional[float] = None
    id_25: Optional[float] = None
    id_26: Optional[float] = None
    id_27: Optional[str] = None
    id_28: Optional[str] = None
    id_29: Optional[str] = None
    id_30: Optional[str] = None
    id_31: Optional[str] = None
    id_32: Optional[float] = None
    id_33: Optional[str] = None
    id_34: Optional[str] = None
    id_35: Optional[str] = None
    id_36: Optional[str] = None
    id_37: Optional[str] = None
    id_38: Optional[str] = None

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    transactions: list[TransactionRecord] = Field(..., min_length=1, max_length=1000)


# ── output data ────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    TransactionID: int
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model_version: str


# ── test data source ───────────────────────────────────────────────

class SampleRecord(BaseModel):
    transaction: dict[str, Any]
    ground_truth: Optional[int] = Field(None, description="isFraud, if known")


class SampleResponse(BaseModel):
    records: list[SampleRecord]
    source: str   # "validation" or "test"
    total_available: int


# ── health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    val_sample_size: int
    test_sample_size: int


# ── /score ──────────────────────────────────────────────────

class Reason(BaseModel):
    source: Literal["rule", "xgb", "gnn"]
    code: str
    detail: str
    contribution: Optional[float] = None
