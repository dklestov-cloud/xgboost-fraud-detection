"""
app/reasons.py — merges heterogeneous signals into a list[Reason].

Ordering (stable, for UI):
    1. Rule hits (most interpretable, deterministic)
    2. SHAP top-K contributions from XGBoost
    3. GNN band (only if non-zero)

All entries are capped at MAX_REASONS to keep payloads small.
"""

from __future__ import annotations

from app.rules import RuleHit
from app.schemas import Reason

MAX_REASONS: int = 8

# ── label map ──────────────────────────────────────────────────────────────────

# TODO This is a temporary mapping of feature name, should be moved to training pipeline and emitted as part of the model metadata. 
# Kept here for simplicity and to avoid coupling reasons builder to training code for now.

LABELS: dict[str, str] = {
    # transaction
    "TransactionAmt": "transaction amount",
    "TransactionDT":  "transaction timestamp",
    "ProductCD":      "product code",
    "hour":           "hour of day",
    "day":            "day of week",
    # card
    "card1": "card id (anonymized)",
    "card2": "card sub-id",
    "card3": "card region",
    "card4": "card network",
    "card5": "card bin",
    "card6": "card type",
    # address / distance
    "addr1": "billing region",
    "addr2": "billing country",
    "dist1": "distance (billing↔txn)",
    "dist2": "distance (secondary)",
    # email
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "receiver email domain",
    # counting features
    "C1":  "count 1d, same card",
    "C2":  "count 7d, same card",
    "C13": "distinct merchants, same card",
    "C14": "count, same device",
    # time-delta features
    "D1":  "days since prev card txn",
    "D4":  "days since first card txn",
    "D10": "days since address change",
    "D15": "days since email change",
    # match features
    "M4": "address match (name→billing)",
    "M6": "billing country matches card",
    # device / identity
    "DeviceType": "device type",
    "DeviceInfo": "device info",
    "id_30":      "device OS",
    "id_31":      "browser/app",
    "id_33":      "screen resolution",
    "id_38":      "session cookie present",
}


def humanize(feature: str) -> str:
    """Return a human-readable label for a model feature name.

    IEEE-CIS column codes → human-readable labels.

    Used by reasons builder to render SHAP feature contributions in plain English.
    Kept here temporarily; Stream 1 will migrate to training/label_map.py and make
    the training Job emit a per-model labels dict so ingest-api can load whichever
    labels match the active model.

    Missing entries fall back to the raw column name.
    """
    return LABELS.get(feature, feature)

# ── reasons ──────────────────────────────────────────────────────────────────

def build(
    rule_hits: list[RuleHit],
    shap_contributions: list[tuple[str, float]],
    gnn_risk_band: float,
) -> list[Reason]:
    """Merge heterogeneous signals into a list[Reason].

    Ordering (stable, for UI):
        1. Rule hits (most interpretable, deterministic)
        2. SHAP top-K contributions from XGBoost
        3. GNN band
    
    """
    out: list[Reason] = []

    for hit in rule_hits:
        out.append(Reason(
            source="rule",
            code=hit.code,
            detail=hit.detail,
        ))

    for feature, contribution in shap_contributions:
        direction = "↑ fraud" if contribution > 0 else "↓ fraud"
        out.append(Reason(
            source="xgb",
            code=feature.upper(),
            detail=f"{humanize(feature)} {direction}",
            contribution=round(contribution, 4),
        ))

    if gnn_risk_band >= 1.0:
        out.append(Reason(
            source="gnn",
            code="NETWORK_RISK_ELEVATED",
            detail="card embedding close to known-fraud cluster",
        ))
    elif gnn_risk_band >= 0.5:
        out.append(Reason(
            source="gnn",
            code="NETWORK_RISK_UNUSUAL",
            detail="card embedding in transition zone",
        ))

    return out[:MAX_REASONS]

