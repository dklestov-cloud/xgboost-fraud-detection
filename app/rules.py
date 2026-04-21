"""
app/rules.py — hardcoded rules engine. No config surface.

A rule looks at a transaction (+ side inputs like velocity counters) and emits
zero or more RuleHit entries. Each hit carries a bump the aggregator adds to
the final score, plus a code/detail pair for the reasons payload.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── config (hardcoded, per simplification principle) ───────────────────────

# TODO Should be loaded from a config file or env vars, but hardcoded for simplicity. 
# These are just examples, not based on actual data analysis.
AMOUNT_REVIEW_THRESHOLD: float = 5000.0
AMOUNT_DECLINE_THRESHOLD: float = 20000.0

VELOCITY_BURST_COUNT_60S: int = 5          # > this in 60s → decline bump

HIGH_RISK_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "mail.ru",
    "yandex.ru",
    "protonmail.com",
    "guerrillamail.com",
    "tempmail.com",
    "10minutemail.com",
})

# ── types ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RuleHit:
    code: str
    detail: str
    bump: float  # added to aggregator score (before clipping)

# ── rules ──────────────────────────────────────────────────────────────────

def _amount_rule(txn: dict) -> list[RuleHit]:
    amount = txn.get("TransactionAmt")
    if amount is None:
        return []
    if amount >= AMOUNT_DECLINE_THRESHOLD:
        return [RuleHit(
            code="AMOUNT_OVER_DECLINE",
            detail=f"amount ${amount:,.2f} ≥ ${AMOUNT_DECLINE_THRESHOLD:,.0f}",
            bump=0.9,
        )]
    if amount >= AMOUNT_REVIEW_THRESHOLD:
        return [RuleHit(
            code="AMOUNT_OVER_REVIEW",
            detail=f"amount ${amount:,.2f} ≥ ${AMOUNT_REVIEW_THRESHOLD:,.0f}",
            bump=0.3,
        )]
    return []


def _email_rule(txn: dict) -> list[RuleHit]:
    domain = txn.get("P_emaildomain")
    if not domain:
        return []
    if domain.lower() in HIGH_RISK_EMAIL_DOMAINS:
        return [RuleHit(
            code="HIGH_RISK_EMAIL_DOMAIN",
            detail=f"purchaser email domain '{domain}' on risk list",
            bump=0.4,
        )]
    return []


# TODO add velocity rule - tracks how many transactions occur within a specific time window (60s) for a given entity.


def evaluate(txn: dict) -> list[RuleHit]:
    """Run all rules. Order is stable for reproducible reasons."""
    hits: list[RuleHit] = []
    hits.extend(_amount_rule(txn))
    hits.extend(_email_rule(txn))
    return hits


def total_bump(hits: list[RuleHit]) -> float:
    """Sum of bumps, clipped to [0, 1]. Fed into the aggregator."""
    return max(0.0, min(1.0, sum(h.bump for h in hits)))
