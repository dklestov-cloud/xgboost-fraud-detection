"""
scripts/smoke_test.py — fast check of API without real data.

Run (after uvicorn app.main:app --reload):
    python scripts/smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys

import requests

BASE_URL = "http://localhost:8000"


def ok(label: str):
    print(f"  ✅ {label}")


def fail(label: str, detail: str):
    print(f"  ❌ {label}: {detail}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"\n🔍 Smoke test → {base}\n")

    # ── /health ──────────────────────────────────────────────────────────
    r = requests.get(f"{base}/health")
    r.raise_for_status()
    h = r.json()
    print("GET /health")
    ok(f"status={h['status']}, model_loaded={h['model_loaded']}")
    print(f"     val_sample_size={h['val_sample_size']}, test_sample_size={h['test_sample_size']}")

    if not h["model_loaded"]:
        print("\n⚠️  Model not loaded. Run scripts/train.py first. Skipping predict tests.")
        return

    # ── /sample?source=val ───────────────────────────────────────────────
    r = requests.get(f"{base}/sample", params={"source": "val", "n": 5})
    r.raise_for_status()
    s = r.json()
    print(f"\nGET /sample?source=val&n=5")
    ok(f"Got {len(s['records'])} records, total_available={s['total_available']}")

    # ── /evaluate ────────────────────────────────────────────────────────
    transactions = []
    for rec in s["records"]:
        t = rec["transaction"].copy()
        if rec["ground_truth"] is not None:
            t["isFraud"] = rec["ground_truth"]
        transactions.append(t)

    r = requests.post(f"{base}/evaluate", json={"transactions": transactions})
    r.raise_for_status()
    ev = r.json()
    print(f"\nPOST /evaluate ({len(transactions)} transactions)")
    ok(f"ROC-AUC={ev['metrics']['roc_auc']}, accuracy={ev['metrics']['accuracy']:.3f}")

    # ── /predict (JSON) ──────────────────────────────────────────────────
    tx_no_label = [dict(t, isFraud=None) for t in transactions]
    # remove isFraud from request
    for t in tx_no_label:
        t.pop("isFraud", None)

    r = requests.post(f"{base}/predict", json={"transactions": tx_no_label})
    r.raise_for_status()
    pr = r.json()
    print(f"\nPOST /predict ({len(tx_no_label)} transactions)")
    ok(f"Got {len(pr['predictions'])} predictions, model_version={pr['model_version']}")

    # ── /sample?source=test (optional) ────────────────────────────────
    r2 = requests.get(f"{base}/sample", params={"source": "test", "n": 3})
    if r2.status_code == 200:
        s2 = r2.json()
        print(f"\nGET /sample?source=test&n=3")
        ok(f"Got {len(s2['records'])} records (no groundtruth)")
    else:
        print(f"\nGET /sample?source=test  → {r2.status_code} (test_sample.csv not generated, OK)")

    print("\n✅ All tests passed!\n")


if __name__ == "__main__":
    main()