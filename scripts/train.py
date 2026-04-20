"""
train.py — learning XGBoost model on IEEE-CIS Fraud Detection data.

Usage:
    python scripts/train.py \
        --train-transaction train_transaction.csv \
        --train-identity   train_identity.csv \
        [--test-transaction test_transaction.csv] \
        [--test-identity    test_identity.csv] \
        --output-dir        models/

Output artifacts:
    models/xgb_model.joblib        — trained model
    models/label_encoders.joblib   — dict {col: LabelEncoder}
    models/feature_columns.joblib  — feature list (after Vxxx-columns drop)
    models/val_sample.csv          — 500 lines from validation (with groundtruth)
    models/test_sample.csv         — 500 lines from test (without groundtruth)
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── categorical columns ─────────────────────────────────────
V_COLS = [f"V{i}" for i in range(1, 340)]

def get_cat_cols(X: pd.DataFrame) -> list[str]:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    extra = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    cat_cols += [c for c in extra if c not in cat_cols]
    return list(dict.fromkeys(cat_cols))          # deduplication, keeping order


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day"]  = (df["TransactionDT"] // (3600 * 24)) % 7
    return df


def load_and_merge(tx_path: str, id_path: str | None) -> pd.DataFrame:
    df = pd.read_csv(tx_path)
    if id_path and os.path.exists(id_path):
        df_id = pd.read_csv(id_path)
        df = df.merge(df_id, how="left", on="TransactionID")
    return df


def preprocess(df: pd.DataFrame,
               cat_cols: list[str],
               encoders: dict[str, LabelEncoder] | None = None,
               fit: bool = False) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Performs cat features encoding and adds time-features.
    if fit=True — uses LabelEncoder's and returns them.
    if fit=False — uses encoders from arg list.
    """
    df = add_time_features(df)
    df = df.drop(columns=[c for c in V_COLS if c in df.columns], errors="ignore")

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                # just enumerate
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                # unknown values → -1 via unseen-safe transform
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
                if "__unknown__" not in known:
                    le.classes_ = np.append(le.classes_, "__unknown__")
                df[col] = le.transform(df[col])

    return df, encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-transaction", required=True)
    parser.add_argument("--train-identity",    default=None)
    parser.add_argument("--test-transaction",  default=None)
    parser.add_argument("--test-identity",     default=None)
    parser.add_argument("--output-dir",        default="models")
    parser.add_argument("--val-split",         type=float, default=0.8)
    parser.add_argument("--sample-size",       type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── loading and merging ──────────────────────────────────────────────
    print("Loading train data...")
    df = load_and_merge(args.train_transaction, args.train_identity)

    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])

    cat_cols = get_cat_cols(X)
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # ── preprocessing + split ───────────────────────────────────────────────
    X, encoders = preprocess(X, cat_cols, fit=True)

    split = int(len(X) * args.val_split)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=50,
        n_jobs=-1,
        tree_method="hist",   # faster, than exact
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print(f"\nValidation ROC-AUC: {auc:.4f}")

    # ── artifacts saving ───────────────────────────────────────────────
    feature_columns = list(X_train.columns)

    joblib.dump(model,          os.path.join(args.output_dir, "xgb_model.joblib"))
    joblib.dump(encoders,       os.path.join(args.output_dir, "label_encoders.joblib"))
    joblib.dump(feature_columns, os.path.join(args.output_dir, "feature_columns.joblib"))
    joblib.dump(cat_cols,       os.path.join(args.output_dir, "cat_cols.joblib"))
    print("Saved model artifacts to", args.output_dir)

    # ── validation subset for API testing ──────────────────────────
    val_ids = df["TransactionID"].iloc[split:].reset_index(drop=True)
    val_raw = df.iloc[split:].reset_index(drop=True)   # with isFraud
    sample = val_raw.sample(min(args.sample_size, len(val_raw)), random_state=42)
    sample.to_csv(os.path.join(args.output_dir, "val_sample.csv"), index=False)
    print(f"Saved val_sample.csv ({len(sample)} rows)")

    # ── test subset (without groundtruth, if passed) ──────────────────
    if args.test_transaction:
        print("Loading test data...")
        df_test = load_and_merge(args.test_transaction, args.test_identity)
        test_sample = df_test.sample(min(args.sample_size, len(df_test)), random_state=42)
        test_sample.to_csv(os.path.join(args.output_dir, "test_sample.csv"), index=False)
        print(f"Saved test_sample.csv ({len(test_sample)} rows)")
    preds = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, preds)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
    plt.plot(fpr, thresholds, label="Thresholds")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()