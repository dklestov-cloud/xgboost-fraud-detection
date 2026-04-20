"""
app/preprocessing.py — preprocessing pipeline for inference.
Uses the same LabelEncoder's, as in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

V_COLS = [f"V{i}" for i in range(1, 340)]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day"]  = (df["TransactionDT"] // (3600 * 24)) % 7
    return df


def apply_encoders(df: pd.DataFrame,
                   cat_cols: list[str],
                   encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Applies saved LabelEncoder's.
    Unknown values are replaced with '__unknown__' and encoded
    (encoder enhances at fit, so class always exists)
    """
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(str)
        le = encoders.get(col)
        if le is None:
            continue
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
        if "__unknown__" not in known:
            le.classes_ = np.append(le.classes_, "__unknown__")
        df[col] = le.transform(df[col])
    return df


def preprocess_for_inference(raw_df: pd.DataFrame,
                              cat_cols: list[str],
                              encoders: dict[str, LabelEncoder],
                              feature_columns: list[str]) -> pd.DataFrame:
    """
    Full pipeline: time-features → drop V* → encode cats → align columns.
    Returns DataFrame with the columns, that model expects.
    """
    df = add_time_features(raw_df)
    df = df.drop(columns=[c for c in V_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=["TransactionID", "isFraud"], errors="ignore")
    df = apply_encoders(df, cat_cols, encoders)

    # add missed columns as NaN, removing redundant columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_columns]

    return df