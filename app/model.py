"""
app/model.py — singleton for model artifacts loading
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


@dataclass
class ModelArtifacts:
    model: Optional[XGBClassifier] = None
    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    val_sample: Optional[pd.DataFrame] = None   # with groundtruth
    test_sample: Optional[pd.DataFrame] = None  # without groundtruth
    loaded: bool = False


_artifacts = ModelArtifacts()


def load_artifacts(model_dir: str = MODEL_DIR) -> None:
    global _artifacts

    model_path    = os.path.join(model_dir, "xgb_model.joblib")
    enc_path      = os.path.join(model_dir, "label_encoders.joblib")
    feat_path     = os.path.join(model_dir, "feature_columns.joblib")
    cat_path      = os.path.join(model_dir, "cat_cols.joblib")
    val_path      = os.path.join(model_dir, "val_sample.csv")
    test_path     = os.path.join(model_dir, "test_sample.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run scripts/train.py first."
        )

    _artifacts.model           = joblib.load(model_path)
    _artifacts.encoders        = joblib.load(enc_path)
    _artifacts.feature_columns = joblib.load(feat_path)
    _artifacts.cat_cols        = joblib.load(cat_path)

    if os.path.exists(val_path):
        _artifacts.val_sample = pd.read_csv(val_path)

    if os.path.exists(test_path):
        _artifacts.test_sample = pd.read_csv(test_path)

    _artifacts.loaded = True


def get_artifacts() -> ModelArtifacts:
    return _artifacts