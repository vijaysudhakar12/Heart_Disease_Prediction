"""Feature engineering for the Heart Disease UCI dataset.

Location in project: src/features/feature_engineering.py

This module works on *cleaned* data (output of src/data/data_cleaning.py).
It only does stateless, row-level work (no statistics learned from data), so it
is safe to apply to both training data and new data at prediction time.
Anything that must be *fitted* (encoders, scalers) lives in preprocessing.py.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

RAW_TARGET_COL = "num"      # 0 = no disease, 1-4 = disease severity
TARGET_COL = "target"       # binary: 0 = no disease, 1 = disease
ID_COL = "id"               # identifier, carries no medical information


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the multi-class `num` column (0-4) into a binary `target`."""
    df = df.copy()
    if RAW_TARGET_COL not in df.columns:
        return df  # prediction time: no label available

    df[TARGET_COL] = (df[RAW_TARGET_COL] > 0).astype(int)
    df = df.drop(columns=RAW_TARGET_COL)
    logger.info("Target distribution:\n%s", df[TARGET_COL].value_counts().to_string())
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that must never be used as model inputs."""
    return df.drop(columns=[ID_COL], errors="ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all stateless feature engineering steps and return a new dataframe.

    Add any new derived features here (e.g. age groups, ratios) so training
    and prediction always go through exactly the same code.
    """
    df = df.copy()
    df = drop_identifier_columns(df)
    df = create_target(df)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered dataframe into X (features) and y (target)."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' not found. Run engineer_features() first.")
    return df.drop(columns=TARGET_COL), df[TARGET_COL]
