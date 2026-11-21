"""Data cleaning for the Heart Disease UCI dataset.

Location in project: src/data/data_cleaning.py
Run from anywhere:   python src/data/data_cleaning.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import load_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_heart.csv"

# --------------------------------------------------------------------------
# Cleaning configuration
# --------------------------------------------------------------------------
DROP_COLUMNS = ["ca", "thal"]                      # too many missing values
ZERO_IS_MISSING = ["chol", "trestbps"]             # 0 is not a valid reading
MODE_COLUMNS = ["exang", "slope", "fbs", "restecg"]  # categorical / binary
MEDIAN_COLUMNS = ["trestbps", "oldpeak", "chol", "thalch"]  # numeric
OUTLIER_COLUMNS = ["trestbps", "thalch", "oldpeak"]
INT_COLUMNS = ["trestbps", "chol", "thalch"]
BOOL_COLUMNS = ["fbs", "exang"]


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in dataset: {missing}")


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "clip",
    factor: float = 1.5,
) -> pd.DataFrame:
    """Handle outliers using the IQR rule.

    method="clip": cap values at the IQR fences (keeps every row).
    method="drop": remove rows outside the fences (fences are computed once
                   on the original data, so column order doesn't matter).
    NaN values are ignored when computing quartiles and are left untouched.
    """
    if method not in {"clip", "drop"}:
        raise ValueError("method must be 'clip' or 'drop'")

    df = df.copy()
    fences = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        fences[col] = (q1 - factor * iqr, q3 + factor * iqr)

    if method == "clip":
        for col, (low, high) in fences.items():
            n_out = ((df[col] < low) | (df[col] > high)).sum()
            logger.info("Clipping %d outliers in '%s'", n_out, col)
            df[col] = df[col].clip(lower=low, upper=high)
        return df

    keep = pd.Series(True, index=df.index)
    for col, (low, high) in fences.items():
        keep &= df[col].isna() | df[col].between(low, high)
    logger.info("Dropping %d outlier rows", (~keep).sum())
    return df[keep]


def clean(df: pd.DataFrame, outlier_method: str = "clip") -> pd.DataFrame:
    """Return a cleaned copy of the raw heart disease dataframe."""
    df = df.copy()  # never modify the caller's dataframe

    _check_columns(
        df,
        MODE_COLUMNS + MEDIAN_COLUMNS + OUTLIER_COLUMNS + BOOL_COLUMNS,
    )

    # 1. Drop columns with too many missing values
    existing = [c for c in DROP_COLUMNS if c in df.columns]
    if existing:
        pct = (df[existing].isna().mean() * 100).round(1).to_dict()
        logger.info("Dropping columns (missing %%): %s", pct)
        df = df.drop(columns=existing)

    # 2. Invalid zeros -> NaN (before imputing, so they don't skew statistics)
    for col in ZERO_IS_MISSING:
        n_zero = (df[col] == 0).sum()
        if n_zero:
            logger.info("Replacing %d zero values in '%s' with NaN", n_zero, col)
        df[col] = df[col].replace(0, np.nan)

    # 3. Outliers (before imputing so imputed values don't affect the fences)
    df = handle_outliers(df, OUTLIER_COLUMNS, method=outlier_method)

    # 4. Impute missing values
    for col in MODE_COLUMNS:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    # 5. Fix dtypes (after imputation, so there are no NaNs left)
    for col in INT_COLUMNS:
        df[col] = df[col].round().astype(int)
    for col in BOOL_COLUMNS:
        df[col] = df[col].astype(bool)

    remaining = int(df.isna().sum().sum())
    logger.info("Cleaning done. Shape: %s | remaining NaNs: %d", df.shape, remaining)
    return df.reset_index(drop=True)


def save(df: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save the cleaned dataframe to CSV and return the file path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if not output_path.exists():
        raise IOError(f"Failed to save dataset to {output_path}")

    logger.info("Saved to %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


def load_and_clean_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    outlier_method: str = "clip",
) -> pd.DataFrame:
    """Load raw data, clean it, save it, and return the cleaned dataframe."""
    df = load_data(input_path)
    if df is None:
        raise RuntimeError(f"Could not load data from {input_path}")

    clean_df = clean(df, outlier_method=outlier_method)
    save(clean_df, output_path)
    return clean_df


if __name__ == "__main__":
    load_and_clean_data()
