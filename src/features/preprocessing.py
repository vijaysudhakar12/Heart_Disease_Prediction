"""Preprocessing (encoding + scaling) for the Heart Disease UCI dataset.

Location in project: src/features/preprocessing.py

Replaces the encoding done by hand in 05_Feature_Engineering.ipynb.
Everything is wrapped in ONE scikit-learn ColumnTransformer, so it can be:
  * fitted on the training split only (no data leakage),
  * saved to artifacts/preprocessor.joblib,
  * reused unchanged in the prediction pipeline and the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.joblib"

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------
NUMERIC_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Nominal categories (no natural order) -> one-hot
ONEHOT_COLS = ["restecg", "cp", "dataset"]

# Binary / ordered categories -> ordinal, with the order stated explicitly
# (the notebook relied on OrdinalEncoder's alphabetical default).
ORDINAL_CATEGORIES = {
    "sex": ["Female", "Male"],
    "fbs": [False, True],
    "exang": [False, True],
    "slope": ["downsloping", "flat", "upsloping"],
}
ORDINAL_COLS = list(ORDINAL_CATEGORIES)

FEATURE_COLS = NUMERIC_COLS + ONEHOT_COLS + ORDINAL_COLS


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build an (unfitted) preprocessor.

    scale_numeric: standardise numeric columns. Helps logistic regression,
    SVM and KNN; tree-based models don't need it but are not harmed by it.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"

    onehot = OneHotEncoder(
        drop="if_binary",           # binary columns -> 1 column, others -> all levels
        handle_unknown="ignore",    # unseen category at prediction time -> all zeros
        sparse_output=False,
    )

    ordinal = OrdinalEncoder(
        categories=[ORDINAL_CATEGORIES[c] for c in ORDINAL_COLS],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUMERIC_COLS),
            ("onehot", onehot, ONEHOT_COLS),
            ("ordinal", ordinal, ORDINAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")  # keep column names
    return preprocessor


def build_preprocessing_pipeline(scale_numeric: bool = True) -> Pipeline:
    """Same as build_preprocessor but wrapped in a Pipeline (handy for CV)."""
    return Pipeline([("preprocessor", build_preprocessor(scale_numeric))])


def check_input_columns(df: pd.DataFrame) -> None:
    """Fail early with a clear message if required columns are missing."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


def fit_preprocessor(
    X_train: pd.DataFrame, scale_numeric: bool = True
) -> ColumnTransformer:
    """Fit a new preprocessor on TRAINING data only."""
    check_input_columns(X_train)
    preprocessor = build_preprocessor(scale_numeric)
    preprocessor.fit(X_train[FEATURE_COLS])
    return preprocessor


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor; returns a dataframe with named columns."""
    check_input_columns(X)
    return preprocessor.transform(X[FEATURE_COLS])


def save_preprocessor(
    preprocessor: ColumnTransformer, path: Path = DEFAULT_PREPROCESSOR_PATH
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: Path = DEFAULT_PREPROCESSOR_PATH) -> ColumnTransformer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run the training pipeline first."
        )
    return joblib.load(path)
