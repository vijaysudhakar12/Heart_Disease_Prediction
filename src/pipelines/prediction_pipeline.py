"""Prediction pipeline for new patients (used by the Streamlit app).

Location in project: src/pipelines/prediction_pipeline.py

    from src.pipelines.prediction_pipeline import PredictionPipeline

    pipe = PredictionPipeline()
    pipe.predict_one({
        "age": 54, "sex": "Male", "dataset": "Cleveland", "cp": "asymptomatic",
        "trestbps": 140, "chol": 239, "fbs": False, "restecg": "normal",
        "thalch": 160, "exang": False, "oldpeak": 1.2, "slope": "flat",
    })

Input uses the same format as cleaned_heart.csv (before encoding). The saved
preprocessor does all encoding/scaling, so training and prediction can never
drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineering import engineer_features  # noqa: E402
from src.features.preprocessing import (  # noqa: E402
    FEATURE_COLS,
    check_input_columns,
    load_preprocessor,
    transform,
)
from src.utils.common import (  # noqa: E402
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    load_config,
)

BOOL_COLS = ["fbs", "exang"]
_TRUE = {"true", "1", "1.0", "yes", "y", "t"}
_FALSE = {"false", "0", "0.0", "no", "n", "f"}


def _to_bool(value) -> bool:
    """Accept True/False, 1/0 and 'yes'/'no'-style strings (e.g. from a web form)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


class PredictionPipeline:
    """Loads the saved preprocessor and model once and predicts on new data."""

    def __init__(
        self,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        model_path: Path = BEST_MODEL_PATH,
        threshold: float | None = None,
    ) -> None:
        self.preprocessor = load_preprocessor(preprocessor_path)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run the training pipeline first."
            )
        self.model = joblib.load(model_path)

        self.threshold = (
            threshold
            if threshold is not None
            else load_config()["prediction_threshold"]
        )

    def _prepare(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = df[col].map(_to_bool)

        df = engineer_features(df)  # drops id; no-op for target when absent
        check_input_columns(df)
        return transform(self.preprocessor, df)

    def predict(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """Return one row per patient: prediction (0/1), probability and label."""
        X = self._prepare(data)

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(X)[:, 1]
            prediction = (probability >= self.threshold).astype(int)
        else:
            prediction = self.model.predict(X)
            probability = np.full(len(X), np.nan)

        return pd.DataFrame(
            {
                "prediction": prediction,
                "probability": np.round(probability, 4),
                "label": np.where(prediction == 1, "Heart disease", "No heart disease"),
            }
        )

    def predict_one(self, record: dict) -> dict:
        """Convenience wrapper for a single patient (returns a plain dict)."""
        row = self.predict(record).iloc[0]
        return {
            "prediction": int(row["prediction"]),
            "probability": float(row["probability"]),
            "label": str(row["label"]),
        }

    @staticmethod
    def required_columns() -> list[str]:
        return list(FEATURE_COLS)
