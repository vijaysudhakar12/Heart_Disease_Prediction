"""Shared pytest fixtures.

Location in project: tests/conftest.py

The tests use small synthetic data in the same format as cleaned_heart.csv,
so they never depend on your real dataset or on trained artifacts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_cleaned_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic data shaped like data/processed/cleaned_heart.csv."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    oldpeak = np.round(rng.random(n) * 4, 1)
    exang = rng.choice([True, False], n)

    # Make the label depend on a few features so a model can learn something.
    risk = (age - 55) / 20 + oldpeak / 2 + exang * 1.0 + rng.normal(0, 1, n)
    num = np.where(risk > 0.6, rng.integers(1, 5, n), 0)
    num[:2] = [0, 1]  # guarantee both classes exist

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": age,
            "sex": rng.choice(["Male", "Female"], n),
            "dataset": rng.choice(
                ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"], n
            ),
            "cp": rng.choice(
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"], n
            ),
            "trestbps": rng.integers(90, 180, n),
            "chol": rng.integers(120, 400, n),
            "fbs": rng.choice([True, False], n),
            "restecg": rng.choice(
                ["normal", "lv hypertrophy", "st-t abnormality"], n
            ),
            "thalch": rng.integers(70, 200, n),
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": rng.choice(["upsloping", "flat", "downsloping"], n),
            "num": num,
        }
    )


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    return make_cleaned_df()


@pytest.fixture
def sample_patient() -> dict:
    """One valid patient record, as the app would send it."""
    return {
        "age": 62,
        "sex": "Male",
        "dataset": "Cleveland",
        "cp": "asymptomatic",
        "trestbps": 150,
        "chol": 260,
        "fbs": False,
        "restecg": "normal",
        "thalch": 120,
        "exang": True,
        "oldpeak": 3.1,
        "slope": "flat",
    }
