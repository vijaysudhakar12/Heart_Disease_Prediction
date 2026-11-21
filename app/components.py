"""Reusable Streamlit UI pieces for the Heart Disease app.

Location in project: app/components.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.preprocessing import ONEHOT_COLS, ORDINAL_COLS  # noqa: E402
from src.utils.common import FIGURES_DIR, MODEL_COMPARISON_PATH, MODEL_REPORT_PATH  # noqa: E402

DISCLAIMER = (
    "This tool is a machine-learning project for learning purposes. "
    "It is **not** a medical device and does not give medical advice. "
    "Please talk to a doctor about any health concerns."
)

LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "dataset": "Data source (hospital)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar above 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalch": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of the peak exercise ST segment",
}

HELP = {
    "dataset": "Which hospital's data the patient record resembles.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "exang": "Chest pain triggered by exercise.",
    "thalch": "Highest heart rate reached during the exercise test.",
}

# Number inputs: (min, max, default, step). Ints and floats must not be mixed
# within one entry, otherwise Streamlit raises an error.
NUMERIC_INPUTS = {
    "age": dict(min_value=18, max_value=100, value=54, step=1),
    "trestbps": dict(min_value=80, max_value=220, value=130, step=1),
    "chol": dict(min_value=100, max_value=600, value=240, step=1),
    "thalch": dict(min_value=60, max_value=220, value=150, step=1),
    "oldpeak": dict(min_value=0.0, max_value=7.0, value=1.0, step=0.1),
}

LEFT_FIELDS = ["age", "sex", "dataset", "cp", "trestbps", "chol"]
RIGHT_FIELDS = ["fbs", "restecg", "thalch", "exang", "oldpeak", "slope"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_category_options(preprocessor) -> dict[str, list]:
    """Valid values of every categorical column, read from the fitted preprocessor.

    The dropdowns therefore always match what the model was trained on,
    with no hard-coded lists to keep in sync.
    """
    options: dict[str, list] = {}

    onehot = preprocessor.named_transformers_["onehot"]
    for col, categories in zip(ONEHOT_COLS, onehot.categories_):
        options[col] = list(categories.tolist())

    ordinal = preprocessor.named_transformers_["ordinal"]
    for col, categories in zip(ORDINAL_COLS, ordinal.categories_):
        options[col] = list(categories.tolist())

    return options


def _pretty(value) -> str:
    """Display text for a category (the underlying value is left untouched)."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = str(value)
    return text[:1].upper() + text[1:]


def _render_field(container, column: str, options: dict[str, list]):
    label = LABELS[column]
    help_text = HELP.get(column)

    if column in NUMERIC_INPUTS:
        return container.number_input(
            label, help=help_text, **NUMERIC_INPUTS[column]
        )

    return container.selectbox(
        label,
        options[column],
        format_func=_pretty,
        help=help_text,
    )


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("About")
        st.write(
            "Enter a patient's details and the trained model estimates "
            "whether they are likely to have heart disease."
        )
        st.caption(
            "Trained on the UCI Heart Disease dataset. "
            "Run `python -m src.pipelines.training_pipeline` to retrain."
        )
        st.warning(DISCLAIMER)


def render_patient_inputs(options: dict[str, list]) -> dict:
    """Draw all input widgets in two columns and return the patient record."""
    left, right = st.columns(2)
    record: dict = {}

    for column in LEFT_FIELDS:
        record[column] = _render_field(left, column, options)
    for column in RIGHT_FIELDS:
        record[column] = _render_field(right, column, options)

    return record


def render_result(result: dict, threshold: float) -> None:
    """Show the prediction, probability and a short explanation."""
    probability = result["probability"]
    has_probability = not math.isnan(probability)

    if result["prediction"] == 1:
        st.error(f"Model prediction: **{result['label']}**")
    else:
        st.success(f"Model prediction: **{result['label']}**")

    if has_probability:
        st.metric("Estimated probability of heart disease", f"{probability:.0%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"A prediction of 'Heart disease' is made when the probability is "
            f"{threshold:.0%} or higher."
        )
    else:
        st.caption("This model does not provide probabilities.")

    st.caption(DISCLAIMER)


def render_input_summary(record: dict) -> None:
    """Show the submitted values in a readable table."""
    summary = pd.DataFrame(
        {
            "Field": [LABELS[k] for k in record],
            "Value": [_pretty(v) if isinstance(v, (bool, str)) else v for v in record.values()],
        }
    )
    with st.expander("Values used for this prediction"):
        st.dataframe(summary, hide_index=True)


def render_model_performance() -> None:
    """Show the saved training results (comparison table, confusion matrix, report)."""
    if not MODEL_COMPARISON_PATH.exists():
        st.info("No training results yet. Run the training pipeline first.")
        return

    st.subheader("Model comparison (cross-validation on training data)")
    st.dataframe(pd.read_csv(MODEL_COMPARISON_PATH), hide_index=True)

    confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        st.subheader("Confusion matrix (test set)")
        st.image(str(confusion_matrix_path))

    if MODEL_REPORT_PATH.exists():
        with st.expander("Full model report"):
            st.code(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
