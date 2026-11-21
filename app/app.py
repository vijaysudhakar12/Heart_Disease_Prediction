"""Streamlit app for heart disease prediction.

Location in project: app/app.py
Run from the project root:

    streamlit run app/app.py

Train first so the artifacts exist:

    python -m src.pipelines.training_pipeline
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402

# `components` (not `app.components`): this file is called app.py, so a package
# named "app" would clash with it.
from components import (  # noqa: E402
    get_category_options,
    render_input_summary,
    render_model_performance,
    render_patient_inputs,
    render_result,
    render_sidebar,
)
from src.pipelines.prediction_pipeline import PredictionPipeline  # noqa: E402

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline() -> PredictionPipeline:
    """Load the preprocessor and model once and reuse them across reruns."""
    return PredictionPipeline()


def main() -> None:
    st.title("❤️ Heart Disease Prediction")
    render_sidebar()

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python -m src.pipelines.training_pipeline", language="bash")
        return  # st.stop() alone does nothing when not run via `streamlit run`

    options = get_category_options(pipeline.preprocessor)

    predict_tab, performance_tab = st.tabs(["Predict", "Model performance"])

    with predict_tab:
        with st.form("patient_form"):
            record = render_patient_inputs(options)
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            try:
                result = pipeline.predict_one(record)
            except (KeyError, ValueError) as error:
                st.error(f"Could not make a prediction: {error}")
            else:
                render_result(result, pipeline.threshold)
                render_input_summary(record)

    with performance_tab:
        render_model_performance()


main()
