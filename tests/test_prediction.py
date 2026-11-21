"""Tests for src/pipelines/prediction_pipeline.py.

Location in project: tests/test_prediction.py
Run from the project root:  pytest tests/test_prediction.py -v

A tiny model is trained on synthetic data inside a temp folder, so these tests
do not need your real data or the files in artifacts/.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.features.feature_engineering import engineer_features, split_features_target
from src.features.preprocessing import (
    FEATURE_COLS,
    fit_preprocessor,
    save_preprocessor,
    transform,
)
from src.pipelines.prediction_pipeline import PredictionPipeline


def _train_artifacts(cleaned_df, tmp_path, model):
    X, y = split_features_target(engineer_features(cleaned_df))
    pre = fit_preprocessor(X)
    model.fit(transform(pre, X), y)

    pre_path = save_preprocessor(pre, tmp_path / "preprocessor.joblib")
    model_path = tmp_path / "best_model.joblib"
    joblib.dump(model, model_path)
    return pre_path, model_path


@pytest.fixture
def paths(cleaned_df, tmp_path):
    model = RandomForestClassifier(n_estimators=25, random_state=0)
    return _train_artifacts(cleaned_df, tmp_path, model)


@pytest.fixture
def pipe(paths):
    pre_path, model_path = paths
    return PredictionPipeline(pre_path, model_path, threshold=0.5)


def test_predict_one_returns_expected_structure(pipe, sample_patient):
    result = pipe.predict_one(sample_patient)

    assert set(result) == {"prediction", "probability", "label"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in ("Heart disease", "No heart disease")


def test_label_matches_prediction(pipe, sample_patient):
    result = pipe.predict_one(sample_patient)
    expected = "Heart disease" if result["prediction"] == 1 else "No heart disease"

    assert result["label"] == expected


def test_predict_dataframe_has_one_row_per_patient(pipe, cleaned_df):
    out = pipe.predict(cleaned_df.head(7))

    assert len(out) == 7
    assert list(out.columns) == ["prediction", "probability", "label"]
    assert set(out["prediction"].unique()) <= {0, 1}
    assert out["probability"].between(0, 1).all()


def test_predict_accepts_raw_cleaned_rows_with_id_and_num(pipe, cleaned_df):
    """cleaned_heart.csv rows (with id and num) must work without extra steps."""
    out = pipe.predict(cleaned_df.head(3))

    assert len(out) == 3


def test_predict_does_not_modify_input_dataframe(pipe, cleaned_df):
    rows = cleaned_df.head(5)
    before = rows.copy()
    pipe.predict(rows)

    pd.testing.assert_frame_equal(rows, before)


def test_prediction_is_deterministic(pipe, sample_patient):
    assert pipe.predict_one(sample_patient) == pipe.predict_one(sample_patient)


@pytest.mark.parametrize(
    "true_value, false_value", [("Yes", "No"), ("true", "false"), (1, 0), ("y", "n")]
)
def test_yes_no_inputs_equal_booleans(pipe, sample_patient, true_value, false_value):
    as_bool = {**sample_patient, "fbs": True, "exang": False}
    as_other = {**sample_patient, "fbs": true_value, "exang": false_value}

    assert pipe.predict_one(as_other) == pipe.predict_one(as_bool)


def test_invalid_boolean_raises_value_error(pipe, sample_patient):
    with pytest.raises(ValueError):
        pipe.predict_one({**sample_patient, "fbs": "maybe"})


def test_missing_feature_raises_key_error(pipe, sample_patient):
    incomplete = {k: v for k, v in sample_patient.items() if k != "age"}

    with pytest.raises(KeyError):
        pipe.predict_one(incomplete)


def test_threshold_controls_the_decision(paths, sample_patient):
    pre_path, model_path = paths

    always_positive = PredictionPipeline(pre_path, model_path, threshold=0.0)
    always_negative = PredictionPipeline(pre_path, model_path, threshold=1.01)

    assert always_positive.predict_one(sample_patient)["prediction"] == 1
    assert always_negative.predict_one(sample_patient)["prediction"] == 0


def test_model_without_predict_proba_still_predicts(
    cleaned_df, tmp_path, sample_patient
):
    pre_path, model_path = _train_artifacts(
        cleaned_df, tmp_path, LinearSVC(random_state=0)
    )
    pipe = PredictionPipeline(pre_path, model_path)
    result = pipe.predict_one(sample_patient)

    assert result["prediction"] in (0, 1)
    assert np.isnan(result["probability"])


def test_missing_model_file_raises(paths, tmp_path):
    pre_path, _ = paths

    with pytest.raises(FileNotFoundError):
        PredictionPipeline(pre_path, tmp_path / "missing_model.joblib")


def test_missing_preprocessor_file_raises(paths, tmp_path):
    _, model_path = paths

    with pytest.raises(FileNotFoundError):
        PredictionPipeline(tmp_path / "missing_preprocessor.joblib", model_path)


def test_required_columns_match_preprocessing():
    assert PredictionPipeline.required_columns() == list(FEATURE_COLS)
