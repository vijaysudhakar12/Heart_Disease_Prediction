"""Tests for src/features/feature_engineering.py and preprocessing.py.

Location in project: tests/test_preprocessing.py
Run from the project root:  pytest tests/test_preprocessing.py -v
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    TARGET_COL,
    engineer_features,
    split_features_target,
)
from src.features.preprocessing import (
    FEATURE_COLS,
    NUMERIC_COLS,
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform,
)


# --------------------------------------------------------------------------
# feature_engineering
# --------------------------------------------------------------------------
def test_target_is_binary_version_of_num(cleaned_df):
    out = engineer_features(cleaned_df)

    assert set(out[TARGET_COL].unique()) <= {0, 1}
    assert (out[TARGET_COL] == (cleaned_df["num"] > 0).astype(int)).all()


def test_id_and_num_columns_are_removed(cleaned_df):
    out = engineer_features(cleaned_df)

    assert "id" not in out.columns
    assert "num" not in out.columns


def test_engineer_features_does_not_modify_input(cleaned_df):
    before = cleaned_df.copy()
    engineer_features(cleaned_df)

    pd.testing.assert_frame_equal(cleaned_df, before)


def test_engineer_features_without_label_for_prediction(cleaned_df):
    out = engineer_features(cleaned_df.drop(columns="num"))

    assert TARGET_COL not in out.columns
    assert "id" not in out.columns


def test_already_binary_num_is_unchanged(cleaned_df):
    cleaned_df["num"] = (cleaned_df["num"] > 0).astype(int)
    out = engineer_features(cleaned_df)

    assert (out[TARGET_COL] == cleaned_df["num"]).all()


def test_split_features_target(cleaned_df):
    X, y = split_features_target(engineer_features(cleaned_df))

    assert TARGET_COL not in X.columns
    assert len(X) == len(y) == len(cleaned_df)
    assert set(FEATURE_COLS) <= set(X.columns)


def test_split_features_target_requires_target(cleaned_df):
    with pytest.raises(KeyError):
        split_features_target(cleaned_df.drop(columns="num"))


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
@pytest.fixture
def X(cleaned_df):
    features, _ = split_features_target(engineer_features(cleaned_df))
    return features


def test_output_has_no_nans_and_keeps_all_rows(X):
    pre = fit_preprocessor(X)
    out = transform(pre, X)

    assert len(out) == len(X)
    assert out.isna().sum().sum() == 0


def test_output_is_a_dataframe_with_expected_columns(X):
    out = transform(fit_preprocessor(X), X)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns[: len(NUMERIC_COLS)]) == NUMERIC_COLS
    for col in ["sex", "fbs", "exang", "slope"]:
        assert col in out.columns
    assert any(c.startswith("cp_") for c in out.columns)
    assert any(c.startswith("restecg_") for c in out.columns)
    assert any(c.startswith("dataset_") for c in out.columns)


def test_ordinal_columns_follow_the_stated_order(X):
    pre = fit_preprocessor(X)
    rows = X.iloc[:3].copy()
    rows["sex"] = ["Female", "Male", "Female"]
    rows["fbs"] = [False, True, False]
    rows["exang"] = [True, False, True]
    rows["slope"] = ["downsloping", "flat", "upsloping"]

    out = transform(pre, rows)

    assert out["sex"].tolist() == [0, 1, 0]
    assert out["fbs"].tolist() == [0, 1, 0]
    assert out["exang"].tolist() == [1, 0, 1]
    assert out["slope"].tolist() == [0, 1, 2]


def test_numeric_columns_are_standardised_on_training_data(X):
    out = transform(fit_preprocessor(X, scale_numeric=True), X)

    for col in NUMERIC_COLS:
        assert out[col].mean() == pytest.approx(0, abs=1e-8)
        assert out[col].std(ddof=0) == pytest.approx(1, abs=1e-8)


def test_scaling_can_be_switched_off(X):
    out = transform(fit_preprocessor(X, scale_numeric=False), X)

    for col in NUMERIC_COLS:
        np.testing.assert_allclose(out[col].to_numpy(), X[col].to_numpy())


def test_scaler_uses_training_statistics_only(X):
    """New data must be scaled with the TRAIN mean/std (no refitting)."""
    train, new = X.iloc[:150], X.iloc[150:]
    pre = fit_preprocessor(train)
    out = transform(pre, new)

    expected = (new["age"] - train["age"].mean()) / train["age"].std(ddof=0)
    np.testing.assert_allclose(out["age"].to_numpy(), expected.to_numpy())


def test_unseen_nominal_category_becomes_all_zeros(X):
    pre = fit_preprocessor(X)
    row = X.iloc[[0]].copy()
    row["cp"] = "never seen before"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = transform(pre, row)

    cp_cols = [c for c in out.columns if c.startswith("cp_")]
    assert out[cp_cols].to_numpy().sum() == 0


def test_unseen_ordinal_category_becomes_minus_one(X):
    pre = fit_preprocessor(X)
    row = X.iloc[[0]].copy()
    row["sex"] = "Other"

    assert transform(pre, row)["sex"].iloc[0] == -1


def test_missing_required_column_raises(X):
    pre = fit_preprocessor(X)

    with pytest.raises(KeyError):
        transform(pre, X.drop(columns="chol"))


def test_extra_columns_are_ignored(X):
    pre = fit_preprocessor(X)
    baseline = transform(pre, X)

    noisy = X.copy()
    noisy["something_else"] = "ignored"
    noisy["id"] = range(len(noisy))

    pd.testing.assert_frame_equal(transform(pre, noisy), baseline)


def test_row_order_and_index_are_preserved(X):
    shuffled = X.sample(frac=1, random_state=1)
    out = transform(fit_preprocessor(X), shuffled)

    assert out.index.equals(shuffled.index)


def test_save_and_load_roundtrip(X, tmp_path):
    pre = fit_preprocessor(X)
    path = save_preprocessor(pre, tmp_path / "artifacts" / "preprocessor.joblib")

    assert path.exists()
    reloaded = load_preprocessor(path)
    pd.testing.assert_frame_equal(transform(reloaded, X), transform(pre, X))


def test_loading_missing_preprocessor_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preprocessor(tmp_path / "nope.joblib")


def test_unfitted_preprocessor_can_be_built():
    assert build_preprocessor() is not None
