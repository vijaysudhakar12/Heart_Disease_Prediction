"""Model definitions, cross-validated comparison and training.

Location in project: src/models/train_models.py

Models come from 06_Model_Experiments.ipynb (Decision Tree, Random Forest,
Gradient Boosting, XGBoost) with the notebook's hyperparameters.

Every model is wrapped in a pipeline whose last step is named "classifier":

    [SMOTE (optional)] -> classifier

Putting SMOTE inside the pipeline means oversampling happens separately inside
each cross-validation fold and only on the training part. The notebook
oversampled BEFORE cross-validation, so duplicated/synthetic copies of the same
patient landed in both the training and validation folds and inflated scores.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return the candidate models (unfitted)."""
    models: dict[str, BaseEstimator] = {
        # Lightly regularised: a default, unlimited tree just memorises the data.
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            ccp_alpha=0.005,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=5,
            min_samples_split=10,
            subsample=0.8,
            max_features="sqrt",
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost is not installed; skipping XGBoost.")

    return models


def build_model_pipeline(
    estimator: BaseEstimator,
    use_smote: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Wrap an estimator as [SMOTE ->] classifier. The estimator is cloned."""
    classifier = clone(estimator)

    if not use_smote:
        return Pipeline([("classifier", classifier)])

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise ImportError(
            "use_smote=True needs imbalanced-learn: pip install imbalanced-learn"
        ) from exc

    # Note: SMOTE interpolates between rows, so one-hot / binary columns can
    # get fractional values. If that bothers you, use SMOTENC or class_weight.
    return ImbPipeline(
        [
            (
                "oversampler",
                SMOTE(
                    sampling_strategy="not majority",
                    k_neighbors=3,
                    random_state=random_state,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def compare_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool = False,
    cv_folds: int = 5,
    sort_by: str = "f1",
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-validate every model on the training data and rank them.

    Returns one row per model with mean/std of each metric in SCORING_METRICS.
    """
    if sort_by not in SCORING_METRICS:
        raise ValueError(f"sort_by must be one of {SCORING_METRICS}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows = []

    for name, estimator in models.items():
        logger.info("Cross-validating %s ...", name)
        pipeline = build_model_pipeline(estimator, use_smote, random_state)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=SCORING_METRICS,
            n_jobs=1,  # the models already use all cores themselves
            error_score="raise",
        )
        row = {"model": name}
        for metric in SCORING_METRICS:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(f"{sort_by}_mean", ascending=False)
        .reset_index(drop=True)
    )


def train_model(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Fit one model (inside its pipeline) on the full training set."""
    pipeline = build_model_pipeline(estimator, use_smote, random_state)
    pipeline.fit(X_train, y_train)
    return pipeline
