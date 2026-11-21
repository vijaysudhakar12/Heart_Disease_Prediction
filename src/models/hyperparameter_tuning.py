"""Hyperparameter tuning.

Location in project: src/models/hyperparameter_tuning.py

Tunes the whole pipeline (SMOTE + classifier) with stratified cross-validation
on the TRAINING data only, so SMOTE is applied inside every fold. The notebook
tuned on already-oversampled data and then peeked at the test set to compare
searches; here the test set is never used for tuning.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Keys are prefixed with "classifier__" because the estimator is the
# "classifier" step of the pipeline built in train_models.py.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "decision_tree": {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [2, 5, 10],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [4, 6, 8, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [True, False],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 5],
        "classifier__min_samples_leaf": [3, 5, 10],
        "classifier__subsample": [0.7, 0.8, 1.0],
    },
    "xgboost": {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__reg_lambda": [1.0, 5.0, 10.0],
    },
}


def tune_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search: str = "random",
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = 1,
):
    """Run a hyperparameter search and return the fitted search object.

    The search refits the best configuration on all of X_train, so
    `search_obj.best_estimator_` is ready to use.
    """
    if model_name not in PARAM_GRIDS:
        raise KeyError(
            f"No parameter grid for '{model_name}'. Available: {list(PARAM_GRIDS)}"
        )
    if search not in {"random", "grid"}:
        raise ValueError("search must be 'random' or 'grid'")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_grid = PARAM_GRIDS[model_name]

    if search == "random":
        searcher = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )
    else:
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )

    logger.info("Tuning %s with %s search (scoring=%s) ...", model_name, search, scoring)
    searcher.fit(X_train, y_train)
    logger.info("Best CV %s: %.4f", scoring, searcher.best_score_)
    logger.info("Best params: %s", searcher.best_params_)
    return searcher
