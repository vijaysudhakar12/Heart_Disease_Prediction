"""End-to-end training pipeline.

Location in project: src/pipelines/training_pipeline.py

Run from the project root:
    python -m src.pipelines.training_pipeline            # compare models
    python -m src.pipelines.training_pipeline --tune     # + tune the winner
    python -m src.pipelines.training_pipeline --smote    # oversample in CV folds

Steps
    1. load cleaned data  ->  engineer features (target, drop id)
    2. stratified train/test split
    3. fit preprocessor on TRAIN only  ->  save artifacts/preprocessor.joblib
    4. cross-validate all models on TRAIN  ->  artifacts/model_comparison.csv
    5. pick the best model by CV score (the test set plays no part in this)
    6. optionally tune it (kept only if it beats the untuned CV score)
    7. evaluate ONCE on the test set  ->  report + confusion matrix
    8. save artifacts/best_model.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import load_data  # noqa: E402
from src.features.feature_engineering import (  # noqa: E402
    engineer_features,
    split_features_target,
)
from src.features.preprocessing import (  # noqa: E402
    FEATURE_COLS,
    fit_preprocessor,
    save_preprocessor,
    transform,
)
from src.models.hyperparameter_tuning import tune_model  # noqa: E402
from src.models.model_evaluation import (  # noqa: E402
    evaluate_model,
    plot_confusion_matrix,
    save_model_comparison,
    write_model_report,
)
from src.models.train_models import (  # noqa: E402
    SCORING_METRICS,
    compare_models,
    get_models,
    train_model,
)
from src.utils.common import (  # noqa: E402
    BEST_MODEL_PATH,
    CLEANED_DATA_PATH,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_REPORT_PATH,
    PREPROCESSOR_PATH,
    ensure_dir,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _load_cleaned_data():
    """Load cleaned data, running the cleaning step first if it is missing."""
    if not CLEANED_DATA_PATH.exists():
        logger.info("%s not found - running data cleaning first.", CLEANED_DATA_PATH)
        from src.data.data_cleaning import load_and_clean_data

        return load_and_clean_data()

    df = load_data(CLEANED_DATA_PATH)
    if df is None:
        raise RuntimeError(f"Could not read {CLEANED_DATA_PATH}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{CLEANED_DATA_PATH.name} is missing columns {missing}. It probably "
            "contains already-encoded data (the old 05_Feature_Engineering "
            "notebook overwrote it). Re-create it with: "
            "python src/data/data_cleaning.py"
        )
    return df


def run_training(
    tune: bool = False,
    use_smote: bool | None = None,
    config: dict | None = None,
) -> dict:
    """Run the full training pipeline and return a small summary dict."""
    cfg = config or load_config()
    if use_smote is not None:
        cfg["use_smote"] = use_smote

    seed = cfg["random_state"]
    scoring = cfg["scoring"]
    if scoring not in SCORING_METRICS:
        raise ValueError(f"config 'scoring' must be one of {SCORING_METRICS}")
    set_seed(seed)

    # 1. Data + features
    df = engineer_features(_load_cleaned_data())
    X, y = split_features_target(df)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=seed,
        stratify=y,
    )
    logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)

    # 3. Preprocessor (fitted on train only)
    preprocessor = fit_preprocessor(X_train, scale_numeric=cfg["scale_numeric"])
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)
    X_train_t = transform(preprocessor, X_train)
    X_test_t = transform(preprocessor, X_test)

    # 4. Compare models with cross-validation
    models = get_models(seed)
    comparison = compare_models(
        models,
        X_train_t,
        y_train,
        use_smote=cfg["use_smote"],
        cv_folds=cfg["cv_folds"],
        sort_by=scoring,
        random_state=seed,
    )
    save_model_comparison(comparison, MODEL_COMPARISON_PATH)
    logger.info("\n%s", comparison.to_string(index=False))

    # 5. Best model by CV score
    best_name = comparison.iloc[0]["model"]
    baseline_cv_score = comparison.iloc[0][f"{scoring}_mean"]
    logger.info("Best model by CV %s: %s (%.4f)", scoring, best_name, baseline_cv_score)

    best_model = train_model(
        models[best_name], X_train_t, y_train, cfg["use_smote"], seed
    )
    best_params = None

    # 6. Optional tuning
    if tune:
        search = tune_model(
            best_name,
            best_model,
            X_train_t,
            y_train,
            search=cfg["tuning"]["search"],
            n_iter=cfg["tuning"]["n_iter"],
            cv_folds=cfg["cv_folds"],
            scoring=scoring,
            random_state=seed,
        )
        if search.best_score_ > baseline_cv_score:
            logger.info(
                "Tuning improved CV %s: %.4f -> %.4f. Using tuned model.",
                scoring,
                baseline_cv_score,
                search.best_score_,
            )
            best_model = search.best_estimator_
            best_params = {
                k.replace("classifier__", ""): v
                for k, v in search.best_params_.items()
            }
        else:
            logger.info("Tuning did not beat the default settings; keeping them.")

    # 7. Single final evaluation on the held-out test set
    test_metrics = evaluate_model(best_model, X_test_t, y_test)
    logger.info("Test metrics: %s", test_metrics)

    ensure_dir(FIGURES_DIR)
    plot_confusion_matrix(
        best_model,
        X_test_t,
        y_test,
        FIGURES_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_name}",
    )
    write_model_report(
        MODEL_REPORT_PATH,
        best_name,
        best_model,
        X_test_t,
        y_test,
        comparison,
        best_params=best_params,
        config=cfg,
    )

    # 8. Save the model
    ensure_dir(BEST_MODEL_PATH.parent)
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info("Saved best model to %s", BEST_MODEL_PATH)

    return {
        "best_model": best_name,
        "cv_comparison": comparison,
        "test_metrics": test_metrics,
        "best_params": best_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart disease models.")
    parser.add_argument("--tune", action="store_true", help="tune the best model")
    parser.add_argument(
        "--smote", action="store_true", help="use SMOTE inside the CV folds"
    )
    args = parser.parse_args()

    setup_logging()
    run_training(tune=args.tune, use_smote=True if args.smote else None)


if __name__ == "__main__":
    main()
