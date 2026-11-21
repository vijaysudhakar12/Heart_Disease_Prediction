"""Model evaluation, plots and report writing.

Location in project: src/models/model_evaluation.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.common import ensure_dir  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return the main classification metrics for a fitted model."""
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        metrics["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])

    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    return metrics


def plot_confusion_matrix(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a confusion matrix figure to `path`."""
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y,
        model.predict(X),
        display_labels=["No disease", "Disease"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)
    return path


def save_model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    comparison.to_csv(path, index=False)
    logger.info("Saved model comparison to %s", path)
    return path


def write_model_report(
    path: Path,
    best_model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    comparison: pd.DataFrame,
    best_params: dict | None = None,
    config: dict | None = None,
) -> Path:
    """Write a plain-text summary of the run to `path`."""
    path = Path(path)
    ensure_dir(path.parent)

    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)

    lines = [
        "HEART DISEASE PREDICTION - MODEL REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"Best model (chosen by cross-validation): {best_model_name}",
        "",
        "Cross-validation comparison (training data)",
        "-" * 60,
        comparison.to_string(index=False),
        "",
    ]
    if best_params:
        lines += ["Tuned hyperparameters", "-" * 60]
        lines += [f"{k}: {v}" for k, v in best_params.items()]
        lines.append("")

    lines += [
        "Final test-set evaluation (used once, after model selection)",
        "-" * 60,
        *[f"{k}: {v}" for k, v in metrics.items() if k != "confusion_matrix"],
        "",
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        str(metrics["confusion_matrix"]),
        "",
        "Classification report:",
        classification_report(
            y_test, y_pred, target_names=["No disease", "Disease"], zero_division=0
        ),
    ]
    if config:
        lines += ["Config used", "-" * 60]
        lines += [f"{k}: {v}" for k, v in config.items()]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved model report to %s", path)
    return path
