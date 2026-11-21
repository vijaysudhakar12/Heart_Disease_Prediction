"""Shared paths, config loading and small helpers.

Location in project: src/utils/common.py
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_heart.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_REPORT_PATH = REPORTS_DIR / "model_report.txt"

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Defaults are used when config.yaml is missing or only contains some keys.
DEFAULT_CONFIG: dict[str, Any] = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "scoring": "f1",           # metric used to pick/tune the best model
    "scale_numeric": True,     # StandardScaler on numeric columns
    "use_smote": False,        # oversample the minority class inside each CV fold
    "tuning": {
        "search": "random",    # "random" or "grid"
        "n_iter": 20,          # only used for random search
    },
    "prediction_threshold": 0.5,
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml on top of DEFAULT_CONFIG (the file is optional)."""
    path = Path(path)
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        import yaml
    except ImportError:
        logging.getLogger(__name__).warning(
            "PyYAML not installed; ignoring %s and using default config.", path
        )
        return copy.deepcopy(DEFAULT_CONFIG)

    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user_config)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
