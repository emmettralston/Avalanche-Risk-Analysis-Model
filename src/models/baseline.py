"""Training, evaluation, and cross-validation for zone-level danger classifiers.

Implements the Phase 1 baseline models (XGBoost and LightGBM) for predicting
NWAC avalanche danger ratings (1–5).  All cross-validation uses season-aware
splits — never random row splits — to prevent temporal data leakage.

Key design decisions implemented here:
- Stratified k-fold is approximated by holding out full seasons, not rows.
- Class weights are computed from inverse class frequency to address the
  rarity of danger levels 4 and 5.
- Evaluation reports per-class F1 in addition to overall accuracy, since
  overall accuracy is misleading under class imbalance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DANGER_CLASS_NAMES = ["Low", "Moderate", "Considerable", "High", "Extreme"]


def compute_class_weights(y: pd.Series) -> dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced danger labels.

    Weight for class c = (total samples) / (n_classes * count(c)).
    These weights are passed directly to XGBoost / LightGBM to up-weight
    rare high-danger labels.

    Parameters
    ----------
    y:
        Series of integer danger ratings (1–5).

    Returns
    -------
    dict[int, float]
        Mapping from danger rating integer to sample weight float.
    """
    raise NotImplementedError


def get_season_cv_splits(
    df: pd.DataFrame,
    season_col: str = "season",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate season-aware cross-validation splits.

    Produces ``n_splits`` (train_idx, val_idx) pairs where each validation
    fold is one complete season.  Seasons are ordered chronologically so that
    each fold validates on a later season than its training data.

    This is equivalent to a time-series walk-forward split at season granularity,
    ensuring no temporal leakage.

    Parameters
    ----------
    df:
        DataFrame with a ``season_col`` column.  Must include all rows
        (including the held-out test seasons) so splits are computed on the
        full chronological sequence.
    season_col:
        Name of the column containing season strings (e.g. "2018-19").
    n_splits:
        Number of cross-validation folds (seasons to use as validation).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) integer arrays.
    """
    raise NotImplementedError


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict[int, float] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    """Train an XGBoost multi-class classifier.

    Parameters
    ----------
    X_train:
        Feature DataFrame (no NaNs — impute upstream).
    y_train:
        Integer danger ratings (1–5).  XGBoost requires 0-indexed classes;
        this function remaps 1–5 to 0–4 internally and stores the offset.
    class_weights:
        Per-class sample weights as returned by ``compute_class_weights``.
        If None, all classes are weighted equally.
    hyperparams:
        Keyword arguments forwarded to ``xgb.XGBClassifier``.  Defaults to a
        sensible starting configuration (n_estimators=500, max_depth=6,
        learning_rate=0.05, subsample=0.8).

    Returns
    -------
    xgb.XGBClassifier
        Fitted classifier.
    """
    raise NotImplementedError


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict[int, float] | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    """Train a LightGBM multi-class classifier.

    Parameters
    ----------
    X_train:
        Feature DataFrame.
    y_train:
        Integer danger ratings (1–5).
    class_weights:
        Per-class sample weights.  If None, uses LightGBM's built-in
        ``is_unbalance=True``.
    hyperparams:
        Keyword arguments forwarded to ``lgb.LGBMClassifier``.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted classifier.
    """
    raise NotImplementedError


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute evaluation metrics for a fitted classifier.

    Metrics computed:
    - Overall accuracy.
    - Per-class precision, recall, F1 (macro and weighted averages).
    - Confusion matrix (5×5 for danger levels 1–5).
    - Cohen's kappa (accounts for chance agreement, standard in avalanche
      forecasting literature).

    Parameters
    ----------
    model:
        Fitted XGBoost or LightGBM classifier.
    X_test:
        Feature DataFrame.
    y_test:
        True danger ratings.

    Returns
    -------
    dict[str, Any]
        Keys: "accuracy", "macro_f1", "weighted_f1", "kappa",
        "per_class_f1" (Series indexed by danger level), "confusion_matrix"
        (np.ndarray).
    """
    raise NotImplementedError


def cross_validate(
    model_fn: Callable[..., Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run season-aware cross-validation and collect per-fold metrics.

    Parameters
    ----------
    model_fn:
        A callable that accepts (X_train, y_train, **model_kwargs) and returns
        a fitted model.  Use ``train_xgboost`` or ``train_lightgbm``.
    X:
        Full feature DataFrame (train split only — do not include test seasons).
    y:
        Full target Series aligned with ``X``.
    cv_splits:
        List of (train_idx, val_idx) pairs from ``get_season_cv_splits``.
    model_kwargs:
        Extra keyword arguments forwarded to ``model_fn``.

    Returns
    -------
    pd.DataFrame
        One row per fold.  Columns: fold, accuracy, macro_f1, weighted_f1,
        kappa, plus per_class_f1_{1..5}.
    """
    raise NotImplementedError


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = DANGER_CLASS_NAMES,
    output_path: Path | None = None,
) -> None:
    """Plot a labelled confusion matrix using matplotlib.

    Parameters
    ----------
    cm:
        Square confusion matrix array (true label rows, predicted label cols).
    class_names:
        Labels for each class.  Defaults to ``DANGER_CLASS_NAMES``.
    output_path:
        If provided, saves the figure to this path instead of displaying it.
    """
    raise NotImplementedError


def save_model(model: Any, output_path: Path) -> None:
    """Serialize a fitted model to disk.

    Uses the model's native serialisation (XGBoost JSON, LightGBM txt) for
    portability.  Falls back to joblib for other model types.

    Parameters
    ----------
    model:
        Fitted classifier.
    output_path:
        File path to write.  Extension determines format: ``.json`` for
        XGBoost, ``.txt`` for LightGBM, ``.joblib`` otherwise.
    """
    raise NotImplementedError


def load_model(model_path: Path) -> Any:
    """Load a serialised model from disk.

    Parameters
    ----------
    model_path:
        Path previously written by ``save_model``.

    Returns
    -------
    Any
        Fitted classifier ready for inference.
    """
    raise NotImplementedError
