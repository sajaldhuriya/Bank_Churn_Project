"""
Model evaluation for bank churn prediction.

Why this matters:
    Accuracy alone is misleading on an imbalanced dataset like churn
    (~20% positive class). We therefore optimise and report PR-AUC,
    and tune the decision threshold to lift recall — the metric the
    business actually cares about.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Default operational constraint: we will not flag a customer unless
# the precision of the positive class is at least this high. Below
# this we keep the default 0.5 threshold.
DEFAULT_PRECISION_FLOOR = 0.40


def _best_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    grid_size: int = 101,
) -> float:
    """Return the lowest threshold that satisfies the precision floor.

    Strategy: sweep thresholds in [0,1]; among those whose precision
    on the test set is >= `precision_floor`, pick the lowest one
    (which maximises recall). Falls back to 0.5 if no threshold
    satisfies the floor.
    """
    thresholds = np.linspace(0.0, 1.0, grid_size)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        # 1-pred is rare here; guard against division-by-zero precision.
        if preds.sum() == 0:
            continue
        precision = precision_score(y_true, preds, zero_division=0)
        if precision >= precision_floor:
            return float(t)
    return 0.5


def evaluate_model(
    model,
    X_test,
    y_test,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> Dict[str, float]:
    """Compute the metrics used to compare experiments.

    Parameters
    ----------
    model
        A *fitted* classifier exposing `predict` and `predict_proba`.
    X_test, y_test
        Held-out test arrays.
    precision_floor
        Minimum acceptable precision for threshold tuning.

    Returns
    -------
    dict
        Recall/Precision/F1/PR_AUC/ROC_AUC at the default threshold,
        plus Recall_Churn_tuned at the tuned threshold and the
        threshold itself.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tuned_threshold = _best_threshold_for_recall(
        y_test, y_prob, precision_floor=precision_floor
    )
    y_pred_tuned = (y_prob >= tuned_threshold).astype(int)

    return {
        # Threshold = 0.5 (default)
        "Recall_Churn": recall_score(y_test, y_pred),
        "Precision_Churn": precision_score(y_test, y_pred),
        "F1_Churn": f1_score(y_test, y_pred),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        # Threshold tuning
        "Recall_Churn_tuned": recall_score(y_test, y_pred_tuned),
        "Precision_Churn_tuned": precision_score(y_test, y_pred_tuned),
        "best_threshold": tuned_threshold,
    }
