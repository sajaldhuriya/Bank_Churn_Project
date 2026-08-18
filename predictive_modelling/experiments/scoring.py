"""
Project-wide scorers.

Why this file exists:
    Scikit-learn's `make_scorer` argument `needs_proba` was removed
    in sklearn 1.4+. As of sklearn 1.9 (your install) it raises
    TypeError, which crashes every Optuna trial. To stay portable
    we build custom scorers that call `predict_proba` themselves,
    bypassing the `make_scorer` mechanism for proba-based metrics.

    All scorers return *larger = better* (Optuna / CV convention).
"""

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Standard scorers (predicted-label based)
# ---------------------------------------------------------------------
def f1_scorer():
    """F1 on the positive class. *larger = better*.

    sklearn calls label-based scorers as `(estimator, X, y_true)` and
    computes `y_pred` itself. We mirror that pattern.
    """

    def _score(estimator, X, y_true, **kwargs):
        y_pred = estimator.predict(X)
        return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return _score


def recall_scorer():
    """Recall on the positive class. *larger = better*."""

    def _score(estimator, X, y_true, **kwargs):
        y_pred = estimator.predict(X)
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    return _score


# ---------------------------------------------------------------------
# Proba-based scorers (custom-wrapped so we don't need needs_proba)
# ---------------------------------------------------------------------
def pr_auc_scorer():
    """Average precision = PR-AUC. *larger = better*.

    sklearn will call this with `(estimator, X, y_true)`.
    """

    def _score(estimator, X, y_true, **kwargs):
        proba = estimator.predict_proba(X)[:, 1]
        return average_precision_score(y_true, proba)

    return _score


def roc_auc_scorer():
    """ROC-AUC. *larger = better*."""

    def _score(estimator, X, y_true, **kwargs):
        proba = estimator.predict_proba(X)[:, 1]
        return roc_auc_score(y_true, proba)

    return _score
