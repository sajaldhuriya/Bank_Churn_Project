"""
XGBoost v2 — recall-optimized experiment.

Strategy:
    1. Use scale_pos_weight = N_neg / N_pos so XGBoost treats the
       minority (churn) class with appropriate weight.
    2. Optuna searches against PR-AUC (not F1) — PR-AUC is the
       correct threshold-free metric under imbalance.
    3. Report both default-threshold recall and threshold-tuned recall
       via `evaluate_model`.

This is the script reviewers should run when asked "did you do
anything about the recall problem?"
"""

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from _common import load_dataset, save_model_locally, setup_mlflow
from evaluation_script import evaluate_model
from scoring import pr_auc_scorer

# ---------------------------------------------------------------------
# Load pre-processed data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset()

# ---------------------------------------------------------------------
# Class weights for imbalance
# ---------------------------------------------------------------------
N_NEG = float(np.sum(y_train == 0))
N_POS = float(np.sum(y_train == 1))
SCALE_POS_WEIGHT = N_NEG / N_POS  # tells XGBoost to up-weight churners

# ---------------------------------------------------------------------
# Configure MLflow
# ---------------------------------------------------------------------
EXPERIMENT_ID = setup_mlflow()

# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Return averaged PR-AUC across stratified CV folds."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        # recall booster
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    model = xgb.XGBClassifier(
        **params,
        random_state=200,
        eval_metric="logloss",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)
    return cross_val_score(
        model, X_train, y_train, scoring=pr_auc_scorer(), cv=cv, n_jobs=-1
    ).mean()

# ---------------------------------------------------------------------
# Train + track
# ---------------------------------------------------------------------
with mlflow.start_run(run_name="XGBoost (Recall-Optimized)"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = {**study.best_params, "scale_pos_weight": SCALE_POS_WEIGHT}
    best_model = xgb.XGBClassifier(
        **best_params, random_state=200, eval_metric="logloss"
    )
    best_model.fit(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test)

    mlflow.log_params(best_params)
    mlflow.log_param("scale_pos_weight", SCALE_POS_WEIGHT)
    mlflow.log_metric("best_cv_score", study.best_value)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.xgboost.log_model(
        xgb_model=best_model.get_booster(), artifact_path="model"
    )
    save_model_locally(best_model, "xgboost_balanced.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (XGBoost — Recall-Optimized)")
print("=" * 60)
print(f"\nscale_pos_weight: {SCALE_POS_WEIGHT:.3f}")
print(f"\nBest Parameters:\n {study.best_params}")
print(f"\nBest CV PR-AUC:\n {study.best_value:.4f}")
print("\nTest Metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
