"""
XGBoost experiment for bank churn prediction.

Pipeline: load dataset -> Optuna hyperparameter search -> evaluate
-> log to MLflow.
"""

import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from _common import load_dataset, save_model_locally, setup_mlflow
from evaluation_script import evaluate_model
from scoring import f1_scorer

# ---------------------------------------------------------------------
# Load pre-processed data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset()

# ---------------------------------------------------------------------
# Configure MLflow
# ---------------------------------------------------------------------
EXPERIMENT_ID = setup_mlflow()

# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Return the cross-validated F1 for the given XGBoost config."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, random_state=200, eval_metric="logloss")
    return cross_val_score(
        model, X_train, y_train, scoring=f1_scorer(), cv=5, n_jobs=-1
    ).mean()

# ---------------------------------------------------------------------
# Train + track
# ---------------------------------------------------------------------
with mlflow.start_run(run_name="XGBoost"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_model = xgb.XGBClassifier(
        **study.best_params, random_state=200, eval_metric="logloss"
    )
    best_model.fit(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_score", study.best_value)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.xgboost.log_model(
        xgb_model=best_model.get_booster(), artifact_path="model"
    )
    save_model_locally(best_model, "xgboost.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (XGBoost)")
print("=" * 60)
print("\nBest Parameters:\n", study.best_params)
print("\nBest CV Score:\n", study.best_value)
print("\nTest Metrics:\n", metrics)
