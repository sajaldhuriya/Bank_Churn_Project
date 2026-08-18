"""
Logistic Regression baseline for bank churn prediction.

Pipeline: load dataset -> GridSearch over C/solver/class_weight ->
evaluate -> log to MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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
# Model + search space
# ---------------------------------------------------------------------
model = LogisticRegression(random_state=200)

param_grid = {
    "C": [0.1, 0.5, 1, 10, 50, 100],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"],
    "max_iter": [250, 500, 1000],
    "tol": [1e-4, 1e-5, 1e-6],
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=f1_scorer(),
    cv=10,
    n_jobs=-1,
    verbose=2,
)

# ---------------------------------------------------------------------
# Train + track
# ---------------------------------------------------------------------
with mlflow.start_run(run_name="Logistic Regression"):
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_cv_score", search.best_score_)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")
    save_model_locally(best_model, "logistic_regression.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (Logistic Regression)")
print("=" * 60)
print("\nBest Parameters:\n", search.best_params_)
print("\nBest CV Score:\n", search.best_score_)
print("\nTest Metrics:\n", metrics)
