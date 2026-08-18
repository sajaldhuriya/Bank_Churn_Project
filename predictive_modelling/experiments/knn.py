"""
K-Nearest Neighbours classifier for bank churn prediction.

Pipeline: load dataset -> GridSearch over n_neighbors / weights / p ->
evaluate -> log to MLflow.

Note on class imbalance:
    KNN has no native `class_weight` argument, so we lean on
    (1) PR-AUC as the CV objective (threshold-free, honest under imbalance)
    (2) the threshold tuner in `evaluation_script.py` to lift recall after fit.
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from _common import load_dataset, save_model_locally, setup_mlflow
from evaluation_script import evaluate_model
from scoring import pr_auc_scorer

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
model = KNeighborsClassifier()

param_grid = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],          # 1 = Manhattan, 2 = Euclidean
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=pr_auc_scorer(),
    cv=5,
    n_jobs=-1,
    verbose=2,
)

# ---------------------------------------------------------------------
# Train + track
# ---------------------------------------------------------------------
with mlflow.start_run(run_name="KNN"):
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_cv_score", search.best_score_)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")
    save_model_locally(best_model, "knn.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (KNN)")
print("=" * 60)
print("\nBest Parameters:\n", search.best_params_)
print("\nBest CV Score:\n", search.best_score_)
print("\nTest Metrics:\n", metrics)
