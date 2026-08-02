# =============================================
# Import Libraries
# =============================================
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

from evaluation_script import evaluate_model


# =============================================
# Load Processed Dataset
# =============================================
# Using a raw string to fix the invalid escape sequence warning
data = joblib.load(r'..\processed_data\dataset_bundle.pkl')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# =============================================
# MLflow Configuration
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRACKING_DIR = BASE_DIR / "mlflow_tracking"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

db_path = TRACKING_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

mlflow.set_experiment("Bank Churn Prediction")


# =============================================
# Model
# =============================================
# Using cache_size to speed up kernel computations
model = SVC(
    probability=True, 
    cache_size=500,
    random_state=200
)


# =============================================
# Hyperparameter Distributions (Streamlined for speed)
# =============================================
param_distributions = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", 0.01, 0.1],
    "class_weight": [None, "balanced"]
}


# =============================================
# Randomized Search
# =============================================
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10, # Reduced slightly to prevent long stalls
    scoring="f1",
    cv=3,      # Reduced to 3 folds for faster testing
    n_jobs=-1,
    verbose=2,
    random_state=200
)


# =============================================
# MLflow Run
# =============================================
with mlflow.start_run(run_name="Support Vector Machine"):

    # Train
    random_search.fit(X_train, y_train)

    # Best Model
    best_model = random_search.best_estimator_

    # Evaluation
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # Log Parameters
    mlflow.log_params(
        random_search.best_params_
    )

    # Log CV Score
    mlflow.log_metric(
        "best_cv_score",
        random_search.best_score_
    )

    # Log Test Metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(
            metric_name,
            metric_value
        )

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # Save Model
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "svm.pkl"
    )

print("=" * 60)
print("Training Completed Successfully (SVM)")
print("=" * 60)
print("\nBest Parameters:\n", random_search.best_params_)
print("\nBest CV Score:\n", random_search.best_score_)
print("\nTest Metrics:\n", metrics)