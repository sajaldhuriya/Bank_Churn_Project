# =============================================
# Import Libraries
# =============================================
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from evaluation_script import evaluate_model


# =============================================
# Load Processed Dataset
# =============================================
data = joblib.load('..\processed_data\dataset_bundle.pkl')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# =============================================
# MLflow Configuration (Safe Storage Folder)
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Create a dedicated tracking directory to match your project structure
TRACKING_DIR = BASE_DIR / "mlflow_tracking"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Define explicit absolute path for the SQLite database
db_path = TRACKING_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

mlflow.set_experiment("Bank Churn Prediction")


# =============================================
# Model
# =============================================
model = KNeighborsClassifier()


# =============================================
# Hyperparameter Grid
# =============================================
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "p": [1, 2] # 1 = Manhattan, 2 = Euclidean
}


# =============================================
# Grid Search
# =============================================
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)


# =============================================
# MLflow Run
# =============================================
with mlflow.start_run(run_name="K-Nearest Neighbors"):

    # Train
    grid_search.fit(X_train, y_train)

    # Best Model
    best_model = grid_search.best_estimator_

    # Evaluation
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # Log Parameters
    mlflow.log_params(
        grid_search.best_params_
    )

    # Log CV Score
    mlflow.log_metric(
        "best_cv_score",
        grid_search.best_score_
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

    # Save Model Locally
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "knn.pkl"
    )

print("=" * 60)
print("Training Completed Successfully (KNN)")
print("=" * 60)
print("\nBest Parameters:\n", grid_search.best_params_)
print("\nBest CV Score:\n", grid_search.best_score_)
print("\nTest Metrics:\n", metrics)