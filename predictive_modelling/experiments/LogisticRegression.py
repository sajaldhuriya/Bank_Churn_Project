# =============================================
# Import Libraries
# =============================================
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from evaluation_script import evaluate_model


# =============================================
# Load Processed Dataset
# =============================================
# Load Processed Data
data = joblib.load('..\processed_data\dataset_bundle.pkl')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# =============================================
# MLflow Configuration
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent

mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR/'mlflow.db'}"
)

mlflow.set_experiment("Bank Churn Prediction")


# =============================================
# Model
# =============================================
model = LogisticRegression(
    random_state=200
)


# =============================================
# Hyperparameter Grid
# =============================================
param_grid = {

    "C": [0.1, 0.5, 1, 10, 50, 100],

    "solver": [
        "lbfgs",
        "liblinear"
    ],

    "class_weight": [
        None,
        "balanced"
    ],

    "max_iter": [
        250,
        500,
        1000
    ],

    "tol": [
        1e-4,
        1e-5,
        1e-6
    ]

}


# =============================================
# Grid Search
# =============================================
grid_search = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    scoring="f1",

    cv=10,

    n_jobs=-1,

    verbose=2

)


# =============================================
# MLflow Run
# =============================================
with mlflow.start_run(run_name="Logistic Regression"):

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

    # Save Model
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "logistic_regression.pkl"
    )


print("=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print("\nBest Parameters")
print(grid_search.best_params_)

print("\nBest CV Score")
print(grid_search.best_score_)

print("\nTest Metrics")
print(metrics)