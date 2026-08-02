# =============================================
# Import Libraries
# =============================================
import joblib
import mlflow
import mlflow.sklearn
import optuna

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from evaluation_script import evaluate_model


# =============================================
# Load Processed Dataset
# =============================================
data = joblib.load(r'..\processed_data\dataset_bundle.pkl')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# =============================================
# MLflow Configuration (Safe Storage Folder)
# =============================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRACKING_DIR = BASE_DIR / "mlflow_tracking"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

db_path = TRACKING_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

mlflow.set_experiment("Bank Churn Prediction")


# =============================================
# Optuna Objective Function
# =============================================
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
    }
    
    model = DecisionTreeClassifier(**params, random_state=200)
    score = cross_val_score(model, X_train, y_train, scoring="f1", cv=5, n_jobs=-1).mean()
    return score


# =============================================
# MLflow Run
# =============================================
with mlflow.start_run(run_name="Decision Tree"):

    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Train Final Best Model
    best_params = study.best_params
    best_model = DecisionTreeClassifier(**best_params, random_state=200)
    best_model.fit(X_train, y_train)

    # Evaluation
    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # Log Parameters
    mlflow.log_params(
        best_params
    )

    # Log CV Score
    mlflow.log_metric(
        "best_cv_score",
        study.best_value
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
        MODEL_DIR / "decision_tree.pkl"
    )

print("=" * 60)
print("Training Completed Successfully (Decision Tree)")
print("=" * 60)
print("\nBest Parameters:\n", study.best_params)
print("\nBest CV Score:\n", study.best_value)
print("\nTest Metrics:\n", metrics)