# =============================================
# Import Libraries
# =============================================
import joblib
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb

from pathlib import Path
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
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
    }
    
    model = xgb.XGBClassifier(**params, random_state=200, eval_metric="logloss")
    score = cross_val_score(model, X_train, y_train, scoring="f1", cv=5, n_jobs=-1).mean()
    return score


# =============================================
# MLflow Run
# =============================================
with mlflow.start_run(run_name="XGBoost"):

    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    # Train Final Best Model
    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, random_state=200, eval_metric="logloss")
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
    mlflow.xgboost.log_model(
        xgb_model=best_model,
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
        MODEL_DIR / "xgboost.pkl"
    )

print("=" * 60)
print("Training Completed Successfully (XGBoost)")
print("=" * 60)
print("\nBest Parameters:\n", study.best_params)
print("\nBest CV Score:\n", study.best_value)
print("\nTest Metrics:\n", metrics)