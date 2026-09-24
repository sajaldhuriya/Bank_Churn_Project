import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from _common import load_dataset, save_model_locally
from evaluation_script import evaluate_model

# ---------------------------------------------------------------------
# Load pre-processed data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset()

# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Return the cross-validated F1 for the given Random Forest config."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }
    model = RandomForestClassifier(**params, random_state=200)
    return cross_val_score(
        model, X_train, y_train, scoring='f1', cv=5, n_jobs=-1
    ).mean()

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_model = RandomForestClassifier(**study.best_params, random_state=200)
best_model.fit(X_train, y_train)

metrics = evaluate_model(best_model, X_test, y_test)
save_model_locally(best_model, "random_forest.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (Random Forest)")
print("=" * 60)
print("\nBest Parameters:\n", study.best_params)
print("\nBest CV Score:\n", study.best_value)
print("\nTest Metrics:\n", metrics)
