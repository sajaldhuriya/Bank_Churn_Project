from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from _common import load_dataset, save_model_locally
from evaluation_script import evaluate_model

# ---------------------------------------------------------------------
# Load pre-processed data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset()

# ---------------------------------------------------------------------
# Model + search space
# ---------------------------------------------------------------------
model = SVC(probability=True, cache_size=500, random_state=200)

param_distributions = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", 0.01, 0.1],
    "class_weight": [None, "balanced"],
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=200,
)

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
search.fit(X_train, y_train)

best_model = search.best_estimator_
metrics = evaluate_model(best_model, X_test, y_test)
save_model_locally(best_model, "svm.pkl")

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("=" * 60)
print("Training Completed Successfully (SVM)")
print("=" * 60)
print("\nBest Parameters:\n", search.best_params_)
print("\nBest CV Score:\n", search.best_score_)
print("\nTest Metrics:\n", metrics)
