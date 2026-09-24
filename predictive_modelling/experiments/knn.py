from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from _common import load_dataset, save_model_locally
from evaluation_script import evaluate_model

# ---------------------------------------------------------------------
# Load pre-processed data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset()

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
# Train
# ---------------------------------------------------------------------
search.fit(X_train, y_train)

best_model = search.best_estimator_
metrics = evaluate_model(best_model, X_test, y_test)
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
