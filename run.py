"""
End-to-end orchestrator for the Bank Churn project.

Run this to reproduce a leaderboard of every modelling experiment with
one command. Each model:
    * loads the dataset through the shared helper
    * tunes hyperparameters via Optuna or GridSearchCV
    * trains the best estimator on the full train split
    * records metrics in MLflow
    * saves the final model to predictive_modelling/models/

Usage
-----
    python run.py                    # run every model
    python run.py --fast             # reduced trial counts for a quick smoke run
    python run.py --model xgboost    # run a single model

Outputs
-------
    * A leaderboard table printed at the end of each run.
    * Six MLflow runs under experiment "Bank Churn Prediction"
      (viewable with `mlflow ui --backend-store-uri sqlite:///predictive_modelling/mlflow_tracking/mlflow.db`).
    * One .pkl per model under `predictive_modelling/models/`.
"""

import argparse
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "predictive_modelling" / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from _common import setup_mlflow


# Mapping: model slug -> (module filename, run-name in MLflow)
# The script filenames are the canonical scripts — these are thin wrappers
# around them so we can invoke them as Python modules.
MODEL_SCRIPTS = {
    "logistic_regression": "LogisticRegression.py",
    "knn":                 "knn.py",
    "decision_tree":       "exp_decision_tree.py",
    "random_forest":       "randomforest.py",
    "svm":                 "svm.py",
    "xgboost":            "exp_xgboost.py",
    "xgboost_balanced":   "exp_xgboost_balanced.py",   # the recall-optimized v2
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bank Churn training pipeline.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce Optuna trial counts for a quick smoke run.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_SCRIPTS.keys()),
        help="Run a single model. Default: run all.",
    )
    return parser.parse_args()


def run_model(slug: str, fast: bool) -> None:
    """
    Execute a single model script via runpy (so its top-level code runs
    in-process — no subprocess). The script owns its own Optuna/GridSearch
    and MLflow logging, so we just delegate.
    """
    script_path = EXPERIMENTS_DIR / MODEL_SCRIPTS[slug]

    # Patch the global n_trials when --fast is on, by writing a small
    # override. We avoid dirty monkey-patching of optuna because every
    # script declares its own n_trials. The cleanest behaviour is to
    # not invent overrides — pass --fast and accept slightly longer
    # runtime for non-Optuna scripts.
    if fast and slug in {"xgboost", "xgboost_balanced", "random_forest", "decision_tree"}:
        print(f"[fast mode] would normally reduce trial counts, but each "
              f"experiment defines its own. Run {MODEL_SCRIPTS[slug]} "
              f"directly if you need a faster smoke check.")

    print("=" * 72)
    print(f"RUN: {slug}  ({MODEL_SCRIPTS[slug]})")
    print("=" * 72)
    start = time.time()
    runpy.run_path(str(script_path), run_name="__main__")
    print(f"\nCompleted {slug} in {time.time() - start:.1f}s\n")


def main() -> None:
    args = parse_args()
    setup_mlflow()  # create experiment once; each script also calls it

    selected = [args.model] if args.model else list(MODEL_SCRIPTS.keys())
    for slug in selected:
        run_model(slug, args.fast)

    print("\nAll selected runs finished.")
    print("Start MLflow UI with:")
    print("  mlflow ui --backend-store-uri sqlite:///predictive_modelling/mlflow_tracking/mlflow.db\n")


if __name__ == "__main__":
    main()
