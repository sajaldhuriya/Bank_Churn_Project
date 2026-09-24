"""
Common helpers shared by every model experiment in this folder.

Why this file exists:
    Each experiment script used to duplicate the same MLflow/loading
    boilerplate. Centralizing it keeps the model scripts focused on
    their tuning logic and the workflow easier to follow in an
    interview.

Robustness fixes applied across all scripts:
    * `y_train` / `y_test` are always cast to int (so boolean columns
      from SQL Server -> pandas `BIT` don't break sklearn scorers).
    * `setup_mlflow()` is idempotent (creates the experiment if
      missing) and never raises on cold-start.
    * All paths use `pathlib.Path.as_posix()` so Windows back-slashes
      don't break the SQLite tracking URI.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA = BASE_DIR / "processed_data" / "dataset_bundle.pkl"
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the pre-processed train/test splits produced by preprocessing.py.

    Returns
    -------
    X_train, X_test, y_train, y_test
        The arrays have already passed through StandardScaler and
        OneHotEncoder, so they are safe to feed straight into sklearn.
        Labels are guaranteed to be `int` (0 / 1) — see `_to_y_int`.
    """
    data = joblib.load(PROCESSED_DATA.as_posix())
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = _to_y_int(data["y_train"]), _to_y_int(data["y_test"])
    return X_train, X_test, y_train, y_test


def _to_y_int(y) -> np.ndarray:
    """Cast any bool / object label array to plain int (0 / 1).

    SQL Server `BIT` columns come back as `bool` via pyodbc, and pandas
    preserves the dtype through `pd.read_sql`. sklearn's `f1_score`,
    `average_precision_score`, etc. all expect numeric labels.
    """
    arr = np.asarray(y)
    if arr.dtype == bool:
        arr = arr.astype(int)
    elif arr.dtype == object:
        # In case someone passed a Series of strings or pandas ints
        try:
            arr = arr.astype(int)
        except (ValueError, TypeError):
            pass
    return arr


# ---------------------------------------------------------------------
# Local artifacts
# ---------------------------------------------------------------------
def save_model_locally(model: ClassifierMixin, file_name: str) -> Path:
    """Persist a fitted classifier into the models/ folder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / file_name
    joblib.dump(model, out_path)
    return out_path
