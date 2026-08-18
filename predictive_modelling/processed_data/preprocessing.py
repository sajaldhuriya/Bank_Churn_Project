"""
Bank Churn Preprocessing

Reads cleaned CSVs produced by scripts/data_cleaning/*.py, joins them,
splits, scales, one-hot-encodes, and persists a `dataset_bundle.pkl`
that the modelling scripts can load directly.

By default this module reads from the cleaned CSV files (the source of
truth used by both EDA and the statistical testing notebook). It can
optionally rebuild from SQL Server when BANKCHURN_USE_SQL=1 is set in
the environment and a local instance is available. This removes the
hard-coded server name dependency that earlier versions carried.

CSV inputs expected under data/processed/:
    account.csv      CustomerId, Tenure, Balance, NumProducts, HasCreditCard, IsActive
    demographic.csv  CustomerId, Gender, Age, Salary, LocationId, Churned
    location.csv     LocationId, Geography
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------
# Load (CSV or SQL)
# ---------------------------------------------------------------------
def _load_from_sql() -> pd.DataFrame:
    """Read the joined dataset from a local SQL Server (when configured)."""
    import pyodbc

    server = os.environ.get("BANKCHURN_SQL_SERVER", "localhost")
    database = os.environ.get("BANKCHURN_SQL_DB", "BankChurn")
    conn = pyodbc.connect(
        "Driver={SQL Server};"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=yes;"
    )
    query = """
        SELECT d.Gender, d.Age, d.Salary, l.Geography,
               a.Tenure, a.Balance, a.NumProducts, a.HasCreditCard, a.IsActive,
               d.Churned
        FROM demographic d
        JOIN account a   ON a.CustomerId  = d.CustomerId
        JOIN location  l ON l.LocationId  = d.LocationId
    """
    return pd.read_sql(query, conn)


def _load_from_csv() -> pd.DataFrame:
    """Join the three processed CSVs in memory (no SQL dependency)."""
    account = pd.read_csv(DATA_PROCESSED / "account.csv")
    demographic = pd.read_csv(DATA_PROCESSED / "demographic.csv")
    location = pd.read_csv(DATA_PROCESSED / "location.csv")

    df = demographic.merge(account, on="CustomerId", how="inner")
    df = df.merge(location, on="LocationId", how="inner")
    return df


def load_dataframe() -> pd.DataFrame:
    """Load the modelling dataframe from SQL if configured, else CSVs."""
    if os.environ.get("BANKCHURN_USE_SQL") == "1":
        return _load_from_sql()
    return _load_from_csv()


# ---------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Identify column types and return a fitted ColumnTransformer."""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def main() -> None:
    df = load_dataframe()
    # SQL Server `BIT` columns come back as bool; cast for sklearn.
    if df["Churned"].dtype == bool:
        df["Churned"] = df["Churned"].astype(int)
    for col in ["HasCreditCard", "IsActive"]:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)

    X = df.drop("Churned", axis=1)
    y = df["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=200, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    bundle_path = Path(__file__).resolve().parent / "dataset_bundle.pkl"
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.astype(int),
            "y_test": y_test.astype(int),
            "preprocessor": preprocessor,
        },
        bundle_path,
    )
    print(f"Wrote {bundle_path}")


if __name__ == "__main__":
    main()
