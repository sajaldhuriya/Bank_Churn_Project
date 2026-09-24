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
# Load (CSV)
# ---------------------------------------------------------------------
def _load_from_csv() -> pd.DataFrame:
    """Join the three processed CSVs in memory (no SQL dependency)."""
    account = pd.read_csv(DATA_PROCESSED / "account.csv")
    demographic = pd.read_csv(DATA_PROCESSED / "demographic.csv")
    location = pd.read_csv(DATA_PROCESSED / "location.csv")

    df = demographic.merge(account, on="CustomerId", how="inner")
    df = df.merge(location, on="LocationId", how="inner")
    return df


def load_dataframe() -> pd.DataFrame:
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
