# =============================================
# Import packages and libraries
# =============================================
import numpy as np
import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# =============================================
# Read data from SQL Server Database
# =============================================
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=LAPTOP-ANP0GB7A\TEW_SQLEXPRESS;"
    "Database=BankChurn;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

query = """
SELECT
    d.Gender, d.Age, d.Salary, l.[Geography] , 
    a.Tenure, a.Balance, a.NumProducts, a.HasCreditCard, a.IsActive,
    d.Churned
FROM demographic d
JOIN account a ON a.CustomerId = d.CustomerId
JOIN [location] l ON l.LocationId = d.LocationId
"""

df = pd.read_sql(query, conn)


# =====================================
# Features & Target
# =====================================

X = df.drop("Churned", axis=1)
y = df["Churned"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=200,
    stratify=y
)

# =====================================
# Column Lists
# =====================================
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "bool", "category"]
).columns.tolist()

# =============================================
# Feature Engineering in Training
# =============================================

# =====================================
# Pipelines
# =====================================

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# =====================================
# Transform Data
# =====================================

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# =============================================
# Save Data
# =============================================
artifacts = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "preprocessor": preprocessor
}

joblib.dump(artifacts, 'dataset_bundle.pkl')