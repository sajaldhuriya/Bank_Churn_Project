# libraries and packages
from pathlib import Path
import pandas as pd
import pyodbc

# load data
BASE_DIR = Path(__file__).resolve().parents[2]
FILE_NAME = "account.csv"
DATA_PATH = BASE_DIR / "data" / "processed" / FILE_NAME
df = pd.read_csv(DATA_PATH)

# create conenction
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=LAPTOP-ANP0GB7A\TEW_SQLEXPRESS;"
    "Database=BankChurn;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()


# Push demographic to database
cursor.execute("SET IDENTITY_INSERT demographic ON")
conn.commit()

for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO demographic (
            CustomerId,
            Gender,
            Age,
            Salary,
            LocationId,
            Churned
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    int(row.CustomerId),
    row.Gender,
    int(row.Age),
    float(row.Salary),
    int(row.LocationId),
    int(row.Churned)
    )

conn.commit()
print("Data inserted successfully")




# Push location to database
cursor.execute("SET IDENTITY_INSERT location ON")
conn.commit()

for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO location (
            LocationId,
            Geography
        )
        VALUES (?, ?)
    """,
    int(row.LocationId),
    row.Geography
    )

conn.commit()
print("Data inserted successfully")




# Push Account to database
cursor.execute("SET IDENTITY_INSERT account ON")
conn.commit()

for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO account (
            CustomerId,
            Tenure, 
            Balance,     
            NumProducts,
            HasCreditCard,
            IsActive
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    int(row.CustomerId),
    int(row.Tenure),
    float(row.Balance),
    int(row.NumProducts),
    int(row.HasCreditCard),
    int(row.IsActive),
    )

conn.commit()
print("Data inserted successfully")