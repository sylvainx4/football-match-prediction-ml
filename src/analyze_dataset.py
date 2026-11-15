
import pandas as pd
from pathlib import Path

# Locate and load our dataset
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "raw"
csv_file = list(data_folder.glob("*.csv"))[0]

df = pd.read_csv(csv_file, encoding="cp1252")

print("\n========== DATASET ANALYSIS ==========")

# 1. Dataset size
print("\n1. DATASET SIZE")
print("Total matches:", len(df))
print("Total columns:", len(df.columns))

# 2. Date range
df["DateTime"] = pd.to_datetime(
    df["DateTime"], errors="coerce", utc=True
)

print("\n2. MATCH DATE RANGE")
print("First match:", df["DateTime"].min())
print("Last match:", df["DateTime"].max())

# 3. Match outcome distribution
print("\n3. MATCH OUTCOME DISTRIBUTION")
print(df["FTR"].value_counts())
print("\nOutcome percentages:")
print((df["FTR"].value_counts(normalize=True) * 100).round(2))

# 4. Missing values
print("\n4. MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# 5. Duplicate matches
print("\n5. DUPLICATE ROWS")
print("Duplicates:", df.duplicated().sum())

# 6. Number of unique teams
print("\n6. TEAM INFORMATION")
print("Home teams:", df["HomeTeam"].nunique())
print("Away teams:", df["AwayTeam"].nunique())

# 7. Data types
print("\n7. DATA TYPES")
print(df.dtypes)

print("\n========== ANALYSIS COMPLETE ==========")
