
import pandas as pd
from pathlib import Path

# Locate the project folder
project_folder = Path(__file__).resolve().parent.parent

# Locate the raw data folder
data_folder = project_folder / "data" / "raw"

# Find CSV files
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in data/raw!")
else:
    # Load the first CSV file
    file_path = csv_files[0]
    df = pd.read_csv(file_path, encoding="cp1252")

    print("\n========== FOOTBALL DATASET INSPECTION ==========")

    print("\nDataset filename:")
    print(file_path.name)

    print("\nFirst 5 matches:")
    print(df.head())

    print("\nDataset dimensions:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDataset information:")
    print(df.info())

    print("\n========== INSPECTION COMPLETE ==========")
