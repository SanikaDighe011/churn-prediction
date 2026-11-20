from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
files = list(RAW.glob("*.csv"))
print("CSV files found:", [f.name for f in files])
df = pd.read_csv(files[0])

print("\nRows, columns:", df.shape)
print("\nColumns, types, missing values:")
print(pd.DataFrame({"dtype": df.dtypes.astype(str),
                    "missing": df.isna().sum(),
                    "unique": df.nunique()}))
print("\nDuplicated rows:", df.duplicated().sum())
print("\nFirst 3 rows:")
print(df.head(3).T)

target = [c for c in df.columns if "churn" in c.lower()]
print("\nTarget column guess:", target)
if target:
    print(df[target[0]].value_counts())
    print((df[target[0]].value_counts(normalize=True) * 100).round(1))