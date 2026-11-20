"""Phase 2: clean the Telco churn data. Raw file is only READ."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)


df = pd.read_csv(RAW)
n_raw = len(df)

# 1. consistent column names
df = df.rename(columns={
    "customerID": "customer_id", "SeniorCitizen": "senior_citizen",
    "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
    "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup", "DeviceProtection": "device_protection",
    "TechSupport": "tech_support", "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies", "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method", "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
})
df.columns = [c.lower() for c in df.columns]

# 2. strip spaces from every text column
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].str.strip()

# 3. total_charges: text -> number (bad values become NaN so we can SEE them)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
bad = df[df["total_charges"].isna()]
print(f"total_charges that could not be converted: {len(bad)}")
if len(bad):
    print(bad[["customer_id", "tenure", "monthly_charges", "total_charges", "churn"]]
          .to_string(index=False))

# 4. numeric target
df["churn_flag"] = (df["churn"] == "Yes").astype(int)

# 5. look at the category columns (find redundant values)
print("\nValues in each category column:")
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("customer_id", "churn"):
        print(f"  {c}: {sorted(df[c].unique())}")

# 6. numeric summary
print("\nNumeric summary:")
print(df[["tenure", "monthly_charges", "total_charges"]].describe().round(2))

# 7. validation
print("\n=== VALIDATION ===")
check("row count unchanged (7043)", len(df) == n_raw == 7043)
check("customer_id unique", df["customer_id"].is_unique)
check("churn has only Yes/No", set(df["churn"].unique()) == {"Yes", "No"})
check("churn_flag matches churn", (df["churn_flag"] == (df["churn"] == "Yes")).all())
check("tenure not negative", (df["tenure"] >= 0).all())
check("monthly_charges positive", (df["monthly_charges"] > 0).all())
print(f"Churn rate: {df['churn_flag'].mean():.1%}")

df.to_csv(OUT / "churn_clean.csv", index=False)
print("\nSaved data/processed/churn_clean.csv")