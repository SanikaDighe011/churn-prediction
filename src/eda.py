"""Phase 3: EDA on the cleaned churn data."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
overall = df["churn_flag"].mean()
print(f"Customers: {len(df):,} | overall churn rate: {overall:.1%}")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("   saved", name)


# ---- 1. churn rate by each category column ----
cat_cols = ["gender", "senior_citizen", "partner", "dependents", "phone_service",
            "multiple_lines", "internet_service", "online_security", "online_backup",
            "device_protection", "tech_support", "streaming_tv", "streaming_movies",
            "contract", "paperless_billing", "payment_method"]
rows = []
for c in cat_cols:
    g = df.groupby(c)["churn_flag"].agg(customers="count", churn_rate="mean")
    for value, r in g.iterrows():
        rows.append({"column": c, "value": value, "customers": int(r["customers"]),
                     "churn_rate_%": round(r["churn_rate"] * 100, 1)})
tbl = pd.DataFrame(rows)
print("\n=== CHURN RATE BY CATEGORY (overall is "
      f"{overall:.1%}) ===")
print(tbl.to_string(index=False))

# spread of churn rate inside each column = rough usefulness
spread = (tbl.groupby("column")["churn_rate_%"].agg(lambda s: s.max() - s.min())
             .sort_values(ascending=False))
print("\nGap between highest and lowest churn rate inside each column:")
print(spread.round(1).to_string())

# ---- 2. tenure bands ----
df["tenure_band"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                           labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
t = df.groupby("tenure_band", observed=True)["churn_flag"].agg(customers="count",
                                                              churn_rate="mean")
t["churn_rate"] = (t["churn_rate"] * 100).round(1)
print("\n=== CHURN RATE BY TENURE (months) ===")
print(t.to_string())

# ---- 3. monthly charge bands ----
df["charge_band"] = pd.qcut(df["monthly_charges"], 4,
                            labels=["lowest 25%", "25-50%", "50-75%", "highest 25%"])
m = df.groupby("charge_band", observed=True)["churn_flag"].agg(customers="count",
                                                              churn_rate="mean")
m["churn_rate"] = (m["churn_rate"] * 100).round(1)
print("\n=== CHURN RATE BY MONTHLY CHARGES ===")
print(m.to_string())

# ---- 4. numeric columns vs churn ----
print("\n=== CORRELATION WITH churn_flag (-1 to +1) ===")
print(df[["tenure", "monthly_charges", "total_charges", "senior_citizen",
          "churn_flag"]].corr()["churn_flag"].drop("churn_flag").round(3).to_string())
print("\nAverage by outcome:")
print(df.groupby("churn")[["tenure", "monthly_charges", "total_charges"]].mean().round(1))

# ---- 5. charts ----
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="churn", data=df, ax=ax, color="steelblue")
ax.set(title="Customers who stayed (No) vs left (Yes)")
save("01_class_balance.png")

for col, name in [("contract", "02_churn_by_contract.png"),
                  ("internet_service", "03_churn_by_internet.png"),
                  ("payment_method", "04_churn_by_payment.png")]:
    r = df.groupby(col)["churn_flag"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=r.values, y=r.index, ax=ax, color="steelblue")
    ax.axvline(overall * 100, color="red", linestyle="--", label="overall rate")
    ax.set(title=f"Churn rate (%) by {col}", xlabel="Churn rate (%)", ylabel="")
    ax.legend()
    save(name)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=t.index.astype(str), y=t["churn_rate"], ax=ax, color="darkorange")
ax.set(title="Churn rate (%) by tenure band (months)", xlabel="Tenure", ylabel="%")
save("05_churn_by_tenure.png")

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df, x="monthly_charges", hue="churn", bins=30, ax=ax)
ax.set(title="Monthly charges: stayed vs left")
save("06_monthly_charges.png")
print("\nDone. Charts are in reports/figures/")