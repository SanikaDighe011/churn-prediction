"""Phase 7: explain the churn model (coefficients + SHAP)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(ROOT / "models" / "churn_model.pkl")
model = bundle["model"]
prep, lr = model.named_steps["prep"], model.named_steps["m"]

df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
y = df["churn_flag"]
X = df.drop(columns=["customer_id", "churn", "churn_flag"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
coef = pd.DataFrame({"feature": names, "coef": lr.coef_[0]})
coef["odds_ratio"] = np.exp(coef["coef"])
coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
pd.options.display.float_format = "{:.3f}".format
print("TOP 12 FEATURES BY STRENGTH")
print(coef.head(12).to_string(index=False))

top = coef.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top["feature"], top["coef"],
        color=["#c0392b" if c > 0 else "#2e86c1" for c in top["coef"]])
ax.set(title="Strongest model weights (red = raises churn risk, blue = lowers)")
plt.tight_layout()
plt.savefig(FIG / "07_coefficients.png", dpi=150)
plt.close()

# SHAP on the transformed test data
Xt = prep.transform(X_test)
Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
Xb = prep.transform(X_train)
Xb = Xb.toarray() if hasattr(Xb, "toarray") else Xb
explainer = shap.LinearExplainer(lr, Xb)
sv = explainer.shap_values(Xt)
shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
plt.tight_layout()
plt.savefig(FIG / "08_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved 07_coefficients.png and 08_shap_summary.png")