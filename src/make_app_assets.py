"""Save honest model scores for the Streamlit app."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "data"
OUT.mkdir(parents=True, exist_ok=True)

model = joblib.load(ROOT / "models" / "churn_model.pkl")["model"]
df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
y = df["churn_flag"]
X = df.drop(columns=["customer_id", "churn", "churn_flag"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
pd.DataFrame({"y": y_train.values, "score": oof}).to_csv(OUT / "train_oof_scores.csv", index=False)
pd.DataFrame({"y": y_test.values, "score": model.predict_proba(X_test)[:, 1]}).to_csv(
    OUT / "test_scores.csv", index=False)
print("Saved train_oof_scores.csv:", len(X_train), "rows and test_scores.csv:", len(X_test), "rows")