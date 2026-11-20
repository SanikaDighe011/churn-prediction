"""Phase 6: choose a threshold from ASSUMED business costs; score test set once."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
y = df["churn_flag"]
X = df.drop(columns=["customer_id", "churn", "churn_flag"])
num_cols = ["tenure", "monthly_charges", "total_charges"]
cat_cols = [c for c in X.columns if c not in num_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ASSUMPTIONS (not from the dataset)
OFFER_COST, CUSTOMER_VALUE, SUCCESS = 10.0, 200.0, 0.30
gain_tp = SUCCESS * CUSTOMER_VALUE - OFFER_COST      # net gain per flagged leaver
loss_fp = OFFER_COST                                  # cost per flagged stayer


def make_model():
    prep = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("prep", prep), ("m", LogisticRegression(max_iter=1000))])


def net_value(y_true, flagged):
    tp = ((flagged == 1) & (y_true == 1)).sum()
    fp = ((flagged == 1) & (y_true == 0)).sum()
    return tp * gain_tp - fp * loss_fp


# 1. choose threshold using cross-validated probabilities on TRAIN only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(make_model(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
vals = [net_value(y_train.values, (oof >= t).astype(int)) for t in grid]
best_t = float(grid[int(np.argmax(vals))])
print(f"Assumptions: offer cost {OFFER_COST}, customer value {CUSTOMER_VALUE}, "
      f"success rate {SUCCESS:.0%} -> gain per caught leaver {gain_tp}, loss per wasted offer {loss_fp}")
print(f"Best threshold on training data: {best_t}")

# 2. fit on train, score test set ONCE
model = make_model().fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print("\nTEST SET (scored once)")
print(f"ROC-AUC {roc_auc_score(y_test, proba):.3f}")
for label, t in [("default 0.50", 0.5), (f"chosen {best_t}", best_t)]:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\nThreshold {label}: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  precision {precision_score(y_test, pred):.3f}  recall {recall_score(y_test, pred):.3f}"
          f"  F1 {f1_score(y_test, pred):.3f}  net value {net_value(y_test.values, pred):,.0f}")

joblib.dump({"model": model, "threshold": best_t,
             "assumptions": {"offer_cost": OFFER_COST, "customer_value": CUSTOMER_VALUE,
                             "success_rate": SUCCESS}}, ROOT / "models" / "churn_model.pkl")
print("\nSaved models/churn_model.pkl")