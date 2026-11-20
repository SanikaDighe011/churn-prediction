"""Phase 4: baseline and logistic regression for churn."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")

y = df["churn_flag"]
X = df.drop(columns=["customer_id", "churn", "churn_flag"])   # no ID, no answer
num_cols = ["tenure", "monthly_charges", "total_charges"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train {len(X_train)} rows ({y_train.mean():.1%} churn) | "
      f"Test {len(X_test)} rows ({y_test.mean():.1%} churn)")


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def report(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\n=== {name} ===")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Accuracy  {accuracy_score(y_test, pred):.3f}")
    print(f"Precision {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"Recall    {recall_score(y_test, pred):.3f}")
    print(f"F1        {f1_score(y_test, pred):.3f}")
    print(f"ROC-AUC   {roc_auc_score(y_test, proba):.3f}")


report("Baseline: always predict 'stay'",
       Pipeline([("prep", preprocessor()), ("m", DummyClassifier(strategy="most_frequent"))]))
report("Logistic Regression (default threshold 0.5)",
       Pipeline([("prep", preprocessor()),
                 ("m", LogisticRegression(max_iter=1000))]))