"""Phase 5: compare models with 5-fold cross-validation (training data only)."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
y = df["churn_flag"]
X = df.drop(columns=["customer_id", "churn", "churn_flag"])
num_cols = ["tenure", "monthly_charges", "total_charges"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)   # same split as Phase 4


def prep():
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Logistic Regression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                       class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["roc_auc", "precision", "recall", "f1"]
rows = []
for name, m in models.items():
    r = cross_validate(Pipeline([("prep", prep()), ("m", m)]), X_train, y_train,
                       cv=cv, scoring=scoring, return_train_score=True)
    rows.append({"model": name,
                 "AUC_cv": r["test_roc_auc"].mean(), "AUC_std": r["test_roc_auc"].std(),
                 "AUC_train": r["train_roc_auc"].mean(),
                 "precision": r["test_precision"].mean(),
                 "recall": r["test_recall"].mean(), "F1": r["test_f1"].mean()})
res = pd.DataFrame(rows).sort_values("AUC_cv", ascending=False)
pd.options.display.float_format = "{:.3f}".format
print(res.to_string(index=False))