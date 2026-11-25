"""Automated checks. Run from the project root:  python -m pytest -v"""
from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(ROOT / "models" / "churn_model.pkl")


def test_row_count(df):
    assert len(df) == 7043


def test_customer_id_unique(df):
    assert df["customer_id"].is_unique


def test_churn_values(df):
    assert set(df["churn"].unique()) == {"Yes", "No"}
    assert (df["churn_flag"] == (df["churn"] == "Yes")).all()


def test_churn_rate(df):
    assert abs(df["churn_flag"].mean() - 0.265) < 0.001


def test_numbers_valid(df):
    assert (df["tenure"] >= 0).all()
    assert (df["monthly_charges"] > 0).all()
    assert pd.api.types.is_numeric_dtype(df["total_charges"])


def test_threshold_in_range(bundle):
    assert 0 < bundle["threshold"] < 1


def test_model_beats_random(df, bundle):
    y = df["churn_flag"]
    X = df.drop(columns=["customer_id", "churn", "churn_flag"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    auc = roc_auc_score(y_test, bundle["model"].predict_proba(X_test)[:, 1])
    assert auc > 0.80


def test_model_returns_probabilities(df, bundle):
    X = df.drop(columns=["customer_id", "churn", "churn_flag"]).head(20)
    p = bundle["model"].predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()