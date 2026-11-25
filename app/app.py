from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
D = Path(__file__).parent / "data"
st.set_page_config(page_title="Churn Predictor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.pkl")["model"]


@st.cache_data
def load_scores():
    return (pd.read_csv(D / "train_oof_scores.csv"), pd.read_csv(D / "test_scores.csv"))


def net_value(y, flagged, gain, loss):
    tp = ((flagged == 1) & (y == 1)).sum()
    fp = ((flagged == 1) & (y == 0)).sum()
    return tp * gain - fp * loss


def best_threshold(y, s, gain, loss):
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    vals = [net_value(y, (s >= t).astype(int), gain, loss) for t in grid]
    return float(grid[int(np.argmax(vals))])


model = load_model()
train, test = load_scores()

# ---------- sidebar: ASSUMED business costs ----------
st.sidebar.header("Assumed business costs")
st.sidebar.caption("These are assumptions, not facts from the dataset.")
offer_cost = st.sidebar.slider("Cost of one retention offer", 1, 100, 10)
cust_value = st.sidebar.slider("Value of a retained customer", 50, 1000, 200, step=10)
success = st.sidebar.slider("Offer success rate (%)", 5, 100, 30) / 100
gain = success * cust_value - offer_cost
loss = offer_cost
threshold = best_threshold(train["y"].values, train["score"].values, gain, loss)
st.sidebar.metric("Best threshold (chosen on training data)", f"{threshold:.2f}")
if gain <= 0:
    st.sidebar.warning("With these costs an offer never pays off for a leaver, "
                       "so the threshold is pushed to its maximum.")

st.title("Telco Customer Churn Predictor")
tab1, tab2, tab3 = st.tabs(["Predict a customer", "Business view", "About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)",
                                                  "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges", 10.0, 150.0, 70.0)
        total = st.number_input("Total charges (approx. tenure x monthly)", 0.0, 10000.0,
                                float(round(tenure * monthly, 2)))
    with c3:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        svc = ["No", "Yes", "No internet service"]
        security = st.selectbox("Online security", svc)
        backup = st.selectbox("Online backup", svc)
        protection = st.selectbox("Device protection", svc)
        support = st.selectbox("Tech support", svc)
        tv = st.selectbox("Streaming TV", svc)
        movies = st.selectbox("Streaming movies", svc)

    row = pd.DataFrame([{
        "gender": gender, "senior_citizen": senior, "partner": partner,
        "dependents": dependents, "tenure": tenure, "phone_service": phone,
        "multiple_lines": lines, "internet_service": internet, "online_security": security,
        "online_backup": backup, "device_protection": protection, "tech_support": support,
        "streaming_tv": tv, "streaming_movies": movies, "contract": contract,
        "paperless_billing": paperless, "payment_method": payment,
        "monthly_charges": monthly, "total_charges": total}])
    p = float(model.predict_proba(row)[:, 1][0])
    st.subheader(f"Predicted churn probability: {p:.1%}")
    if p >= threshold:
        st.error(f"Flag for a retention offer (probability is above the threshold {threshold:.2f}).")
    else:
        st.success(f"Do not flag (probability is below the threshold {threshold:.2f}).")
    st.caption("Overall churn rate in the data is 26.5%. The probability is the model's estimate "
               "from patterns in past customers. It does not prove why a customer would leave. "
               "Combinations that do not occur in the data (for example fiber with 'No internet "
               "service' for a service) can give unreliable results.")

with tab2:
    st.write("Test set: 1,409 customers the model never saw (scored with the threshold chosen on "
             "training data).")
    rows = []
    for label, t in [("Default 0.50", 0.5), (f"Chosen {threshold:.2f}", threshold)]:
        flagged = (test["score"].values >= t).astype(int)
        y = test["y"].values
        tp = int(((flagged == 1) & (y == 1)).sum())
        fp = int(((flagged == 1) & (y == 0)).sum())
        fn = int(((flagged == 0) & (y == 1)).sum())
        rows.append({"threshold": label, "caught leavers (TP)": tp, "wasted offers (FP)": fp,
                     "missed leavers (FN)": fn,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
                     "recall": round(tp / (tp + fn), 3),
                     "net value": round(net_value(y, flagged, gain, loss))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.info("Move the sliders in the sidebar: the best threshold and the net value change. "
            "That is the point: the right cut-off depends on costs the dataset does not contain.")

with tab3:
    st.markdown("""
**Model:** Logistic Regression (tied with Random Forest and Gradient Boosting on ROC-AUC
about 0.85 in cross-validation; chosen because it is simple and explainable). Test ROC-AUC 0.842.

**Limitations**
- The dataset has no dates, so we cannot check how the model ages.
- Offer cost, customer value and success rate are assumptions you set in the sidebar.
- Contract, tenure and charges overlap, so single coefficients should not be over-read.
- Associations, not causes.

**Data:** IBM Telco Customer Churn dataset (Kaggle), licence CC BY-SA 4.0.
""")