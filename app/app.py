from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
D = Path(__file__).parent / "data"
FIG = ROOT / "reports" / "figures"
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


def show(name, caption=None):
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart {name} not found.")


model = load_model()
train, test = load_scores()

# ---------- sidebar: navigation + ASSUMED costs ----------
st.sidebar.title("Churn Predictor")
page = st.sidebar.radio("Page", ["Overview", "Churn insights", "Predict a customer",
                                 "Business view", "Model performance", "About"])
st.sidebar.divider()
st.sidebar.subheader("Assumed business costs")
st.sidebar.caption("Assumptions, not facts from the dataset.")
offer_cost = st.sidebar.slider("Cost of one retention offer", 1, 100, 10)
cust_value = st.sidebar.slider("Value of a retained customer", 50, 1000, 200, step=10)
success = st.sidebar.slider("Offer success rate (%)", 5, 100, 30) / 100
gain = success * cust_value - offer_cost
loss = offer_cost
threshold = best_threshold(train["y"].values, train["score"].values, gain, loss)
st.sidebar.metric("Best threshold (from training data)", f"{threshold:.2f}")
if gain <= 0:
    st.sidebar.warning("With these costs an offer never pays off for a leaver, "
                       "so the threshold goes to its maximum.")

# ---------- pages ----------
if page == "Overview":
    st.title("Telco Customer Churn: Analysis and Prediction")
    st.write("Which customers are likely to leave, why, and what a retention offer is worth. "
             "Built on the IBM Telco Customer Churn dataset (about 7,000 customers).")
    c = st.columns(3)
    c[0].metric("Customers", "7,043")
    c[1].metric("Churned", "1,869")
    c[2].metric("Churn rate", "26.5%")
    st.subheader("Key findings (from the data)")
    st.markdown("""
- **Contract:** month-to-month customers churn at 42.7%, one-year 11.3%, two-year 2.8%.
- **Tenure:** churn falls from 52.9% (0-6 months) to 9.5% (49-72 months).
- **Internet:** fiber optic 41.9%, DSL 19.0%, no internet 7.4%.
- **Payment:** electronic check 45.3%, other methods 15-19%.
- **Model:** Logistic Regression, test ROC-AUC 0.842. Five models tied in cross-validation.
- **Accuracy misleads:** "always predict stay" is 73.5% accurate but catches no leavers.
""")
    st.info("These are associations, not causes. Use the sidebar to explore.")

elif page == "Churn insights":
    st.header("Churn insights")
    a, b = st.columns(2)
    with a:
        show("02_churn_by_contract.png")
        show("04_churn_by_payment.png")
    with b:
        show("05_churn_by_tenure.png")
        show("03_churn_by_internet.png")
    show("06_monthly_charges.png")
    st.caption("Red dashed line = overall churn rate (26.5%). Contract, tenure and charges "
               "overlap, so the patterns are partly the same pattern seen from different angles.")

elif page == "Predict a customer":
    st.header("Predict a customer")
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
    st.progress(min(max(p, 0.0), 1.0))
    if p >= threshold:
        st.error(f"Flag for a retention offer (above the threshold {threshold:.2f}).")
    else:
        st.success(f"Do not flag (below the threshold {threshold:.2f}).")
    st.caption("Overall churn rate in the data is 26.5%. The probability is the model's estimate "
               "from past customers. It does not prove why a customer would leave. Combinations "
               "that do not occur in the data (for example fiber with 'No internet service' for "
               "a service) can give unreliable results.")

elif page == "Business view":
    st.header("Business view")
    st.write("Test set: 1,409 customers the model never saw. The threshold was chosen on "
             "training data only.")
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
    st.info("Move the sliders in the sidebar. The best threshold and net value change, because "
            "the right cut-off depends on costs the dataset does not contain.")

elif page == "Model performance":
    st.header("Model performance")
    st.subheader("5-fold cross-validation on training data")
    st.dataframe(pd.DataFrame({
        "model": ["Gradient Boosting", "Logistic Regression", "Logistic Regression (balanced)",
                  "Random Forest (balanced)", "Random Forest"],
        "AUC": [0.848, 0.846, 0.846, 0.845, 0.845],
        "AUC on training folds": [0.888, 0.850, 0.849, 0.934, 0.933],
        "precision": [0.661, 0.655, 0.517, 0.540, 0.676],
        "recall": [0.534, 0.543, 0.801, 0.765, 0.498]}), use_container_width=True)
    st.write("All five tie on AUC (differences are smaller than the spread). The random forests "
             "overfit (training AUC 0.93 vs 0.845). Logistic Regression was chosen: simple, "
             "explainable, no overfitting. Final test ROC-AUC: 0.842.")
    a, b = st.columns(2)
    with a:
        show("07_coefficients.png")
    with b:
        show("08_shap_summary.png")
    st.caption("Tenure, contract and internet type matter most. Charges columns overlap with "
               "tenure, so their individual weights should not be over-read.")

else:
    st.header("About")
    st.markdown("""
**Model:** Logistic Regression, scored once on a held-out test set (ROC-AUC 0.842).

**Limitations**
- The dataset has no dates, so we cannot check how the model ages.
- Offer cost, customer value and success rate are assumptions set in the sidebar.
- Contract, tenure and charges overlap, so single coefficients should not be over-read.
- Associations, not causes.

**Data:** IBM Telco Customer Churn dataset (Kaggle), licence CC BY-SA 4.0.
""")