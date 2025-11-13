import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# Robust import for utils: try top-level then package-style
try:
    from utils import load_data, train_models, feature_schema_example, DATASETS
except Exception:
    from src.utils import load_data, train_models, feature_schema_example, DATASETS

st.set_page_config(page_title="UPI Fraud Detection (ANN & RF)", layout="wide")
st.title("🛡️ UPI Fraud Detection — Live Prediction")
st.caption("Models: ANN & Random Forest — choose a dataset and train the models to enable predictions.")

# Sidebar dataset picker
dataset_choice = st.sidebar.selectbox("Select Dataset", list(DATASETS.keys()))

# Training helper (cached)
@st.cache_resource(show_spinner=True)
def _train(path):
    df = load_data(path)
    models, testset, results = train_models(df)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(models[0], "models/ann.joblib")
    joblib.dump(models[1], "models/random_forest.joblib")
    return models, testset, results

# Session state for models
if "models" not in st.session_state:
    st.session_state.models = None
    st.session_state.testset = None
    st.session_state.results = None

if st.sidebar.button("Train Models"):
    with st.spinner("Training models..."):
        models, testset, results = _train(DATASETS[dataset_choice])
        st.session_state.models = models
        st.session_state.testset = testset
        st.session_state.results = results
        st.success("Training complete.")

if st.session_state.models is None:
    st.warning("Models are not trained — click 'Train Models' in the sidebar to train on the selected dataset.")
else:
    ann_pipe, rf_pipe = st.session_state.models
    results = st.session_state.results

with st.expander("Dataset info & class balance", expanded=False):
    df = load_data(DATASETS[dataset_choice])
    from utils import split_xy
    X, y = split_xy(df)
    st.write("Rows:", len(df))
    fraud_count = int(y.sum())
    fraud_pct = (fraud_count / len(y)) * 100 if len(y) else 0
    st.write(f"Class counts: Legit={len(y)-fraud_count}, Fraud={fraud_count} ({fraud_pct:.1f}%)")
    st.dataframe(df.head(20))

st.subheader("Enter a transaction")
schema = feature_schema_example()

with st.form("txn"):
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount (INR)", min_value=1.0, max_value=25000.0, value=float(schema["amount"]), step=1.0)
    hour = c2.number_input("Hour (0-23)", min_value=0, max_value=23, value=int(schema["hour"]), step=1)
    tx_type = c3.selectbox("Transaction type", ["P2P","P2M","Refund"], index=["P2P","P2M","Refund"].index(schema["transaction_type"]))
    c4, c5, c6 = st.columns(3)
    sender_state = c4.selectbox("Sender state", ["KA","MH","DL","TN","KL","GA","RJ","UP","GJ","WB","MP","AP","TS","BR","HR"], index=0)
    receiver_state = c5.selectbox("Receiver state", ["KA","MH","DL","TN","KL","GA","RJ","UP","GJ","WB","MP","AP","TS","BR","HR"], index=1)
    device_type = c6.selectbox("Device type", ["Android","iOS","FeaturePhone"], index=0)
    c7, c8, c9 = st.columns(3)
    sender_bank = c7.selectbox("Sender bank", ["HDFC","SBI","ICICI","AXIS","KOTAK","PNB","CANARA","BOI","UBI","IDBI"], index=1)
    receiver_bank = c8.selectbox("Receiver bank", ["HDFC","SBI","ICICI","AXIS","KOTAK","PNB","CANARA","BOI","UBI","IDBI"], index=0)
    is_verified = c9.selectbox("Is VPA verified?", ["No","Yes"], index=1)
    c10, c11 = st.columns(2)
    past_txn_count_7d = c10.number_input("Past txn count (7d)", min_value=0, max_value=200, value=5, step=1)
    past_avg_amount_7d = c11.number_input("Past avg amount (7d)", min_value=1.0, max_value=25000.0, value=700.0, step=1.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    if st.session_state.models is None:
        st.warning("Please train the models first using the 'Train Models' button in the sidebar.")
    else:
        row = pd.DataFrame([{
            "amount": amount,
            "hour": int(hour),
            "sender_state": sender_state,
            "receiver_state": receiver_state,
            "sender_bank": sender_bank,
            "receiver_bank": receiver_bank,
            "transaction_type": tx_type,
            "past_txn_count_7d": int(past_txn_count_7d),
            "past_avg_amount_7d": float(past_avg_amount_7d),
            "device_type": device_type,
            "is_vpa_verified": 1 if is_verified=="Yes" else 0
        }])
        ann_p = ann_pipe.predict_proba(row)[0,1]
        rf_p = rf_pipe.predict_proba(row)[0,1]
        ann_pred = "FRAUD" if ann_p>=0.5 else "LEGIT"
        rf_pred = "FRAUD" if rf_p>=0.5 else "LEGIT"

        st.success("Prediction complete.")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ANN", f"{ann_pred}", f"Prob: {ann_p:.3f}")
        with c2:
            st.metric("Random Forest", f"{rf_pred}", f"Prob: {rf_p:.3f}")
        st.info("Tip: Try odd hours (e.g., 1 AM), high amounts, low 7‑day history, and unverified VPA to see fraud scores increase.")

st.caption("Models retrain when requested and are saved in ./models.")
