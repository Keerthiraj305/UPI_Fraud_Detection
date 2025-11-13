
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="UPI Fraud Detection", layout="centered")

st.title("UPI Fraud Detection (Demo)")
st.write("Enter transaction details below and choose a model to predict whether the transaction is likely fraudulent.")

# Load models and feature columns
@st.cache_resource
def load_assets():
    with open("feature_columns.pkl","rb") as f:
        feature_cols = pickle.load(f)
    with open("model_random_forest.pkl","rb") as f:
        rf = pickle.load(f)
    with open("model_gradient_boosting.pkl","rb") as f:
        gb = pickle.load(f)
    return feature_cols, rf, gb

feature_cols, rf, gb = load_assets()

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction amount (INR)", min_value=0.0, value=500.0, step=10.0)
    txn_hour = st.slider("Transaction hour (0-23)", 0, 23, 14)
    user_age = st.number_input("User age", min_value=12, max_value=100, value=30)
    device_change = st.selectbox("New device used?", ("No","Yes"))
    is_foreign = st.selectbox("Foreign transaction?", ("No","Yes"))
    account_age_days = st.number_input("Account age (days)", min_value=0, max_value=20000, value=365)
with col2:
    prev_txn_24h = st.number_input("Previous transactions in last 24h", min_value=0, value=1)
    avg_amount_30d = st.number_input("Average amount in last 30 days", min_value=0.0, value=800.0, step=10.0)
    location_distance_km = st.number_input("Distance from usual location (km)", min_value=0.0, value=1.0, step=0.1)
    merchant_cat = st.selectbox("Merchant category", ["electronics","grocery","food","utilities","travel","entertainment","others"])
    txn_type = st.selectbox("Transaction type", ["p2p","merchant","bill","recharge"])

model_choice = st.radio("Choose model", ("Random Forest","Gradient Boosting"))

if st.button("Predict"):
    # Build input row matching feature columns
    row = {}
    for c in feature_cols:
        row[c] = 0
    row['amount'] = amount
    row['txn_hour'] = txn_hour
    row['user_age'] = user_age
    row['device_change'] = 1 if device_change=="Yes" else 0
    row['is_foreign'] = 1 if is_foreign=="Yes" else 0
    row['account_age_days'] = account_age_days
    row['prev_txn_24h'] = prev_txn_24h
    row['avg_amount_30d'] = avg_amount_30d
    row['location_distance_km'] = location_distance_km

    # categorical
    merchant_col = "merchant_cat_" + merchant_cat
    txn_col = "txn_type_" + txn_type
    if merchant_col in row:
        row[merchant_col] = 1
    if txn_col in row:
        row[txn_col] = 1

    X = pd.DataFrame([row])
    if model_choice == "Random Forest":
        proba = rf.predict_proba(X)[:,1][0]
        pred = rf.predict(X)[0]
    else:
        proba = gb.predict_proba(X)[:,1][0]
        pred = gb.predict(X)[0]

    st.subheader("Result")
    st.write(f"Fraud probability: **{proba:.3f}**")
    st.write("Prediction:", "🔴 **FRAUD**" if pred==1 else "🟢 **LEGIT**")
    st.write("---")
    st.write("Note: This is a synthetic demo model trained on generated data. Do NOT use in production without proper real data, evaluation, and regulatory checks.")
