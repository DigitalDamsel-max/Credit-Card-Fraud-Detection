import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and columns
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details.")

# --- User inputs ---
amount = st.number_input("Transaction Amount ($)", min_value=0.0)
hour = st.slider("Hour of Transaction", 0, 23)
txn_type = st.selectbox("Transaction Type", ["online", "pos", "atm"])
merchant_category = st.selectbox(
    "Merchant Category", ["grocery", "fuel", "electronics", "travel"]
)
txns_last_24h = st.slider("Transactions in last 24 hours", 0, 50)
avg_amount_7d = st.number_input("Average Amount in last 7 days ($)", min_value=0.0)

# --- Prepare input ---
input_dict = {
    "amount": amount,
    "hour": hour,
    "txns_last_24h": txns_last_24h,
    "avg_amount_7d": avg_amount_7d,
    "txn_type_pos": 0,
    "txn_type_online": 0,
    "merchant_category_electronics": 0,
    "merchant_category_fuel": 0,
    "merchant_category_travel": 0
}

# One-hot encode user selection
if txn_type != "atm":
    input_dict[f"txn_type_{txn_type}"] = 1

if merchant_category != "grocery":
    input_dict[f"merchant_category_{merchant_category}"] = 1

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Scale numeric features
input_scaled = scaler.transform(input_df)

# --- Predict and explain ---
if st.button("Check Fraud"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction (Risk: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk: {probability:.2%})")

    # --- Feature contribution explanation ---
    coefs = model.coef_[0]
    contributions = input_scaled[0] * coefs
    contrib_df = pd.DataFrame({
        "feature": input_df.columns,
        "contribution": contributions
    }).sort_values(by="contribution", ascending=False)

    st.subheader("Why this transaction was flagged:")
    st.table(contrib_df.head(5))  # show top 5 contributing features
