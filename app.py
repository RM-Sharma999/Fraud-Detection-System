
import streamlit as st
import pandas as pd
import pickle
from utils import FeatureEngineer

# Load the pickled file
with open("calibrated_xgb_clf_with_tuned_threshold.pkl", "rb") as f:
    model_dict = pickle.load(f)

# Extract model + threshold
model = model_dict["pipeline"]
threshold = model_dict["threshold"]

# Page title
st.title("Fraud Detection System")
st.markdown("Please enter the transaction details")

st.divider()

# Input fields: 3 columns x 2 rows
col1, col2, col3 = st.columns(3)

with col1:
    transaction_type = st.selectbox(
        "Select Transaction Type",
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    )

with col2:
    amount = st.number_input(
        "Enter the Amount",
        min_value=0.0,
        value=1000.0
    )

with col3:
    oldbalanceOrg = st.number_input(
        "Enter the Old Balance of the Sender",
        min_value=0.0,
        value=10000.0
    )

with col1:
    newbalanceOrig = st.number_input(
        "Enter the New Balance of the Sender",
        min_value=0.0,
        value=9000.0
    )

with col2:
    oldbalanceDest = st.number_input(
        "Enter the Old Balance of the Receiver",
        min_value=0.0,
        value=0.0
    )

with col3:
    newbalanceDest = st.number_input(
        "Enter the New Balance of the Receiver",
        min_value=0.0,
        value=1000.0
    )

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])

    proba = model.predict_proba(input_data)[0, 1]
    prediction = int(proba >= threshold)

    st.subheader(f"Prediction: '{int(prediction)}'")

    if prediction == 1:
        st.error("This Transaction can be fraud")
    else:
        st.success("This Transaction looks legit")
