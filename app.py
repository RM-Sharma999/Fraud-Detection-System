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

st.title("Fraud Detection System")

st.markdown("Please enter the transactions details")

st.divider()

transaction_type = st.selectbox("Select Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount = st.number_input("Enter the Amount", min_value = 0.0, value = 1000.0)
oldbalanceOrg = st.number_input("Enter the Old Balance of the Sender", min_value = 0.0, value = 10000.0)
newbalanceOrig = st.number_input("Enter the New Balance of the Sender", min_value = 0.0, value = 9000.0)
oldbalanceDest = st.number_input("Enter the Old Balance of the Receiver", min_value = 0.0, value = 0.0)
newbalanceDest = st.number_input("Enter the New Balance of the Receiver", min_value = 0.0, value = 1000.0)

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