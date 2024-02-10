import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
model = pickle.load(open('model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Function to preprocess input data
# Function to preprocess input data
def preprocess_data(data):

    feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFlaggedFraud']
    
    data['type'] = data['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})
    
    # Feature scaling
    data_scaled = scaler.transform(data[feature_names])
    
    return data_scaled

# Streamlit App
def main():
    st.title("Fraud Transaction Detection App")

    # Get user input
    st.header("Enter Transaction Details:")
    step = st.number_input("Step", min_value=1)
    type_val = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
    amount = st.number_input("Amount")
    oldbalanceOrg = st.number_input("Old Balance Origin")
    oldbalanceDest = st.number_input("Old Balance Destination")
    isFlaggedFraud = st.checkbox("Flagged Fraud")

    # Submit Button
    if st.button("Submit"):
        # Create a DataFrame with user input
        user_data = pd.DataFrame({
            'step': [step],
            'type': [type_val],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'oldbalanceDest': [oldbalanceDest],
            'isFlaggedFraud': [isFlaggedFraud]
        })

        # Preprocess the user input
        user_data_scaled = preprocess_data(user_data)

        # Make a prediction
        prediction = model.predict(user_data_scaled)

        # Display the result
        st.header("Prediction:")
        if prediction[0] == 1:
            st.error("This transaction is predicted as Fraud!")
        else:
            st.success("This transaction is predicted as Not Fraud.")

if __name__ == '__main__':
    main()
