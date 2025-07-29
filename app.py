import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.compose._column_transformer as ct

if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# Loading the saved Decision Tree model and preprocessor
model = joblib.load('dt_fraud_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Dashboard: Fraud Detection In Transactions")
st.write("""
Upload a CSV file containing transaction data to detect fraudulent transactions.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # Drop unnecessary columns if present
    drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')

    try:
        X_processed = preprocessor.transform(data)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # Predict fraud labels
    try:
        predictions = model.predict(X_processed)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Count the number of transactions in file
    st.write(f"Total Number of Transactions: {len(data)}")

    # Append predictions to the original data
    data['Predicted_Fraud_Label'] = predictions

    st.subheader("Prediction Results")
    st.write(data.head())

    # Show summary statistics
    st.subheader("Prediction Summary")
    fraud_counts = data['Predicted_Fraud_Label'].value_counts()
    st.bar_chart(fraud_counts)

    # Count the number of fraud and not fraud predictions
    fraud_count = (predictions == 1).sum()
    not_fraud_count = (predictions == 0).sum()
    st.write(f"Number of Fraudulent Transactions Detected : {fraud_count}")
    st.write(f"Number of Non-Fraudulent Transactions Detected : {not_fraud_count}")

else:
    st.info("Please upload a CSV file to start fraud detection.")
