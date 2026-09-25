"""Streamlit app for churn prediction inference."""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")

model = joblib.load("models/best_model.joblib")

st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.slider("Total charges ($)", 0.0, 9000.0, 800.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])

with col2:
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

row = pd.DataFrame([{
    "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "PhoneService": phone_service, "MultipleLines": multiple_lines, "InternetService": internet,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup,
    "DeviceProtection": device_protection, "TechSupport": tech_support,
    "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
}])

if st.button("Predict churn risk"):
    proba = model.predict_proba(row)[0, 1]
    pred = model.predict(row)[0]
    st.metric("Churn probability", f"{proba:.1%}")
    if pred == 1:
        st.error("Predicted: Likely to churn")
    else:
        st.success("Predicted: Likely to stay")
