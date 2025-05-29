
import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("loan_model_georgia.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Georgia State University - Loan Prediction")

# User inputs
gender = st.selectbox("Gender", encoders['Gender'].classes_)
married = st.selectbox("Married", encoders['Married'].classes_)
education = st.selectbox("Education", encoders['Education'].classes_)
self_employed = st.selectbox("Self Employed", encoders['Self_Employed'].classes_)
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)

if st.button("Predict Loan Status"):
    input_data = np.array([[
        encoders['Gender'].transform([gender])[0],
        encoders['Married'].transform([married])[0],
        encoders['Education'].transform([education])[0],
        encoders['Self_Employed'].transform([self_employed])[0],
        income,
        loan_amount
    ]])
    prediction = model.predict(input_data)[0]
    result = encoders['Loan_Status'].inverse_transform([prediction])[0]
    if result == 'Y':
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Denied ❌")
