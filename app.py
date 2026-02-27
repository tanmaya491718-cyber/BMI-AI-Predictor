import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------------------
# Load saved ML objects
# ---------------------------
model = pickle.load(open("bmi_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------------------
# Page settings
# ---------------------------
st.set_page_config(page_title="BMI Predictor", layout="centered")

st.title("💪 BMI Level Prediction System")
st.write("Enter your details to predict BMI category using Machine Learning")

# ---------------------------
# User Inputs
# ---------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0)

# ---------------------------
# Prediction button
# ---------------------------
if st.button("Predict BMI Category"):

    # Convert gender to numeric (same as training)
    gender_male = 1 if gender == "Male" else 0

    # Create dataframe in SAME column order as training
    input_data = pd.DataFrame(
        [[height, weight, gender_male]],
        columns=columns
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Model prediction
    prediction = model.predict(input_scaled)[0]

    # Show result
    st.success(f"Predicted BMI Category: {prediction}")