
import streamlit as st
import numpy as np
import joblib

# Load the trained model and separate scalers
Med_Ins_model = joblib.load('medical_insurance.pkl')
x_scaler = joblib.load('x_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

st.title('Medical Insurance Predictor')

age = st.slider("Age", 18, 100, 50)

sex = st.radio("Sex (0: Male, 1: Female)", [0, 1])

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=60.0,
    value=30.0
)

smoker = st.selectbox(
    "Smoker (0: No, 1: Yes)",
    [0, 1]
)

region = st.radio(
    "Region (0: Southeast, 1: Southwest, "
    "2: Northeast, 3: Northwest)",
    [0, 1, 2, 3]
)

if st.button("Predict"):
    # Prepare the five input features in training order
    input_data = np.array([[
        age, sex, bmi, smoker, region
    ]])

    # Scale input features
    scaled_input = x_scaler.transform(input_data)

    # Predict scaled insurance charges
    prediction_scaled = Med_Ins_model.predict(scaled_input)

    # Convert prediction back to original charges
    prediction = y_scaler.inverse_transform(
        np.array(prediction_scaled).reshape(-1, 1)
    )[0, 0]

    st.success(
        f"Predicted Medical Insurance Charges: ${prediction:,.2f}"
    )
