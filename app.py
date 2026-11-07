import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("diabetes_prediction_pipeline.pkl")

st.title("Diabetes Prediction App")

st.write("Enter the details below to get a model prediction.")

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.selectbox(
    "Smoking History",
    ["never", "No Info", "current", "former", "ever", "not current"]
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

hba1c = st.number_input(
    "HbA1c Level",
    min_value=0.0,
    value=5.5
)

blood_glucose = st.number_input(
    "Blood Glucose Level",
    min_value=0.0,
    value=100.0
)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.write(
        f"Estimated class 1 probability: {probability:.2%}"
    )

    if prediction == 1:
        st.warning("Model prediction: Diabetes")
    else:
        st.success("Model prediction: No Diabetes")

st.caption(
    "Educational demonstration only. "
    "This tool is not intended for medical diagnosis."
)