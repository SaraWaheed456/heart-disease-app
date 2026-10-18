"""
Heart Disease Prediction System - Web App
------------------------------------------
A Streamlit web app for the Heart Disease Prediction System. It loads the
trained Support Vector Classifier (svc_trained_model.pkl) produced by the
Heart_Disease_Prediction notebook, and recreates the same LabelEncoders
(fit on the same fixed category lists) used during training, so that
predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes.
    """
    encoders = {}
    encoders["Sex"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    encoders["ExerciseAngina"] = LabelEncoder().fit(["Yes", "No"])
    encoders["FastingBloodSugar"] = LabelEncoder().fit(["High", "Normal"])
    return encoders


model = load_model()
encoders = build_label_encoders()

st.title("Heart Disease Prediction System")
st.markdown(
    """
    This application predicts whether a patient has heart disease, using a
    Support Vector Classifier trained on the Cleveland Heart Disease dataset
    (UCI Machine Learning Repository). Select the four patient attributes
    below and click **Predict**.

    *Model accuracy on held-out test data: **75%** (303-instance dataset).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    )
with col2:
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"], index=1)
    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar (High means above 120 mg/dl)", ["High", "Normal"], index=1
    )

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "FastingBloodSugar": [fasting_blood_sugar],
    })

    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "FastingBloodSugar": encoders["FastingBloodSugar"].transform(user_input["FastingBloodSugar"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### HEART DISEASE PRESENT")
    else:
        st.success("### NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for educational purposes. This tool is not a medical device and "
    "must not be used for real medical diagnosis."
)
