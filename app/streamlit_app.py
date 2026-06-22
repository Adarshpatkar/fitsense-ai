# App Structue:
# User Inputs
#      ↓
# Convert Gender to 0/1
#      ↓
# Create DataFrame
#      ↓
# Apply scaler.transform()
#      ↓
# model.predict()
#      ↓
# Get probabilities
#      ↓
# argmax()
#      ↓
# Convert class → Low/Medium/High
#      ↓
# Display result


import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

model = load_model("models/best_model.keras")

scaler = joblib.load("models/scaler.pkl")

st.title("🏋️ FitSense AI")

st.write(
    "Predict workout intensity based on fitness metrics."
)

gender = st.radio(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=25,
    step=1
)

height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0,
    step=1.0
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)

duration = st.number_input(
    "Duration (minutes)",
    min_value=1.0,
    max_value=180.0,
    value=30.0,
    step=1.0
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=50.0,
    max_value=220.0,
    value=120.0,
    step=1.0
)

body_temp = st.number_input(
    "Body Temperature",
    min_value=35.0,
    max_value=45.0,
    value=38.5,
    step=0.1
)

if st.button("Predict Workout Intensity"):

    gender_encoded = 1 if gender == "Male" else 0

    input_df = pd.DataFrame(
        [[
            gender_encoded,
            age,
            height,
            weight,
            duration,
            heart_rate,
            body_temp
        ]],
        columns=[
            "Gender",
            "Age",
            "Height",
            "Weight",
            "Duration",
            "Heart_Rate",
            "Body_Temp"
        ]
    )

    scaled_input = scaler.transform(input_df)

    try:
        import tensorflow as tf

        prediction = model(
            tf.convert_to_tensor(scaled_input),
            training=False
        )

        prediction = prediction.numpy()

        predicted_class = np.argmax(prediction)

        labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        confidence = prediction[0][predicted_class] * 100

        st.success(
            f"🏋️ Predicted Workout Intensity: {labels[predicted_class]}"
        )

        st.info(
            f"Model Confidence: {confidence:.2f}%"
)

    except Exception as e:
        st.error(f"ERROR: {e}")