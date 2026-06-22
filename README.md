# 🏋️ FitSense AI

FitSense AI is a Machine Learning web application that predicts workout intensity based on user fitness metrics.

## Features

- Predicts workout intensity:
  - Low
  - Medium
  - High
- Built using TensorFlow Neural Networks
- Interactive Streamlit dashboard
- Confidence score for predictions

## Input Features

- Gender
- Age
- Height
- Weight
- Workout Duration
- Heart Rate
- Body Temperature

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit

## Model Architecture

- Input Layer: 7 Features
- Hidden Layer 1: 32 Neurons (ReLU)
- Hidden Layer 2: 16 Neurons (ReLU)
- Output Layer: 3 Neurons (Softmax)

## Project Structure
fitsense-ai/
├── app/
│ └── streamlit_app.py
├── data/
├── models/
│ ├── best_model.keras
│ └── scaler.pkl
├── notebooks/
├── requirements.txt
└── README.md


## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Future Improvements
    -Calorie prediction
    -Workout recommendations
    -Advanced fitness analytics
    -Model explainability


---

### Before pushing
Make sure `.gitignore` contains:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/

Do NOT ignore:
models/best_model.keras
models/scaler.pkl

Those must be pushed, otherwise the deployed app won't work.