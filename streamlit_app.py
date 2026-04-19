import streamlit as st
import pandas as pd
from model_training_water_quality import train_pipeline_improved

st.title("💧 Water Quality Prediction App")

# Run training pipeline
if st.button("Run Training"):
    result = train_pipeline_improved()
    st.write("Training completed!")
    st.write(result)

# Upload CSV for prediction
uploaded_file = st.file_uploader("Upload water quality data (CSV)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Example placeholder for predictions
    # predictions = predict_pipeline(data)
    # st.write("Predictions:", predictions)

