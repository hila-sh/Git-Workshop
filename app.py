import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="ML Experiment Tracker", layout="centered")

st.title("📊 ML Experiment Tracker")
st.write("This application displays the performance metrics of your latest trained model.")

# Check if the training metrics file exists
if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Display key metrics in a dashboard grid
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Architecture", value=metrics["model_type"])
    col2.metric(label="Accuracy Score", value=f"{metrics['accuracy'] * 100:.2f}%")
    col3.metric(label="Max Depth Limit", value=metrics["max_depth"])
    
    st.subheader("Experiment Configuration (Hyperparameters)")
    st.json(metrics)
    
    # Visual simulation of optimization progress
    st.subheader("Performance vs Baseline Visualization")
    chart_data = pd.DataFrame([metrics["accuracy"]], columns=["Current Run Accuracy"])
    st.line_chart(chart_data)

else:
    st.warning("⚠️ No evaluation results found. Please run 'python train.py' first to train the model and generate metrics.")
