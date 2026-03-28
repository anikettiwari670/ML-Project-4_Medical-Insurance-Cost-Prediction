import streamlit as st
import joblib
import numpy as np
import pandas as pd 

# Load the trained model.
model = joblib.load("Medical_Insurance_Cost_Prediction.pkl")

# Load the scaler.
scaler = joblib.load("Scaler.pkl")

# Load the columns. 
model_columns = joblib.load("Model_Columns.pkl")

# UI set up. 
st.set_page_config(page_title = "Medical Insurance Cost Prediction", page_icon = ":hospital:", layout = "centered")
st.title("🏥 **Medical Insurance Cost Prediction**")

st.write("Enter the Details Below to get an Estimate of the Medical Insurance Cost:")

# User input fields.
Age = st.number_input("Age", min_value = 0, max_value = 120)
Sex = st.selectbox("Sex", options = ["male", "female"])
Bmi = st.number_input("BMI", min_value = 0.0, max_value = 50.0)
Children = st.number_input("Number of Children", min_value = 0, max_value = 10)
Smoker = st.selectbox("Smoker", options = ["yes", "no"])
Region = st.selectbox("Region", options = ["northeast", "northwest", "southeast", "southwest"])

# Prepare the input data for prediction.
if st.button("Predict"):
    # Create a DataFrame for the input data.
    input_data = pd.DataFrame([{
        "age": Age,
        "sex": Sex,
        "bmi": Bmi,
        "children": Children,
        "smoker": Smoker,
        "region": Region
    }])

# Replicate the one-hot encoding (get_dummies) for the categorical variables.
    input_encoded = pd.get_dummies(input_data)

# Align with model columns. 
    input_final = input_encoded.reindex(columns = model_columns, fill_value = 0)

# Scale the input data.
    input_scaled = scaler.transform(input_final)

# Predict the insurance cost.
    predicted_cost = model.predict(input_scaled)

# Display the predicted cost.
    st.subheader("Estimated Medical Insurance Cost:")
    st.write(f"${predicted_cost[0]:.2f}")
