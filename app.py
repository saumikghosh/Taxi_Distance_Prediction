import joblib
import numpy as np
import pandas as pd
import streamlit as st

with open('taxi_distance_prediction_model.joblib','rb') as f:   
    model = joblib.load(f)
st.title("Taxi Trip Distance Prediction")
st.subheader("Enter the details below to predict the trip distance:")


# Let's take inputs from the user
fare_estimate = st.number_input("Fare Estimate ($):", min_value=0.0)
tip_amount = st.number_input("Tip Amount ($):", min_value=0.0)
tolls_amount = st.number_input("Tolls Amount ($):", min_value=0.0)
passenger_count = st.number_input("Passenger Count:", min_value=1, step=1)
input_list = [passenger_count,tolls_amount,tip_amount,fare_estimate]
# Make predictions
if st.button("Predict Trip Distance"):
    input_array = np.array(input_list).reshape(1,-1)
    prediction = model.predict(input_array)
    st.success(f"The predicted trip distance is: {prediction[0]:.2f} miles")