import streamlit as st
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("Iris Flower Prediction App")

# Input fields for user data
st.header("Input Features")
sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.slider("Sepal Width", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.slider("Petal Length", min_value=0.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.slider("Petal Width", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Create a 2D array for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Output the prediction
    st.subheader("Prediction")
    st.write(f"Predicted class: {prediction[0]}")
    st.write(f"Prediction probabilities: {prediction_proba}")