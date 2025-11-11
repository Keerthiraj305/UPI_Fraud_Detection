import streamlit as st
import pandas as pd
from utils import load_data, train_models

# Set the title of the app
st.title("UPI Fraud Detection")

# Load the data
@st.cache
def load_data_cached():
    return load_data()

data = load_data_cached()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Explore"])

if page == "Home":
    st.header("Home")
    st.write("Welcome to the UPI Fraud Detection application.")
    st.write("This application uses machine learning models to detect fraudulent transactions.")
    
    # Display some basic statistics
    st.subheader("Data Overview")
    st.write(data.describe())

elif page == "Explore":
    st.header("Explore Data")
    st.write("This page allows you to explore the dataset.")
    
    # Display the data
    st.subheader("Transaction Data")
    st.write(data)

    # Optionally, you can add visualizations or other exploratory analysis here

# Train models button
if st.sidebar.button("Train Models"):
    st.write("Training models...")
    models, _, results = train_models(data)
    st.write("Models trained successfully!")