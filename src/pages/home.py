import streamlit as st
import pandas as pd
from src.utils import load_data, train_models

def main():
    st.title("UPI Fraud Detection")
    st.write("Welcome to the UPI Fraud Detection application.")
    
    # Load data
    try:
        df = load_data()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Display data summary
    st.subheader("Data Summary")
    st.write(df.describe())

    # Train models
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            models, (X_test, y_test), results = train_models(df)
            st.success("Models trained successfully!")
            st.write("Results:", results)

if __name__ == "__main__":
    main()