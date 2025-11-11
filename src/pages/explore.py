import streamlit as st
import pandas as pd
# Support both running as a package (src.utils) and as flat module (utils)
try:
    from src.utils import load_data  # type: ignore
except ModuleNotFoundError:
    from utils import load_data

def main():
    st.title("Exploratory Data Analysis")
    
    # Load data
    df = load_data()
    
    # Display dataset
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    # Display statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    # Display missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    # Display distribution of the target variable
    st.subheader("Target Variable Distribution")
    st.bar_chart(df['fraud_flag'].value_counts())
    
    # Additional EDA features can be added here

if __name__ == "__main__":
    main()