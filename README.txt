
UPI Fraud Detection Demo
------------------------
Contents:
- upi_synthetic.csv : original synthetic dataset (readable)
- upi_synthetic_encoded.csv : encoded dataset used for training
- model_random_forest.pkl : trained RandomForest model
- model_gradient_boosting.pkl : trained GradientBoosting model
- feature_columns.pkl : list of feature columns expected by the app
- app.py : Streamlit app. Run with `streamlit run app.py` from the folder.
- requirements.txt : Python packages

Notes:
- This is a demo using synthetic data. For production use, replace the dataset with real labeled transactions,
  perform proper feature engineering, validation, and comply with legal/privacy requirements.
