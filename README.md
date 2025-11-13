# UPI Fraud Detection (Streamlit App)

This Streamlit application trains and serves two fraud detection models (Neural Network & Random Forest) for UPI transactions. It includes an interactive prediction form and a model insights page with evaluation metrics and visualizations.

## Features
- Single transaction prediction (compares Neural Network & Random Forest probabilities)
- Multiple datasets with different fraud rates (40% and 10%)
- Interactive dataset selection
- Models trained on page load
- Comprehensive metrics: Accuracy, ROC curve & AUC, Precision-Recall curve & Average Precision, Confusion Matrix, Classification Report

## Datasets
Two synthetic datasets are provided:
- `data/upi_transactions_2025.csv` - Standard dataset with 40% fraud rate for testing model performance
- `data/upi_transactions_2025_low_fraud.csv` - Balanced dataset with 10% fraud rate, closer to real-world scenarios

## Project Structure
```
Home.py                  # Main Streamlit entry point
pages/1_Model Insights.py# Insights/metrics page
utils.py                 # Data load, preprocessing, training utilities
models/*.joblib          # Serialized trained models
data/upi_transactions_2025.csv
requirements.txt         # Locked dependencies
runtime.txt              # Python version pin for Streamlit Cloud (python-3.12)
.gitignore               # Git ignore rules
```

## Quick Start
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Home.py
```
Visit http://localhost:8501

## Deployment (Streamlit Cloud)
1. Ensure `runtime.txt` contains: `python-3.12`
2. Push repo to GitHub
3. Set main file to `Home.py` in Streamlit Cloud
4. Redeploy

## Contributing
Open an issue or PR for improvements (tests, new models, data enrichment).

## License
MIT (add LICENSE file if distribution required).
