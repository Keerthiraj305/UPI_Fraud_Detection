# UPI Fraud Detection (Streamlit App)

This Streamlit application trains and serves two fraud detection models (Decision Tree & Random Forest) for UPI transactions. It includes an interactive prediction form and a model insights page with evaluation metrics and visualizations.

## Features
- Single transaction prediction (compares Decision Tree & Random Forest probabilities)
- Automatic training on startup
- Metrics: Accuracy, ROC curve & AUC, Precision-Recall curve & Average Precision, Confusion Matrix, Classification Report

## Dataset
Synthetic dataset: `data/upi_transactions_2025.csv` (with an elevated fraud ratio for demonstration).

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
