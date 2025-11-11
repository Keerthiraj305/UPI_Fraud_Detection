# UPI Fraud Detection (Decision Tree & Random Forest)

Streamlit project with two pages:
- **Home**: enter one transaction, get predictions from Decision Tree & Random Forest.
- **Model Insights**: auto-trains on startup; shows accuracy, ROC AUC, PR AUC, confusion matrix, curves.

### Dataset
Synthetic dataset at `data/upi_transactions_2025.csv` with **~40% fraud**.

### Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```