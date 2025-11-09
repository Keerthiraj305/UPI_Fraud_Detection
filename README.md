# UPI Fraud Detection — Streamlit App

This repository contains a small Streamlit application and helper modules to train and compare binary fraud-detection models on UPI transaction data. The app trains two candidate models (Logistic Regression and Random Forest), shows evaluation metrics and curves, and lets you save/download trained model pipelines.

## What this project does (high level)
- Loads a CSV of transactions (either uploaded in the UI or by placing it at `data/upi_transactions_2024.csv`).
- Detects a label column automatically (common names like `fraud_flag`, `is_fraud`, `label`, or a binary 0/1 column).
- Drops obvious ID-like columns (high uniqueness and names like `txn_id`, `mobile`, `upi`).
- Builds a preprocessing pipeline: numeric imputation + scaling, categorical imputation + one-hot encoding.
- Splits data with stratification, and trains two pipelines:
	- Logistic Regression pipeline
	- Random Forest pipeline
- Optionally applies imbalance strategies: none, downsample majority, or SMOTE (if `imbalanced-learn` is installed).
- Evaluates models (precision/recall/F1, confusion matrix, ROC AUC, PR/ROC curves).
- Lets you adjust a decision threshold and download trained `.joblib` pipeline files.

## Files and important functions
- `streamlit_app.py` — Streamlit UI and orchestration. This is the app entrypoint.
- `src/pipeline.py` — helpers for data handling and preprocessing: `detect_label`, `drop_id_like`, `column_types`, `make_preprocessor`, and `train_test_split_strat`.
- `src/modeling.py` — (used by the app) contains `fit_models` and `evaluate` helpers that build model pipelines, train, and compute metrics.
- `models/` — created at runtime; contains saved `.joblib` pipeline files when you train and save models.
- `data/upi_transactions_2024.csv` — example/default data path (not required to be present if you upload a CSV in the UI).

## Quickstart — Windows PowerShell (recommended)
1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
# relax execution policy for this session to allow Activate.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Optional: install SMOTE support
pip install imbalanced-learn
```

3) Run the Streamlit app (uses the virtualenv python):

```powershell
# run with venv python to ensure correct environment
python -m streamlit run streamlit_app.py --server.port 8501
```

4) Open the printed URL (usually http://localhost:8501) in your browser. Use the sidebar to upload your CSV (or check the `data/` path), set test size, sampling strategy, and hyperparameters, then press "Train / Retrain models".

## How the code works (technical flow)
1. Data ingestion: the app reads CSV via `pd.read_csv()` either from uploaded file or `data/upi_transactions_2024.csv`.
2. Label detection: `detect_label(df)` looks for common label column names, binary 0/1 columns, or text values containing "fraud".
3. Column cleanup: `drop_id_like()` removes columns that are likely identifiers (very high uniqueness and names that match `ID_TOKENS`).
4. Split: `split_xy()` separates X and y and normalizes textual labels to 0/1.
5. Preprocessing: `make_preprocessor()` returns a `ColumnTransformer` that imputes and scales numeric columns and imputes + one-hot-encodes categorical columns. The code is compatible with multiple scikit-learn versions.
6. Modeling: `fit_models()` (in `src/modeling.py`) constructs pipeline(s) that attach the preprocessor to scikit-learn estimators (LogisticRegression and RandomForest), handles optional downsampling or SMOTE, fits on the training set, and returns fitted pipelines and auxiliary messages.
7. Evaluation: `evaluate()` computes metrics and returns objects used to render metrics and plots in the Streamlit UI.

## Notes, tips and troubleshooting
- If the app errors at OneHotEncoder about `sparse` vs `sparse_output`, recent changes were made to `src/pipeline.py` to be compatible with both older and newer scikit-learn releases.
- If `streamlit` is not found, make sure your venv is activated and run `python -m streamlit run streamlit_app.py` to ensure the correct interpreter is used.
- If SMOTE is selected and `imbalanced-learn` is not installed, the app will warn and continue without SMOTE.
- Large datasets will make training (esp. RandomForest) slower — reduce `n_estimators` in the sidebar for quick iteration.
- If you see PowerShell activation errors, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` (temporary) or run `Activate` from CMD.

## Suggested next improvements
- Add explicit training / inference scripts (non-UI) for headless experiments and CI.
- Add unit tests for `src/pipeline.py` functions (label detection, preprocessing behavior).
- Pin dependency versions in `requirements.txt` for reproducible installs.
- Add a small example dataset or a script to sample the CSV for quick local testing.

## Recommended Python versions
- Python 3.8 — 3.11 are recommended for the listed dependencies and pre-built wheels on Windows.

If you want, I can also:
- Add a one‑click `run_streamlit.ps1` that activates the venv and starts the app and logs output.
- Add the helper utilities discussed earlier (`save_model`, `load_model`, `evaluate_model`) into `src/pipeline.py` and create a small example script showing save/load.

---
Updated README to explain the project, how it works, and how to run it on Windows.
