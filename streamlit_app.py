import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os
from pathlib import Path
from src.pipeline import detect_label, drop_id_like, split_xy, column_types, make_preprocessor, train_test_split_strat
from src.modeling import fit_models, evaluate

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

st.title("🧭 UPI Fraud Detection — Model Comparison")
st.write("Train & compare Logistic Regression and Random Forest on your UPI transactions dataset.")

# Sidebar controls
st.sidebar.header("Configuration")
default_path = Path("data/upi_transactions_2024.csv")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox(f"Use default path ({default_path}) if no upload", value=True)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.30, 0.05)
imbalance_strategy = st.sidebar.selectbox("Class imbalance strategy", ["None","Downsample Majority","SMOTE (needs imblearn)"])
rf_n_estimators = st.sidebar.slider("RF n_estimators", 50, 300, 120, 10)
rf_max_depth = st.sidebar.slider("RF max_depth", 6, 30, 14, 1)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

# Data loading
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_default and default_path.exists():
    df = pd.read_csv(default_path)
elif Path("/mnt/data/upi_transactions_2024.csv").exists():
    df = pd.read_csv("/mnt/data/upi_transactions_2024.csv")
else:
    st.info("Please upload a CSV or place it at data/upi_transactions_2024.csv")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20))

# Detect label
try:
    label_col = detect_label(df)
    st.success(f"Detected label column: **{label_col}**")
except Exception as e:
    st.error(str(e))
    st.stop()

# Drop ID-like columns
df = drop_id_like(df, label_col)

# Split X/y
X, y = split_xy(df, label_col)
st.write("Class distribution:", y.value_counts().to_dict())

# Train/test
X_train, X_test, y_train, y_test = train_test_split_strat(X, y, test_size=test_size)

# Build preprocessor
num_cols, cat_cols = column_types(X)
pre = make_preprocessor(num_cols, cat_cols)

# Training
st.subheader("Train models")
do_train = st.button("Train / Retrain models")
if do_train:
    with st.spinner("Training..."):
        use_downsample = imbalance_strategy == "Downsample Majority"
        use_smote = imbalance_strategy.startswith("SMOTE")
        log_pipe, rf_pipe, smote_msg = fit_models(pre, X_train, y_train,
                                                  rf_params=dict(n_estimators=rf_n_estimators, max_depth=rf_max_depth),
                                                  use_downsample=use_downsample, use_smote=use_smote)
        if smote_msg:
            st.warning(smote_msg)

        # Evaluate both
        log_metrics = evaluate(log_pipe, X_test, y_test, threshold=threshold)
        rf_metrics = evaluate(rf_pipe, X_test, y_test, threshold=threshold)

        # Display metrics
        def show_metrics(name, m):
            st.markdown(f"### {name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("ROC AUC", f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "n/a")
            col2.metric("Avg Precision (PR AUC)", f"{m['average_precision']:.4f}")
            cr = m['classification_report']
            col3.metric("F1 (Fraud=1)", f"{cr.get('1', {}).get('f1-score', 0):.4f}")
            st.write("Confusion matrix:")
            st.write(m["confusion_matrix"])

        c1, c2 = st.columns(2)
        with c1: show_metrics("Logistic Regression", log_metrics)
        with c2: show_metrics("Random Forest", rf_metrics)

        # Plots
        st.subheader("Curves")
        fpr1,tpr1 = log_metrics["roc_curve"]
        fpr2,tpr2 = rf_metrics["roc_curve"]

        fig1 = plt.figure(figsize=(6,5))
        plt.plot(fpr1, tpr1, label="Logistic")
        plt.plot(fpr2, tpr2, label="RandomForest")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves"); plt.legend(); plt.grid(True)
        st.pyplot(fig1)

        pr1 = log_metrics["pr_curve"]
        pr2 = rf_metrics["pr_curve"]
        fig2 = plt.figure(figsize=(6,5))
        plt.plot(pr1[1], pr1[0], label="Logistic")  # recall on x, precision on y
        plt.plot(pr2[1], pr2[0], label="RandomForest")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves"); plt.legend(); plt.grid(True)
        st.pyplot(fig2)

        # Threshold table
        st.subheader("Threshold sweep (fractions)")
        for name, m in [("Logistic Regression", log_metrics), ("Random Forest", rf_metrics)]:
            probs = m["y_proba"]
            import numpy as np
            thr_list = np.linspace(0.05, 0.95, 19)
            rows = []
            from sklearn.metrics import precision_score, recall_score
            for thr in thr_list:
                pred = (probs >= thr).astype(int)
                rows.append({
                    "threshold": round(float(thr),3),
                    "precision_1": float(precision_score(y_test, pred, zero_division=0)),
                    "recall_1": float(recall_score(y_test, pred, zero_division=0)),
                })
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(rows))

        # Save models
        models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
        joblib.dump(log_pipe, models_dir/"logistic_upi_model.joblib")
        joblib.dump(rf_pipe, models_dir/"rf_upi_model.joblib")
        st.success("Saved trained models to ./models/")
        with open(models_dir/"logistic_upi_model.joblib","rb") as f:
            st.download_button("Download Logistic Model", f, file_name="logistic_upi_model.joblib")
        with open(models_dir/"rf_upi_model.joblib","rb") as f:
            st.download_button("Download RandomForest Model", f, file_name="rf_upi_model.joblib")
else:
    st.info("Click **Train / Retrain models** to start training.")
