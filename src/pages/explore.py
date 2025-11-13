import streamlit as st
import matplotlib.pyplot as plt
from utils import load_data, train_models, DATASETS

st.set_page_config(page_title="Model Insights", layout="wide")
st.title("📊 Model Insights & Metrics")
st.caption("Comprehensive performance analysis: Neural Network vs Random Forest with ROC curves, precision-recall metrics, and confusion matrices")

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASETS.keys()),
    help="Choose between different fraud rate datasets for training"
)

@st.cache_resource(show_spinner=True)
def _train(dataset_path):
    df = load_data(dataset_path)
    models, testset, results = train_models(df)
    return models, testset, results

models, testset, results = _train(DATASETS[dataset_choice])
(X_test, y_test) = testset

def plot_roc(name, roc):
    fpr, tpr = roc
    fig = plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name}"); plt.grid(True); plt.legend()
    return fig

def plot_pr(name, pr):
    prec, rec = pr
    fig = plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name}"); plt.grid(True); plt.legend()
    return fig

def plot_cm(cm, title):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    return fig

for name, r in results.items():
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{r['acc']:.4f}")
    c2.metric("ROC AUC", f"{(r['auc'] or 0):.4f}")
    c3.metric("PR AUC (AP)", f"{r['ap']:.4f}")
    f1 = r["report"].get("1", {}).get("f1-score", 0.0)
    c4.metric("F1 (Fraud=1)", f"{f1:.4f}")

    c5, c6 = st.columns(2)
    with c5:
        st.pyplot(plot_roc(name, r["roc"]))
    with c6:
        st.pyplot(plot_pr(name, r["pr"]))

    st.pyplot(plot_cm(r["cm"], f"Confusion Matrix — {name}"))
    st.markdown("---")