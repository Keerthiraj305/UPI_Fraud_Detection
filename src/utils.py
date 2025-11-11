import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import sklearn as _sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH_DEFAULT = Path("data/upi_transactions_2025.csv")

FEATURE_COLUMNS = [
    "amount","hour","sender_state","receiver_state","sender_bank","receiver_bank",
    "transaction_type","past_txn_count_7d","past_avg_amount_7d","device_type","is_vpa_verified"
]
LABEL_COLUMN = "fraud_flag"

def load_data(path: Path = DATA_PATH_DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found.")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df

def split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].astype(int)
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num = SimpleImputer(strategy="median")
    try:
        sk_ver = tuple(int(x) for x in _sklearn.__version__.split(".")[:2])
    except Exception:
        sk_ver = (0, 0)

    if sk_ver >= (1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
    pre = ColumnTransformer([("num", num, numeric_cols), ("cat", cat, cat_cols)], remainder="drop", sparse_threshold=0.3)
    return pre

def train_models(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    pre = build_preprocessor(X)

    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=1)

    dt_pipe = Pipeline([("pre", pre), ("clf", dt)])
    rf_pipe = Pipeline([("pre", pre), ("clf", rf)])

    dt_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    results = {}
    for name, pipe in [("Decision Tree", dt_pipe), ("Random Forest", rf_pipe)]:
        proba = pipe.predict_proba(X_test)[:,1]
        pred = (proba >= 0.5).astype(int)
        cr = classification_report(y_test, pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, pred)
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = None
        acc = accuracy_score(y_test, pred)
        results[name] = dict(
            proba=proba, pred=pred, report=cr, cm=cm, roc=(fpr,tpr), pr=(prec,rec), ap=ap, auc=auc, acc=acc
        )

    return (dt_pipe, rf_pipe), (X_test, y_test), results

def feature_schema_example() -> Dict[str, Any]:
    return {
        "amount": 1200.0,
        "hour": 14,
        "sender_state": "KA",
        "receiver_state": "MH",
        "sender_bank": "SBI",
        "receiver_bank": "HDFC",
        "transaction_type": "P2M",
        "past_txn_count_7d": 5,
        "past_avg_amount_7d": 700.0,
        "device_type": "Android",
        "is_vpa_verified": 1
    }