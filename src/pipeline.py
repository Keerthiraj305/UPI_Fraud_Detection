import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABEL_CANDIDATES = ["fraud_flag","is_fraud","fraud","label","isFraud","is_fraudulent","fraudulent","target"]

ID_TOKENS = ['id','txn_id','transaction_id','mobile','phone','upi','ref','reference']

def detect_label(df: pd.DataFrame) -> str:
    # exact name match (case-insensitive)
    for c in df.columns:
        if c.lower() in [x.lower() for x in LABEL_CANDIDATES]:
            return c
    # binary 0/1 columns
    for c in df.columns:
        s = df[c].dropna()
        if s.isin([0,1]).all() and s.nunique() == 2:
            return c
    # text values containing "fraud"
    for c in df.columns:
        vals = df[c].astype(str).str.lower().unique()
        if any("fraud" in v for v in vals):
            return c
    raise ValueError("Could not detect label column automatically. Please specify it.")

def normalize_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    y = df[label_col]
    if y.dtype == object or y.dtype.name == "category":
        y = y.astype(str).str.lower().map(lambda x: 1 if ('fraud' in x) or (x in ['1','true','yes','y']) else 0)
    return y.astype(int)

def drop_id_like(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    drop_cols = []
    n = len(df)
    for c in df.columns:
        if c == label_col: 
            continue
        uniq_ratio = df[c].nunique() / max(1,n)
        if uniq_ratio > 0.95 and any(tok in c.lower() for tok in ID_TOKENS):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_col]).copy()
    y = normalize_label(df, label_col)
    return X, y

def column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    categorical = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    return numeric, categorical

def make_preprocessor(numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    # Create OneHotEncoder in a way that's compatible with multiple scikit-learn versions.
    # Newer sklearn versions (>=1.2) use `sparse_output`; older versions use `sparse`.
    try:
        # prefer sparse_output (returns sparse matrix)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        # fallback for older sklearn versions
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    cat = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', ohe)])
    pre = ColumnTransformer([('num', num, numeric_cols),
                             ('cat', cat, cat_cols)],
                             remainder='drop', sparse_threshold=0.3)
    return pre

def train_test_split_strat(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
