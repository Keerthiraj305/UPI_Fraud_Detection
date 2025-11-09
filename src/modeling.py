import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.utils import resample

def downsample_train(X, y, random_state=42, ratio=10, min_majority=5000):
    data = X.copy()
    data['_y_'] = y.values
    frauds = data[data['_y_']==1]
    nonfrauds = data[data['_y_']==0]
    n_frauds = len(frauds)
    n_non_sample = min(len(nonfrauds), max(min_majority, n_frauds*ratio))
    nonfrauds_s = resample(nonfrauds, replace=False, n_samples=n_non_sample, random_state=random_state)
    sub = np.random.RandomState(random_state).permutation(np.vstack([frauds.values, nonfrauds_s.values]))
    # Restore to DataFrame
    import pandas as pd
    sub_df = pd.DataFrame(sub, columns=data.columns)
    X_sub = sub_df.drop(columns=['_y_'])
    y_sub = sub_df['_y_'].astype(int)
    return X_sub, y_sub

def smote_train(X, y, random_state=42):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception:
        return None, "imblearn is not installed. Please install imbalanced-learn to use SMOTE."
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X, y), None

def fit_models(preprocessor, X_train, y_train, rf_params=None, use_downsample=False, use_smote=False):
    # optionally transform first for SMOTE; otherwise let pipelines handle it
    X_tr, y_tr = X_train, y_train
    smote_msg = None
    if use_smote:
        # For SMOTE, we need numeric arrays; so we transform with preprocessor first (dense not required if algorithm can accept sparse)
        Xt = preprocessor.fit_transform(X_train)
        # Many oversamplers need dense input
        try:
            import scipy.sparse as sp
            if hasattr(Xt, "toarray"):
                Xt_dense = Xt.toarray() if sp.issparse(Xt) else Xt
            else:
                Xt_dense = Xt
        except Exception:
            Xt_dense = Xt
        sm_result, smote_msg = smote_train(Xt_dense, y_train)
        if sm_result is not None:
            Xt_res, y_res = sm_result
            X_tr, y_tr = Xt_res, y_res
        else:
            # fall back to downsampling
            X_tr, y_tr = downsample_train(X_train, y_train)
            # fit preprocessor on the downsampled raw frame
            preprocessor.fit(X_tr, y_tr)
    elif use_downsample:
        X_tr, y_tr = downsample_train(X_train, y_train)
        preprocessor.fit(X_tr, y_tr)
    else:
        preprocessor.fit(X_train, y_train)

    logreg = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
    rf_defaults = dict(n_estimators=120, max_depth=14, class_weight='balanced', random_state=42, n_jobs=1)
    if rf_params:
        rf_defaults.update(rf_params)
    rf = RandomForestClassifier(**rf_defaults)

    if isinstance(X_tr, np.ndarray):
        # if we already transformed (SMOTE path), build pipelines that skip preprocessor
        from sklearn.base import BaseEstimator, TransformerMixin
        class Identity(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return X
        pre_identity = Identity()
        log_pipe = Pipeline([('pre', pre_identity), ('clf', logreg)])
        rf_pipe = Pipeline([('pre', pre_identity), ('clf', rf)])
        log_pipe.fit(X_tr, y_tr)
        rf_pipe.fit(X_tr, y_tr)
    else:
        log_pipe = Pipeline([('pre', preprocessor), ('clf', logreg)])
        rf_pipe = Pipeline([('pre', preprocessor), ('clf', rf)])
        log_pipe.fit(X_tr, y_tr)
        rf_pipe.fit(X_tr, y_tr)

    return log_pipe, rf_pipe, smote_msg

def evaluate(pipe, X_test, y_test, threshold=0.5) -> Dict[str, Any]:
    proba = pipe.predict_proba(X_test)[:,1]
    y_pred = (proba >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        roc_auc = None
    ap = average_precision_score(y_test, proba)
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
        "classification_report": cr,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "pr_curve": (prec, rec),
        "y_proba": proba,
    }
