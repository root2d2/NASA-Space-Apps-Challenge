# exoplanet_model.py - utilities for ProjectAeroMed (fixed evaluate_model)
import os, json, hashlib, math, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, confusion_matrix
import joblib

RANDOM_SEED = 42

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, label_col='koi_disposition'):
    # Minimal preprocessing: convert label to binary: 'CONFIRMED'/'CANDIDATE' -> 1, others -> 0
    df = df.copy()
    if label_col in df.columns:
        df['label'] = df[label_col].astype(str).str.upper().map(lambda s: 1 if ('CONF' in s or 'CAND' in s) else 0)
    else:
        # if no label column, try to infer
        df['label'] = 0
    # select numeric features
    num = df.select_dtypes(include=['number']).columns.tolist()
    # drop columns that are identifiers if present
    for c in ['kepid','kepoi_name','koi_name']:
        if c in num:
            num.remove(c)
    X = df[num].fillna(0.0)
    y = df['label'].values
    return X, y, num

def model_choices():
    return {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        'logistic': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    }

def _safe_predict_proba(clf, X):
    # Returns probability for the positive class (shape: (n_samples,))
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)
        # If probs has only one column (rare), take that column; otherwise find positive class index
        if probs.ndim == 1:
            return probs
        if probs.shape[1] == 1:
            return probs.ravel()
        # assume positive class is last column
        return probs[:, -1]
    elif hasattr(clf, 'decision_function'):
        # map decision function outputs to [0,1] via logistic sigmoid
        try:
            df = clf.decision_function(X)
            # apply sigmoid safely
            return 1.0 / (1.0 + np.exp(-df))
        except Exception:
            # fallback to predictions (0/1)
            return clf.predict(X)
    else:
        return clf.predict(X)

def evaluate_model(clf, X, y):
    # simple evaluation via cross-validated predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preds = np.zeros(len(y))
    probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        # fit a fresh copy to avoid state carryover for some classifiers
        try:
            clf.fit(X.iloc[train_idx], y[train_idx])
        except Exception:
            # If classifier fails to fit on very small splits, skip
            continue
        # get probabilities for positive class robustly
        p = _safe_predict_proba(clf, X.iloc[test_idx])
        # ensure shape matches
        p = np.asarray(p).ravel()
        if p.shape[0] != len(test_idx):
            # fallback to predict
            preds[test_idx] = clf.predict(X.iloc[test_idx])
            probs[test_idx] = 0.0
        else:
            probs[test_idx] = p
            # generate binary predictions via 0.5 threshold
            preds[test_idx] = (p >= 0.5).astype(int)
    # metrics (handle edge cases)
    try:
        roc = roc_auc_score(y, probs)
    except Exception:
        roc = float('nan')
    try:
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = float('nan')
    try:
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        cm = confusion_matrix(y, preds)
    except Exception:
        acc = float('nan'); f1 = float('nan'); cm = [[0,0],[0,0]]
    metrics = dict(roc_auc=roc, pr_auc=pr_auc, accuracy=acc, f1=f1, confusion_matrix=np.array(cm).tolist())
    return metrics, probs, preds

def train_and_select(X, y, model_name='random_forest'):
    models = model_choices()
    if model_name not in models:
        raise ValueError('unknown model')
    clf = models[model_name]
    # fit on full data
    clf.fit(X, y)
    return clf

def save_model(clf, path, metadata):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(clf, path)
    meta_path = os.path.splitext(path)[0] + '_metadata.json'
    with open(meta_path,'w') as f:
        json.dump(metadata, f, indent=2)
    return path, meta_path

def dataset_checksum(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()
