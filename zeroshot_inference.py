#!/usr/bin/env python
"""
Zero-shot SVM inference
"""

import argparse
import pickle
import numpy as np
from joblib import load as joblib_load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import config
from data_loader import DataLoader

def load_model(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        return joblib_load(path)

def build_feature_matrix(data_input, cfg):
    feats = []
    for item in data_input:
        parts = []
        if cfg.use_target_text and cfg.use_bert:
            parts.append(item[7])                    # 768-dim text BERT
        if cfg.use_target_audio:
            parts.append(item[4])                    # 128-dim audio
        if cfg.use_target_video:
            parts.append(np.mean(item[5], axis=0))   # 2048-dim video
        if cfg.use_context and cfg.use_bert:
            ctx = np.mean(item[8], axis=0) if len(item[8]) else np.zeros(768)
            parts.append(ctx)                        # 768-dim context BERT
        feats.append(np.concatenate(parts))
    return np.vstack(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Trained SVM model (pkl/joblib)")
    parser.add_argument("--config-key", default="t", help="Model config key, e.g. t, ta, tav, etc.")
    args = parser.parse_args()

    cfg = config.CONFIG_BY_KEY[args.config_key]

    print(vars(cfg))  # Optional: useful for debugging configuration settings

    print("Loading dataset ...")
    dl = DataLoader(cfg)
    X = build_feature_matrix(dl.data_input, cfg)
    y = np.array(dl.data_output)

    print(f"X shape: {X.shape}")
    print(f"Nonzero standard deviation columns: {np.sum(X.std(axis=0) > 1e-6)}")
    print(f"Mean pairwise L2 distance: {np.mean(np.linalg.norm(X - X[0], axis=1)):.4f}")

    print(f"Loading trained model from {args.model}")
    clf = load_model(args.model)
    svc = clf.named_steps['svc'] if hasattr(clf, "named_steps") else clf

    # Zero-shot protocol
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("Running zero-shot inference ...")
    y_pred = svc.predict(X_scaled)
    scores = svc.decision_function(X_scaled)

    print("\nZero-shot results on MCSD")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    print("Score percentiles:", np.percentile(scores, [0,25,50,75,100]))
    print("Prediction count:", np.unique(y_pred, return_counts=True))

if __name__ == "__main__":
    main()
