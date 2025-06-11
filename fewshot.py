#!/usr/bin/env python
import argparse
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from config import CONFIG_BY_KEY
from data_loader_fewshot import DataLoader
from data_loader_mcsdfewshot import DataLoader as MCSDLoader

def sample_few_shot_per_class(y, n_shot=5, random_state=42):
    """
    Randomly sample a few-shot subset per class and split the rest as test set.
    """
    np.random.seed(random_state)
    fewshot_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        sel = np.random.choice(idx, min(n_shot, len(idx)), replace=False)
        fewshot_idx.extend(sel)
        test_idx.extend(list(set(idx) - set(sel)))
    return np.array(fewshot_idx), np.array(test_idx)

def run_few_shot_transfer(config_src, config_tgt, data_src, data_tgt, n_shot=5):
    # 1. Extract features and labels
    X_src, y_src = data_src.get_all_features_and_labels()
    X_tgt, y_tgt = data_tgt.get_all_features_and_labels()
    print(f"Source samples: {X_src.shape[0]}, Target samples: {X_tgt.shape[0]}")

    # 2. Few-shot sampling
    fewshot_idx, test_idx = sample_few_shot_per_class(y_tgt, n_shot=n_shot)
    print(f"Few-shot samples: {len(fewshot_idx)}, Test set size: {len(test_idx)}")

    # 3. Construct train and test splits
    X_train = np.vstack([X_src, X_tgt[fewshot_idx]])
    y_train = np.concatenate([y_src, y_tgt[fewshot_idx]])
    X_test = X_tgt[test_idx]
    y_test = y_tgt[test_idx]

    # 4. Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train and evaluate SVM
    clf = svm.SVC(C=config_src.svm_c, gamma="scale", kernel="rbf")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-config-key", type=str, required=True, help="Config key for the source domain")
    parser.add_argument("--tgt-config-key", type=str, required=True, help="Config key for the target domain")
    parser.add_argument("--few-shot", type=int, default=5, help="Number of few-shot samples per class")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_src = CONFIG_BY_KEY[args.src_config_key]
    config_tgt = CONFIG_BY_KEY[args.tgt_config_key]
    # Key modification: change loaders to match your experiment
    data_src = MCSDLoader(config_src)  # MCSD as source domain
    data_tgt = DataLoader(config_tgt)  # MUStARD as target domain
    run_few_shot_transfer(config_src, config_tgt, data_src, data_tgt, n_shot=args.few_shot)
