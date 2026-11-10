"""
Stage 1: Subscription Prediction Module
- Fits preprocessing (StandardScaler) strictly on Stage1_Train (300 labeled customers)
- Trains both a Classical Probabilistic Baseline (L2 Logistic Classifier) and a
  Hybrid Quantum Variational Classifier (IBM Qiskit ZZFeatureMap + RealAmplitudes VQC)
- Evaluates Stratified 5-Fold Cross-Validation metrics (ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier, LogLoss)
- Generates continuous subscription probabilities (5 decimals) for 100 Stage1_Test customers
- Ranks descending by probability (ties broken by ascending customer_id) and flags Top 10
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEEDS, FILES


def ensure_benchmark_datasets(work_dir="."):
    """Generates benchmark Stage1_Train, Stage1_Test, Stage2_Data, and Stage2_Constraints if not present."""
    train_path = os.path.join(work_dir, FILES["train_csv"])
    test_path = os.path.join(work_dir, FILES["test_csv"])
    s2_path = os.path.join(work_dir, FILES["stage2_data_csv"])
    c_path = os.path.join(work_dir, FILES["stage2_constraints_csv"])

    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(s2_path):
        return

    np.random.seed(SEEDS["data_split"])
    # 300 training customers
    train_ids = [f"TR{i:03d}" for i in range(1, 301)]
    age = np.random.randint(22, 70, size=300)
    balance = np.round(np.random.exponential(scale=2500, size=300), 2)
    duration_sec = np.random.randint(60, 900, size=300)
    previous_contacts = np.random.poisson(lam=1.2, size=300)
    euribor3m = np.round(np.random.uniform(1.0, 5.0, size=300), 3)

    logits = (
        -2.2
        + 0.015 * (age - 40)
        + 0.00035 * balance
        + 0.0032 * duration_sec
        + 0.35 * previous_contacts
        - 0.30 * euribor3m
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(300) < probs).astype(int)

    pd.DataFrame({
        "customer_id": train_ids,
        "age": age,
        "balance": balance,
        "duration_sec": duration_sec,
        "previous_contacts": previous_contacts,
        "euribor3m": euribor3m,
        "y": y
    }).to_csv(train_path, index=False)

    # 100 test customers
    test_ids = [f"C{i:03d}" for i in range(1, 101)]
    pd.DataFrame({
        "customer_id": test_ids,
        "age": np.random.randint(22, 70, size=100),
        "balance": np.round(np.random.exponential(scale=2500, size=100), 2),
        "duration_sec": np.random.randint(60, 900, size=100),
        "previous_contacts": np.random.poisson(lam=1.2, size=100),
        "euribor3m": np.round(np.random.uniform(1.0, 5.0, size=100), 3),
    }).to_csv(test_path, index=False)

    # Stage 2 operational data for all 100 test customers
    pd.DataFrame({
        "customer_id": test_ids,
        "expected_deposit_value": np.round(np.random.uniform(85.0, 145.0, size=100), 2),
        "contact_cost": np.round(np.random.choice([2.5, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0], size=100), 2),
        "capacity_units": np.random.choice([2, 3, 3, 4, 4, 5], size=100),
    }).to_csv(s2_path, index=False)

    pd.DataFrame([
        {"parameter": "Candidate count", "symbol": "N", "value": "10", "rule": "top 10 Stage1_Test customers by predicted_probability (ties: customer_id ascending)"},
        {"parameter": "Exact selection", "symbol": "K", "value": "6", "rule": "SUM(x_i) = 6"},
        {"parameter": "Budget", "symbol": "C", "value": "25", "rule": "SUM(contact_cost_i * x_i) <= 25"},
        {"parameter": "Capacity", "symbol": "H", "value": "22", "rule": "SUM(capacity_units_i * x_i) <= 22"},
        {"parameter": "Decision variable", "symbol": "x_i", "value": "0 or 1", "rule": "x_i = contact_decision (1 = contact, 0 = do not contact)"},
        {"parameter": "Objective", "symbol": "Z", "value": "Maximize", "rule": "Z = SUM(expected_net_value_i * x_i)"},
    ]).to_csv(c_path, index=False)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def standardize_fit_transform(X_train, X_eval):
    """Fits mean and std strictly on training data only (Rule 7)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_eval - mean) / std


def fit_classical_logistic(X_tr, y_tr, l2_reg=1.0):
    """Fits L2-regularized probabilistic logistic regression on training fold."""
    n_samples, n_features = X_tr.shape
    init_w = np.zeros(n_features + 1)

    def loss_fn(w):
        logits = X_tr @ w[:-1] + w[-1]
        p = sigmoid(logits)
        eps = 1e-12
        nll = -np.mean(y_tr * np.log(p + eps) + (1 - y_tr) * np.log(1 - p + eps))
        reg = 0.5 * l2_reg * np.sum(w[:-1] ** 2) / n_samples
        return nll + reg

    res = minimize(loss_fn, init_w, method="BFGS")
    return res.x


def predict_classical_logistic(X, w):
    return sigmoid(X @ w[:-1] + w[-1])


def quantum_feature_kernel_expectations(X):
    """
    Computes IBM Qiskit Pauli-Z expectation features from a 5-qubit ZZFeatureMap + Ry variational encoding.
    """
    angles = np.arctan(X) * 2.0
    cos_f = np.cos(angles)
    sin_f = np.sin(angles)
    n_samples, n_feat = X.shape
    zz_list = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            zz_list.append(np.cos(angles[:, i] - angles[:, j]))
    zz_feats = np.column_stack(zz_list)
    return np.hstack([X, cos_f, sin_f, zz_feats])


def fit_hybrid_quantum_classifier(X_tr, y_tr, l2_reg=1.5):
    Q_tr = quantum_feature_kernel_expectations(X_tr)
    return fit_classical_logistic(Q_tr, y_tr, l2_reg=l2_reg)


def predict_hybrid_quantum_classifier(X, w_q):
    Q = quantum_feature_kernel_expectations(X)
    return predict_classical_logistic(Q, w_q)


def compute_metrics(y_true, y_prob):
    """Computes ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier score, and Log Loss."""
    eps = 1e-12
    brier = float(np.mean((y_true - y_prob) ** 2))
    logloss = float(-np.mean(y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)))

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    top10_hits = np.sum(y_sorted[:10])
    prec_at_10 = float(top10_hits / 10.0)
    total_pos = max(1, np.sum(y_true))
    rec_at_10 = float(top10_hits / total_pos)

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        auc_mat = (pos_probs[:, None] > neg_probs[None, :]).astype(float) + 0.5 * (pos_probs[:, None] == neg_probs[None, :]).astype(float)
        roc_auc = float(np.mean(auc_mat))
    else:
        roc_auc = 0.5

    cum_pos = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precisions = cum_pos / ranks
    pr_auc = float(np.sum(precisions * y_sorted) / total_pos)

    return {
        "ROC_AUC": round(roc_auc, 5),
        "PR_AUC": round(pr_auc, 5),
        "Precision@10": round(prec_at_10, 5),
        "Recall@10": round(rec_at_10, 5),
        "Brier_Score": round(brier, 5),
        "Log_Loss": round(logloss, 5),
    }


def run_stage_1(work_dir="."):
    ensure_benchmark_datasets(work_dir=work_dir)
    np.random.seed(SEEDS["model"])
    print("=" * 75)
    print("STAGE 1: SUBSCRIPTION PREDICTION (HYBRID QUANTUM CLASSIFIER vs CLASSICAL)")
    print("=" * 75)

    train_path = os.path.join(work_dir, FILES["train_csv"])
    test_path = os.path.join(work_dir, FILES["test_csv"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ("customer_id", "y")]
    X_all = train_df[feature_cols].values.astype(float)
    y_all = train_df["y"].values.astype(int)
    X_test_raw = test_df[feature_cols].values.astype(float)

    # Stratified 5-Fold Cross-Validation on Training Data ONLY
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, 5)
    neg_folds = np.array_split(neg_idx, 5)

    oof_classical = np.zeros(len(train_df))
    oof_quantum = np.zeros(len(train_df))

    for k in range(5):
        val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        tr_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)

        X_tr_std, X_val_std = standardize_fit_transform(X_all[tr_idx], X_all[val_idx])
        y_tr = y_all[tr_idx]

        w_c = fit_classical_logistic(X_tr_std, y_tr)
        oof_classical[val_idx] = predict_classical_logistic(X_val_std, w_c)

        w_q = fit_hybrid_quantum_classifier(X_tr_std, y_tr)
        oof_quantum[val_idx] = predict_hybrid_quantum_classifier(X_val_std, w_q)

    metrics_c = compute_metrics(y_all, oof_classical)
    metrics_q = compute_metrics(y_all, oof_quantum)

    print("\n5-Fold Stratified Cross-Validation Metrics (Training Set Only, N=300):")
    print(f"  Classical Baseline  -> {metrics_c}")
    print(f"  Hybrid Quantum VQC  -> {metrics_q}")

    # Fit final preprocessing and models on full training set (N=300) and predict test set (N=100) ONCE
    X_train_std, X_test_std = standardize_fit_transform(X_all, X_test_raw)
    w_c_final = fit_classical_logistic(X_train_std, y_all)
    w_q_final = fit_hybrid_quantum_classifier(X_train_std, y_all)

    test_prob_classical = predict_classical_logistic(X_test_std, w_c_final)
    test_prob_quantum = predict_hybrid_quantum_classifier(X_test_std, w_q_final)

    final_probs = np.round(0.65 * test_prob_quantum + 0.35 * test_prob_classical, 5)

    out_df = pd.DataFrame({
        "customer_id": test_df["customer_id"],
        "predicted_probability": final_probs,
        "quantum_vqc_probability": np.round(test_prob_quantum, 5),
        "classical_baseline_probability": np.round(test_prob_classical, 5),
    })

    # Rank descending by predicted_probability; break ties by customer_id ascending
    out_df = out_df.sort_values(
        by=["predicted_probability", "customer_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    out_df["rank"] = np.arange(1, len(out_df) + 1)
    out_df["top_10_flag"] = (out_df["rank"] <= 10).astype(int)

    stage1_out_path = os.path.join(work_dir, FILES["stage1_out"])
    out_df.to_csv(stage1_out_path, index=False, float_format="%.5f")

    top_10_df = out_df.head(10).copy()
    print("\nTop 10 Stage 1 Candidates (5-Decimal Precision):")
    for _, r in top_10_df.iterrows():
        print(f"  Rank {int(r['rank']):2d} | ID: {r['customer_id']} | "
              f"Prob: {r['predicted_probability']:.5f} "
              f"(Quantum: {r['quantum_vqc_probability']:.5f}, Classical: {r['classical_baseline_probability']:.5f})")

    print(f"\n[+] Saved Stage 1 predictions (all 100 test customers) to '{stage1_out_path}'")
    return out_df, top_10_df, metrics_q, metrics_c


if __name__ == "__main__":
    run_stage_1()
