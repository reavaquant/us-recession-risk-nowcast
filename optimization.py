import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve
)

from base import compute_features, FILES, H

def best_threshold_by_f1(y_true, y_prob):
    P, R, T = precision_recall_curve(y_true, y_prob)
    if len(T) == 0:
        return 0.5, 0.0
    F1 = (2 * P * R / (P + R + 1e-12))[:-1]
    ix = int(np.argmax(F1))
    return float(T[ix]), float(F1[ix])

def time_cv_with_positives(X, y, n_splits=5, gap=H, min_pos=1):
    """Return a list of (train_idx, val_idx) where both splits contain positives."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    splits = []
    for tr, va in tscv.split(X):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if (y_tr.nunique() == 2) and (y_va.nunique() == 2) and (y_tr.sum() >= min_pos) and (y_va.sum() >= min_pos):
            splits.append((tr, va))
    if not splits:
        raise RuntimeError("No valid CV folds contain positives in train and validation.")
    return splits

def report_metrics(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    base = y_true.mean()
    print(f"\n=== {name} @ thr={threshold:.3f} ===")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.3f}  (baseline={base:.3f})")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=3))

def main(n_iter=60, random_state=42):
    X, y = compute_features(FILES, horizon=H)
    n = len(X)
    i_tr = int(n * 0.6)
    i_va = int(n * 0.8)
    X_tr, y_tr = X.iloc[:i_tr], y.iloc[:i_tr]
    X_va, y_va = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
    X_te, y_te = X.iloc[i_va:], y.iloc[i_va:]

    print(f"Samples: total={n} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")
    print(f"Positives by split: train={y_tr.sum()}, val={y_va.sum()}, test={y_te.sum()}")

    # RF search space + time-aware inner CV (TRAIN only)
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    rf_space = {
        "n_estimators": randint(300, 2000),
        "max_depth": randint(3, 12),
        "min_samples_leaf": randint(1, 40),
        "min_samples_split": randint(2, 40),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }
    cv_splits = time_cv_with_positives(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), n_splits=5, gap=H)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_space,
        n_iter=n_iter,
        scoring="average_precision",   
        cv=cv_splits,
        refit=True,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise"
    )

    print("\n[RF Tuning]")
    search.fit(X_tr, y_tr)

    print("\n[RF Tuning] Best params:")
    print(search.best_params_)
    print(f"[RF Tuning] Best CV PR-AUC: {search.best_score_:.3f}")

    # Freeze tuned model, choose threshold on VALIDATION, evaluate on TEST
    rf_best = search.best_estimator_

    p_val = rf_best.predict_proba(X_va)[:, 1]
    thr, f1_val = best_threshold_by_f1(y_va, p_val)
    print(f"\n[RF Tuning] Validation PR-AUC: {average_precision_score(y_va, p_val):.3f} | "
          f"Best-F1 thr: {thr:.3f} (F1={f1_val:.3f})")

    p_test = rf_best.predict_proba(X_te)[:, 1]
    report_metrics("Random Forest (tuned)", y_te, p_test, thr)

    try:
        pd.DataFrame(search.cv_results_).to_csv("rf_tuning_cv_results.csv", index=False)
        importances = pd.Series(rf_best.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
        importances.to_csv("rf_tuned_feature_importance.csv")
        print("\nSaved: rf_tuning_cv_results.csv, rf_tuned_feature_importance.csv")
    except Exception as e:
        print(f"(Skip saving artifacts) {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=60, help="Number of random search iterations")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    main(n_iter=args.n_iter, random_state=args.seed)
