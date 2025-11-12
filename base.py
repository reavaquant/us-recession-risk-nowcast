import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


FILES = {
    "recession": "data/us_recession.csv",
    "unrate": "data/unrate.csv",
    "sentiment": "data/umcsent.csv",
    "indpro": "data/indpro.csv",
    "housing": "data/houst.csv",
    "cpi": "data/cpi.csv",
    "treasury_spread": "data/1030treasury_spreads.csv",
}

H = 6

def _load_series(path, series_name):
    """
    Load a series from a file and return it as a pandas Series.

    Parameters
    ----------
    path : str
        Path to the file containing the series.
    series_name : str
        Name to give to the series.

    Returns
    -------
    pd.Series
        Loaded series.

    Raises
    ------
    ValueError
        If the file at `path` does not contain exactly one column
        of data (in addition to the "observation_date" column).
    """
    data = pd.read_csv(path, parse_dates=["observation_date"])
    value_cols = [c for c in data.columns if c != "observation_date"]
    if len(value_cols) != 1:
        raise ValueError(f"{path} devrait n’avoir qu’une colonne de données")
    series = data.rename(columns={value_cols[0]: series_name}).set_index("observation_date")
    if series_name == "treasury_spread":
        series = series.resample("MS").mean() # resemple to monthly data as it is in daily originally
    return series

def compute_features(files, horizon=H):
    df = pd.concat([_load_series(p, n) for n,p in files.items()], axis=1, join="inner").sort_index()
    df["sentiment"] = df["sentiment"].ffill().shift(1)
    df["target"] = df["recession"].shift(-horizon).astype(float)

    feats = []
    for col in ["indpro","cpi","housing","sentiment"]:
        s = df[col]
        block = pd.DataFrame(index=df.index)
        block[f"{col}_yoy"] = np.log(s).diff(12)   # YoY log change (stable)
        block[f"{col}_m1"] = np.log(s).diff(1)    # 1m log change
        for L in [3,6,12]:
            block[f"{col}_lag{L}"] = s.shift(L)
        feats.append(block)

    for col in ["unrate","treasury_spread"]:
        s = df[col]
        block = pd.DataFrame(index=df.index)
        block[f"{col}_d1"] = s.diff(1)
        block[f"{col}_d12"] = s.diff(12)
        for L in [3,6,12]:
            block[f"{col}_lag{L}"] = s.shift(L)
        feats.append(block)

    X = pd.concat(feats, axis=1).replace([np.inf,-np.inf], np.nan)
    data = pd.concat([X, df["target"]], axis=1).dropna()
    y = data.pop("target").astype(int)
    return data, y

def report_metrics(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    base_rate = y_true.mean()  # baseline for PR-AUC
    print(f"\n=== {name} @ threshold={threshold:.3f} ===")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_prob):.3f}  (baseline={base_rate:.3f})")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=3))

def best_threshold_by_f1(y_true, y_prob):
    """
    Find the best threshold to maximize the F1-score between the true labels, y_true, and the predicted probabilities, y_prob.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_prob : array-like, shape (n_samples,)
        Predicted probabilities.

    Returns
    -------
    thr : float
        The best threshold.
    f1 : float
        The best F1-score.
    """
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r / (p + r + 1e-12))[:-1]  # drop last point (no threshold)
    if len(f1) == 0:
        return 0.5, 0.0
    ix = int(np.argmax(f1))
    return float(thr[ix]), float(f1[ix])

def rolling_cv_scores(model, X, y, n_splits=5, gap=H):
    """
    Compute rolling cross-validation scores for a given model and data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Model to evaluate.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    n_splits : int, optional
        Number of CV folds. Default is 5.
    gap : int, optional
        Gap, in samples, between train and validation. Default is H.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rolling cross-validation scores.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for k, (tr, va) in enumerate(tscv.split(X)):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            rows.append({"fold": k+1, "roc_auc": np.nan, "pr_auc": np.nan, "base_rate": y_va.mean()})
            continue

        m = clone(model)  # IMPORTANT: fresh model per fold

        if isinstance(m, XGBClassifier):
            # Ensure proper metric; give it a validation set for early stopping
            m.set_params(eval_metric="aucpr", early_stopping_rounds=50)
            m.fit(X.iloc[tr], y_tr,
                  eval_set=[(X.iloc[va], y_va)],
                  verbose=False)
        else:
            # Disable any lingering early stopping params if present
            try:
                m.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            m.fit(X.iloc[tr], y_tr)

        p = m.predict_proba(X.iloc[va])[:, 1]
        rows.append({
            "fold": k+1,
            "roc_auc": roc_auc_score(y_va, p),
            "pr_auc": average_precision_score(y_va, p),
            "base_rate": y_va.mean(),
        })
    return pd.DataFrame(rows)

def sliding_test_scores(model, X, y, window=0.2, step=0.05):
    """
    Evaluate a model on a sliding window of the dataset.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate.
    X : pd.DataFrame
        DataFrame containing the feature data.
    y : pd.Series
        Series containing the target data.
    window : float, optional
        Fraction of the dataset to use as the sliding window. Defaults to 0.2.
    step : float, optional
        Fraction of the dataset to use as the step size. Defaults to 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the evaluation metrics for each window.
    """
    n = len(X)
    w = int(n*window)
    s = int(n*step)
    rows = []
    for start in range(0, n - w + 1, s):
        end = start + w
        X_tr, y_tr = X.iloc[:start], y.iloc[:start]
        X_te, y_te = X.iloc[start:end], y.iloc[start:end]

        if start < max(50, 3*H) or y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        i_split = int(len(X_tr) * 0.8)
        X_tr_in, y_tr_in = X_tr.iloc[:i_split], y_tr.iloc[:i_split]
        X_val_in, y_val_in = X_tr.iloc[i_split:], y_tr.iloc[i_split:]

        # --- fit on internal train (clone)
        m = clone(model)
        if isinstance(m, XGBClassifier):
            m.set_params(eval_metric="aucpr", early_stopping_rounds=50)
            m.fit(X_tr_in, y_tr_in,
                  eval_set=[(X_val_in, y_val_in)],
                  verbose=False)
        else:
            try:
                m.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            m.fit(X_tr_in, y_tr_in)

        # --- choose threshold by F1 on internal val
        p_val = m.predict_proba(X_val_in)[:, 1]
        P, R, T = precision_recall_curve(y_val_in, p_val)
        if len(T) == 0:
            thr = 0.5
        else:
            F1 = (2*P*R/(P+R+1e-12))[:-1]
            thr = float(T[int(np.argmax(F1))])

        p_te = m.predict_proba(X_te)[:, 1]
        y_hat = (p_te >= thr).astype(int)
        rows.append({
            "start": X_te.index.min(), "end": X_te.index.max(),
            "positives": int(y_te.sum()),
            "roc_auc": roc_auc_score(y_te, p_te),
            "pr_auc": average_precision_score(y_te, p_te),
            "bal_acc": balanced_accuracy_score(y_te, y_hat),
            "f1": f1_score(y_te, y_hat, zero_division=0),
            "thr": thr,
            "base_rate": y_te.mean(),
        })
    return pd.DataFrame(rows)


def main():
    X, y = compute_features(FILES, horizon=H)
    n = len(X)
    i_tr = int(n * 0.6)
    i_va = int(n * 0.8)
    X_tr, y_tr = X.iloc[:i_tr], y.iloc[:i_tr]
    X_va, y_va = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
    X_te, y_te = X.iloc[i_va:], y.iloc[i_va:]

    print(f"Samples: total={n} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")
    print(f"Positives by split: train={y_tr.sum()}, val={y_va.sum()}, test={y_te.sum()}")


    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5, solver="lbfgs", random_state=42)),
    ])
    pca_logit = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.9, svd_solver="full")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5, solver="lbfgs", random_state=42)),
    ])
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=6, min_samples_leaf=10,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    pos, neg = int((y_tr==1).sum()), int((y_tr==0).sum())
    xgb = XGBClassifier(
        n_estimators=4000, learning_rate=0.02, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=neg/max(pos,1), tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=200,random_state=42
    )

    # 4) Fit on TRAIN; choose threshold on VALIDATION (max F1)
    # Logistic
    logit.fit(X_tr, y_tr)
    p_val_log = logit.predict_proba(X_va)[:, 1]
    thr_log, f1_log = best_threshold_by_f1(y_va, p_val_log)
    print(f"\n[Logit] Validation PR-AUC: {average_precision_score(y_va, p_val_log):.3f} | Best-F1 thr: {thr_log:.3f} (F1={f1_log:.3f})")

    # Logistic + PCA
    pca_logit.fit(X_tr, y_tr)
    p_val_pca = pca_logit.predict_proba(X_va)[:, 1]
    thr_pca, f1_pca = best_threshold_by_f1(y_va, p_val_pca)
    print(f"[Logit+PCA] Validation PR-AUC: {average_precision_score(y_va, p_val_pca):.3f} | Best-F1 thr: {thr_pca:.3f} (F1={f1_pca:.3f})")

    # XGB (early stopping on validation)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_val_xgb = xgb.predict_proba(X_va)[:, 1]
    thr_xgb, f1_xgb = best_threshold_by_f1(y_va, p_val_xgb)
    print(f"[XGB] Validation PR-AUC: {average_precision_score(y_va, p_val_xgb):.3f} | Best-F1 thr: {thr_xgb:.3f} (F1={f1_xgb:.3f})")

    # RF
    rf.fit(X_tr, y_tr)
    p_val_rf = rf.predict_proba(X_va)[:, 1]
    thr_rf, f1_rf = best_threshold_by_f1(y_va, p_val_rf)
    print(f"[RF]  Validation PR-AUC: {average_precision_score(y_va, p_val_rf):.3f} | Best-F1 thr: {thr_rf:.3f} (F1={f1_rf:.3f})")


    report_metrics("Logistic Regression", y_te, logit.predict_proba(X_te)[:, 1], thr_log)
    report_metrics("XGBoost", y_te, xgb.predict_proba(X_te)[:, 1], thr_xgb)
    report_metrics("Random Forest", y_te, rf.predict_proba(X_te)[:, 1], thr_rf)
    report_metrics("Logistic Regression + PCA", y_te, pca_logit.predict_proba(X_te)[:, 1], thr_pca)

    # Robustness: rolling CV on first 80% (train+val)
    X80, y80 = X.iloc[:i_va], y.iloc[:i_va]
    cv_log = rolling_cv_scores(logit, X80, y80, n_splits=5, gap=H)
    cv_xgb = rolling_cv_scores(xgb, X80, y80, n_splits=5, gap=H)
    cv_rf  = rolling_cv_scores(rf, X80, y80, n_splits=5, gap=H)
    cvlogpca = rolling_cv_scores(pca_logit, X80, y80, n_splits=5, gap=H)

    print("\n-- Rolling CV (mean ± std) on first 80% --")
    for name, dfcv in [("Logit", cv_log), ("XGB", cv_xgb), ("RF", cv_rf), ("Logit+PCA", cvlogpca)]:
        m = dfcv[["roc_auc","pr_auc","base_rate"]].mean()
        s = dfcv[["roc_auc","pr_auc","base_rate"]].std()
        print(f"{name}: ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}±{s['pr_auc']:.3f} | base {m['base_rate']:.3f}")

    # Robustness: sliding tests over the full span
    st_log = sliding_test_scores(logit, X, y, window=0.2, step=0.05)
    st_xgb = sliding_test_scores(xgb, X, y, window=0.2, step=0.05)
    st_rf  = sliding_test_scores(rf, X, y, window=0.2, step=0.05)
    st_logpca = sliding_test_scores(pca_logit, X, y, window=0.2, step=0.05)

    def summarize_sliding(name, dfst):
        if len(dfst)==0:
            print(f"{name}: not enough data to run sliding windows.")
            return
        m = dfst[["roc_auc","pr_auc","bal_acc","f1","base_rate"]].mean()
        s = dfst[["roc_auc","pr_auc","bal_acc","f1","base_rate"]].std()
        print(f"{name} (sliding): ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f} | "
              f"PR-AUC {m['pr_auc']:.3f}±{s['pr_auc']:.3f} | "
              f"BalAcc {m['bal_acc']:.3f}±{s['bal_acc']:.3f} | "
              f"F1 {m['f1']:.3f}±{s['f1']:.3f} | base {m['base_rate']:.3f}")

    print("\n-- Sliding window tests (20% window, 5% step) --")
    summarize_sliding("Logit", st_log)
    summarize_sliding("XGB", st_xgb)
    summarize_sliding("RF", st_rf)
    summarize_sliding("Logit+PCA", st_logpca)


if __name__ == "__main__":
    main()
